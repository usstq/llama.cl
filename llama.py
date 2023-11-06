import numpy as np
import sys, os
import argparse
import time

import time
import math
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoTokenizer, TextStreamer

def get_params_from_model(path):
    print(f'extracting from model "{path}"...')
    beg = time.time()
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to('cpu').eval()

    assert(hf_model.config.num_key_value_heads == hf_model.config.num_attention_heads)
    assert(hf_model.config.hidden_act in ['silu'])
    assert(hf_model.config.rope_scaling is None)

    configs = {
        'layer_num': hf_model.config.num_hidden_layers,
        'head_num': hf_model.config.num_attention_heads,
        'head_size': hf_model.config.hidden_size // hf_model.config.num_attention_heads,
        'hidden_size': hf_model.config.hidden_size,
        'max_position_embeddings': hf_model.config.max_position_embeddings,
        'rotary_dims': int(hf_model.config.hidden_size // hf_model.config.num_attention_heads),
        #'gelu_mode': hf_model.config.hidden_act,
        #'intermediate_size': hf_model.config.intermediate_size,
        #'num_key_value_heads': hf_model.config.num_key_value_heads,
        'rms_norm_eps': hf_model.config.rms_norm_eps,
    }

    consts = {
        'model.embed_tokens.weight': hf_model.model.embed_tokens.weight,
        'model.norm.weight': hf_model.model.norm.weight,
        'lm_head.weight': hf_model.lm_head.weight,
        'lm_head.bias': hf_model.lm_head.bias,
        'layers': [
            {
                'model.layers.input_layernorm.weight': l.input_layernorm.weight,
                'model.layers.post_attention_layernorm.weight': l.post_attention_layernorm.weight,
                'model.layers.self_attn.q_proj.bias': l.self_attn.q_proj.bias,
                'model.layers.self_attn.q_proj.weight': l.self_attn.q_proj.weight,
                'model.layers.self_attn.k_proj.bias': l.self_attn.k_proj.bias,
                'model.layers.self_attn.k_proj.weight': l.self_attn.k_proj.weight,
                'model.layers.self_attn.v_proj.bias': l.self_attn.v_proj.bias,
                'model.layers.self_attn.v_proj.weight': l.self_attn.v_proj.weight,
                'model.layers.self_attn.o_proj.bias': l.self_attn.o_proj.bias,
                'model.layers.self_attn.o_proj.weight': l.self_attn.o_proj.weight,
                'model.layers.mlp.gate_proj.bias': l.mlp.gate_proj.bias,
                'model.layers.mlp.gate_proj.weight': l.mlp.gate_proj.weight,
                'model.layers.mlp.up_proj.bias': l.mlp.up_proj.bias,
                'model.layers.mlp.up_proj.weight': l.mlp.up_proj.weight,
                'model.layers.mlp.down_proj.bias': l.mlp.down_proj.bias,
                'model.layers.mlp.down_proj.weight': l.mlp.down_proj.weight,
            } for l in hf_model.model.layers
        ],
    }
    cost = time.time() - beg
    print(f'extracting done, cost {cost:.2f} seconds.\nmodel configs:')
    for k, v in configs.items():
        print(f'	{k}: {v}')
    return configs, consts


def make_fc(key, input, consts, name_suffix=''):
    return F.linear(input, consts[f'{key}.weight'], consts[f'{key}.bias'])

def make_rms_norm(key, input, consts, variance_epsilon, name_suffix=''):
    weights = consts[f'{key}.weight']
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + variance_epsilon)
    return weights * input.to(input_dtype)


inv_freq = None

class KVCache:
    def __init__(self, configs, batch_size, max_kv_len, verbose) -> None:
        self.cache = torch.zeros(configs["layer_num"] * 2, batch_size, configs["head_num"], max_kv_len, configs["head_size"], dtype=torch.float32)
        self.mask = torch.zeros(max_kv_len,  dtype=torch.int32)
        self.max_kv_len = max_kv_len
        self.position_id_next = 0
        self.slots = []
        self.cur_slot = 0
        self.cur_kv_len = 0
        self.beam_table = torch.zeros(batch_size, max_kv_len, dtype=torch.int32)
        for b in range(batch_size):
            self.beam_table[b,:] = b
        
        self.verbose = verbose

    # the early stop check is vital to guarantee meaningful response
    # because instruction (surrounded by [INST][/INST]) is required
    # to produce meaningful response. 
    def prepare(self, query_len, sentence_id, fail_on_sid):
        # do we have cache entry can be overwriten ?
        # mask[i]=  0 means no valid content
        # mask[i]= -1 means system message
        assert(query_len > 0)
        if self.verbose:
            print(f"qlen={query_len}, sid={sentence_id}, slot={self.cur_slot}")

        cur_slot = self.cur_slot
        self.slots = []
        for k in range(query_len):
            # skip cache slot reserved for system message
            while self.mask[cur_slot] < 0:
                cur_slot = (cur_slot + 1) % self.max_kv_len

            # (sentence_id - 1) is sentence of instruct which
            # must be part of valie context for meaningful response
            if self.mask[cur_slot] != 0 and self.mask[cur_slot] == fail_on_sid:
                if self.verbose:
                    print(f" return false on self.mask[{cur_slot}] = {self.mask[cur_slot]}")
                return False

            self.slots.append(cur_slot)
            cur_slot = (cur_slot + 1) % self.max_kv_len
        self.cur_slot = cur_slot

        # update the mask
        for slot in self.slots:
            self.mask[slot] = sentence_id

        # update position_id
        self.position_id = self.position_id_next
        self.position_id_next = self.position_id + query_len

        # update total kv length
        self.cur_kv_len += query_len
        if self.cur_kv_len > self.max_kv_len:
            self.cur_kv_len = self.max_kv_len

        if self.verbose:
            smask= "".join([str(int(self.mask[i])) if self.mask[i] >= 0 else "N" for i in range(self.mask.shape[0])])
            sslots = ",".join(str(s) for s in self.slots)
            print(f"\t{smask}")
            print(f"\t{sslots}")

        return True

    def update_cache(self, layer_idx, key_states, value_states):
        q_len = key_states.shape[2]
        for k in range(q_len):
            self.cache[2*layer_idx + 0, :, :, self.slots[k], :] = key_states[:, :, k, :]
            self.cache[2*layer_idx + 1, :, :, self.slots[k], :] = value_states[:, :, k, :]

        # us beam_idx to gather(reorder kv cache), skipped in greedy case
        return self.cache[2*layer_idx + 0, :, :, :self.cur_kv_len, :], self.cache[2*layer_idx + 1, :, :, :self.cur_kv_len, :]

def make_mha(query_states, key_states, value_states, kv_cache,
             layer_idx, rotary_dim, hidden_size, num_heads, name):
    global inv_freq
    head_dim = hidden_size//num_heads
    num_kv_heads = num_heads
    # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/models/llama/modeling_llama.py#L331
    # query_states : B, L, H*S
    bsz, q_len, _ = query_states.size()
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # q/k/v states : [batch, nHead, q_len, head_dim]

    # derive total kv length from attn (has limitation)
    # apply_rotary_pos_emb to key_states/value_states    
    def rope_embedd(x):
        half_rotary_dim = rotary_dim//2
        for k in range(q_len):
            cur_position_id = kv_cache.position_id + k
            for i0 in range(half_rotary_dim):
                i1 = i0 + half_rotary_dim
                xita = (inv_freq[i0] * cur_position_id)
                vcos = math.cos(xita)
                vsin = math.sin(xita)
                y0 = vcos * x[:, :, k, i0] - vsin * x[:, :, k, i1]
                y1 = vsin * x[:, :, k, i0] + vcos * x[:, :, k, i1]
                x[:, :, k, i0] = y0
                x[:, :, k, i1] = y1

    rope_embedd(query_states)
    rope_embedd(key_states)

    # kv_cache is a circular buffer, and tokens should be overwritten in word boundary

    key_states, value_states = kv_cache.update_cache(layer_idx, key_states, value_states)

    kv_seq_len = kv_cache.cur_kv_len
    kv_mask = kv_cache.mask[:kv_seq_len]

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    # mask out attn weight for kv-tokens whose [kv_cache_mask == 0]
    # attn_weights[:, :, :, kv_mask==0] = torch.finfo(torch.float32).min

    # apply causal mask, so:
    #    q-token[q_len-1] can use all kv-tokens
    #    q-token[q_len-2] can use all kv-tokens except the last one
    #    q-token[q_len-3] can use all kv-tokens except the last two
    #    q-token[k] can use all kv-tokens except the last (q_len - 1 - k)
    #    ....
    # [batch, num_heads, q_len ,kv_len] 
    for k in range(q_len-1):
        tokens_to_remove = (q_len - 1 - k)
        for d in range(tokens_to_remove):
            pos = kv_cache.slots[q_len - 1 - d]
            attn_weights[:, :, k, pos] = torch.finfo(torch.float32).min

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output


def make_layer(configs, consts, layer_idx, hidden_states, kv_cache):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'model.layers.self_attn'
    # layerNorm operation
    input_layernorm = make_rms_norm('model.layers.input_layernorm', hidden_states, consts['layers'][layer_idx], configs['rms_norm_eps'], name_suffix)

    q = make_fc('model.layers.self_attn.q_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    k = make_fc('model.layers.self_attn.k_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    v = make_fc('model.layers.self_attn.v_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)

    attn_output = make_mha(q, k, v, kv_cache,
                           layer_idx, configs['rotary_dims'], configs['hidden_size'], configs['head_num'],
                           name=f'{name_prefix}.mha{name_suffix}')

    attn_output = make_fc('model.layers.self_attn.o_proj', attn_output, consts['layers'][layer_idx], name_suffix)

    attn_output = hidden_states + attn_output
    post_attention_layernorm = make_rms_norm('model.layers.post_attention_layernorm', attn_output, consts['layers'][layer_idx], configs['rms_norm_eps'], name_suffix)

    def mlp(states):
        gate_proj = make_fc('model.layers.mlp.gate_proj', states, consts['layers'][layer_idx], name_suffix)
        silu = F.silu(gate_proj)
        up_proj = make_fc('model.layers.mlp.up_proj', states, consts['layers'][layer_idx], name_suffix)
        mul = silu * up_proj
        down_proj = make_fc('model.layers.mlp.down_proj', mul, consts['layers'][layer_idx], name_suffix)
        return down_proj

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = attn_output + mlp_output
    return output

#=================================================================
# input_ids  : [batch, query_len]
# kv_cache   : [2 * n_layers, batch, n_head, max_kv_len, head_size]
# beam_table : [batch, max_kv_len]
# attn_mask  : [batch, query_len+past_len]
def model_forward(configs, consts, input_ids, kv_cache):
    inputs_embeds = F.embedding(input_ids, consts['model.embed_tokens.weight'])
    hidden_states = inputs_embeds
    for i in range(configs['layer_num']):
        hidden_states = make_layer(configs, consts, i, hidden_states, kv_cache)

    final_layernorm = make_rms_norm('model.norm', hidden_states, consts, configs['rms_norm_eps'])
    logits = make_fc('lm_head', final_layernorm, consts)
    return logits

#=================================================================
# simple greedy pipeline using model_forward
def simple_chat_pipeline(hf_model_path, max_kv_len, system_message, verbose):
    global inv_freq
    print(f"load Tokenizer from {hf_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token_id
    tokenizer.padding_side = "left"             # pad to left

    streamer = TextStreamer(tokenizer)

    print(f"load config/weight from HF model {hf_model_path} ...")
    configs, consts = get_params_from_model(hf_model_path)

    rope_base = 10000
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, configs["rotary_dims"], 2).float().to("cpu") / configs["rotary_dims"]))

    if max_kv_len > configs["max_position_embeddings"]:
        max_kv_len = configs["max_position_embeddings"]
    print(f"max_kv_len = {max_kv_len}")

    batch_size = 1
    kv_cache = KVCache(configs, batch_size, max_kv_len, verbose)

    next_tokens = None

    sentence_id = 0

    with torch.no_grad():
        sys_msg = system_message
        while True:
            if sys_msg:
                # sentence id <0 is reserved, need to skip when update kv-cache 
                inputs = tokenizer(f"[INST] <<SYS>> {sys_msg} <</SYS>> [/INST]", return_tensors="pt", padding=True, return_token_type_ids=False)
                input_ids = inputs["input_ids"]
                assert(kv_cache.prepare(input_ids.shape[1], -1, -1))
                logits = model_forward(configs, consts, input_ids, kv_cache)
                sys_msg = None

            print("\033[0;32m")
            try:
                prompt = input(">")
            except EOFError:
                break

            # kv-segment for question/instruction part
            sentence_id += 1

            inputs = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt", padding=True, return_token_type_ids=False)
            input_ids = inputs["input_ids"]

            # append last predicted token(usually it's EOS)
            if next_tokens is not None:
                input_ids = torch.cat((next_tokens, input_ids), dim=1)
            assert(kv_cache.prepare(input_ids.shape[1], sentence_id, sentence_id))

            # logits    : [batch, q_len, vocab_size]
            first_tok_latency = time.time()
            logits = model_forward(configs, consts, input_ids, kv_cache)
            first_tok_latency = time.time() - first_tok_latency

            # only the last token in instruct predict the next
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1).reshape(batch_size, 1)

            # sentence_id for response
            sentence_id += 1
            print("\033[0;33m")
            streamer.put(next_tokens)
            second_tok_count = 0
            second_tok_latency = time.time()
            early_stop = None
            while tokenizer.eos_token_id not in next_tokens:
                if not kv_cache.prepare(1, sentence_id, sentence_id - 1):
                    early_stop = "...(Early stop before instruction gets overwritten)"
                    break

                input_ids = next_tokens
                logits = model_forward(configs, consts, input_ids, kv_cache)

                second_tok_count += 1
                next_tokens = torch.argmax(logits, dim=-1).reshape(batch_size, 1)
                streamer.put(next_tokens)
            second_tok_latency = 0 if (second_tok_count == 0) else ((time.time() - second_tok_latency) / second_tok_count)
            streamer.end()
            if early_stop:
                print("\033[0;31m", early_stop)

            print("\033[0;90m", f" position_id: {kv_cache.position_id}  latency: {first_tok_latency*1e3:.2f} ms + {second_tok_latency*1e3:.2f}ms x {second_tok_count}")
            print("\033[00m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--org_model_path', type=str, nargs='?', default='/home/llm_irs/models_original/llama-2-7b-chat/pytorch/')
    parser.add_argument('--sys', type=str, default=None)
    parser.add_argument('--kv-len', type=int, default=128)
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--quant_type', type=str, nargs='?', default='')
    args = parser.parse_args()
    simple_chat_pipeline(args.org_model_path, args.kv_len, args.sys, args.verbose)
