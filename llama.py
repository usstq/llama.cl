import numpy as np
import sys, os
import argparse
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

def make_mha(query_states, key_states, value_states,
             kv_cache, beam_table, attention_mask,
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
    kv_seq_len = attention_mask.shape[-1]
    kv_seq_len0 = kv_seq_len - q_len

    def rope_embedd(x):
        half_rotary_dim = rotary_dim//2
        for k in range(q_len):
            position_idx = kv_seq_len0 + k
            for i0 in range(half_rotary_dim):
                i1 = i0 + half_rotary_dim
                xita = (inv_freq[i0] * position_idx)
                vcos = math.cos(xita)
                vsin = math.sin(xita)
                y0 = vcos * x[:, :, k, i0] - vsin * x[:, :, k, i1]
                y1 = vsin * x[:, :, k, i0] + vcos * x[:, :, k, i1]
                x[:, :, k, i0] = y0
                x[:, :, k, i1] = y1

    rope_embedd(query_states)
    rope_embedd(key_states)

    # kv_cache [2 * n_layers, batch, n_head, max_kv_len, head_size]
    kv_cache[2*layer_idx + 0, :, :, kv_seq_len0:kv_seq_len, :] = key_states
    kv_cache[2*layer_idx + 1, :, :, kv_seq_len0:kv_seq_len, :] = value_states

    # us beam_idx to gather(reorder kv cache), skipped in greedy case
    key_states = kv_cache[2*layer_idx + 0, :, :, :kv_seq_len, :]
    value_states = kv_cache[2*layer_idx + 1, :, :, :kv_seq_len, :]

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    attn_weights = attn_weights + attention_mask

    # apply causal mask, only the last q-token can access all kv-cache
    # [batch, num_heads, q_len ,kv_len] 
    for k in range(q_len-1):
        attn_weights[:, :, k, (k - q_len + 1):] = torch.finfo(torch.float32).min

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    return attn_output


def make_layer(configs, consts, layer_idx, hidden_states, attn_mask, kv_cache, beam_table):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'model.layers.self_attn'
    # layerNorm operation
    input_layernorm = make_rms_norm('model.layers.input_layernorm', hidden_states, consts['layers'][layer_idx], configs['rms_norm_eps'], name_suffix)

    q = make_fc('model.layers.self_attn.q_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    k = make_fc('model.layers.self_attn.k_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    v = make_fc('model.layers.self_attn.v_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)

    attn_output = make_mha(q, k, v, kv_cache, beam_table, attn_mask,
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
def model_forward(configs, consts, input_ids, attn_mask, kv_cache, beam_table):
    inputs_embeds = F.embedding(input_ids, consts['model.embed_tokens.weight'])
    hidden_states = inputs_embeds
    for i in range(configs['layer_num']):
        hidden_states = make_layer(configs, consts, i, hidden_states, attn_mask, kv_cache, beam_table)

    final_layernorm = make_rms_norm('model.norm', hidden_states, consts, configs['rms_norm_eps'])
    logits = make_fc('lm_head', final_layernorm, consts)
    return logits

#=================================================================
# simple greedy pipeline using model_forward
def simple_pipeline(hf_model_path):
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

    batch_size = 1
    max_kv_len = 2048
    kv_cache = torch.zeros(configs["layer_num"] * 2, batch_size, configs["head_num"], max_kv_len, configs["head_size"], dtype=torch.float32)
    position_id = 0
    beam_table = np.zeros([batch_size, max_kv_len], dtype=np.int32)
    for b in range(batch_size):
        beam_table[b,:] = b
    attn_mask = torch.zeros(batch_size, 0, dtype=torch.float32)

    with torch.no_grad():
        while True:
            # attn_mask : [batch, query_len+past_len]
            print("\033[0;32m")
            try:
                prompt = input(">")
            except EOFError:
                break
            inputs = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt", padding=True, return_token_type_ids=False)
            input_ids = inputs["input_ids"]
            # zero means valid, np.finfo(np.float32).min means invalid(padding-part)
            _amask = (1 - inputs["attention_mask"]) * torch.finfo(torch.float32).min
            attn_mask = torch.cat([attn_mask, _amask], dim=-1)

            # logits    : [batch, q_len, vocab_size]
            logits = model_forward(configs, consts, input_ids, attn_mask, kv_cache, beam_table)

            # only the last token
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            print("\033[0;33m")
            while tokenizer.eos_token_id not in next_tokens:
                next_tokens = next_tokens.reshape(batch_size, 1)
                input_ids = next_tokens

                streamer.put(next_tokens)

                attn_mask = torch.cat([attn_mask, torch.zeros(batch_size, 1, dtype=torch.float32)], dim=-1)

                logits = model_forward(configs, consts, input_ids, attn_mask, kv_cache, beam_table)
                next_tokens = torch.argmax(logits, dim=-1)

            streamer.put(next_tokens)
            streamer.end()
            print("\033[00m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--org_model_path', type=str, nargs='?', default='/home/llm_irs/models_original/llama-2-7b-chat/pytorch/')
    parser.add_argument('--quant_type', type=str, nargs='?', default='')
    args = parser.parse_args()
    simple_pipeline(args.org_model_path)
