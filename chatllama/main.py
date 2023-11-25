import sys, os
import argparse
import time
import pickle
import psutil
import json
import time
import math

import torch
import torch.nn.functional as F
from torch import nn

from . import c_ext

from transformers import AutoTokenizer, TextStreamer

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append("C:/Users/tingqian/Syncplicity/LTQ/Code/llama.cl/ops/build/Release/")

import numpy
import llmops

# for our purpose:
#    OP concept is nothing but association of const data with the forward method
# usually the data is tensor and forward method is operating tensor
# but in general:
#   - the data can also have complex data structure which is not a typical tensor
#   - the operation can also be complex/composite operations (fused)
# graph concept is trying to formulate the relation ships between OPs, but
# it's limited to tensor flowing between OPs. for more general case, program
# can do it better.
# 
# in LLama case, we can use OP as building blocks, and directly describe the
# model topology in programing languages like python/C++ w/o the help of graph
#
# torch has already provided a lot of OP/functional helps to describe the topology
# and torch has no general graph concept since python is used to describe the
# topology/algorithm in a more generic & dynamic way.
#
# thus we also can turn python into C++ for the same logic, w/o the need for graph
# concept. we just need to make sure basic building blocks are consistent

class OP_rms_norm:
    def __init__(self, weight, variance_epsilon) -> None:
        self.weight = torch.clone(weight)
        self.variance_epsilon = variance_epsilon
        pass

    def __call__(self, input):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * input.to(input_dtype)
        return output

    def __repr__(self):
        return f"OP_rms_norm(weight: {self.weight.shape}{self.weight.dtype}, esp:{self.variance_epsilon})"

class OP_fc_f32:
    def __init__(self, linear) -> None:
        # weight.shape : [N, K]
        self.bias = torch.clone(linear.bias) if linear.bias else None
        self.weight = torch.clone(linear.weight)

    def __call__(self, input):
        assert(len(input.shape) == 3)
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return f"OP_fc_f32(weight: {self.weight.shape}{self.weight.dtype}" + ")" if self.bias is None else f", bias: {self.bias.shape}{self.bias.dtype})"

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
        self.slots = torch.empty(query_len, dtype=torch.int32)
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

            self.slots[k] = cur_slot
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

class OP_mha:
    def __init__(self, layer_idx, rotary_dims, hidden_size, num_heads) -> None:
        self.rotary_dims = rotary_dims
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size//self.num_heads
        self.layer_idx = layer_idx

    def __call__(self, query_states, key_states, value_states, kv_cache, inv_freq):
        num_kv_heads = self.num_heads
        # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/models/llama/modeling_llama.py#L331
        # query_states : B, L, H*S
        bsz, q_len, _ = query_states.size()
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)

        # q/k/v states : [batch, nHead, q_len, head_dim]

        # derive total kv length from attn (has limitation)
        # apply_rotary_pos_emb to key_states/value_states    
        def rope_embedd(x):
            half_rotary_dim = self.rotary_dims//2
            for k in range(q_len):
                cur_position_id = kv_cache.position_id + k

                # better for python
                xita = inv_freq * cur_position_id
                vcos = torch.cos(xita)
                vsin = torch.sin(xita)
                x0 = x[:, :, k, :half_rotary_dim]
                x1 = x[:, :, k, half_rotary_dim:]
                y0 = vcos * x0 - vsin * x1
                y1 = vsin * x0 + vcos * x1
                x[:, :, k, :half_rotary_dim] = y0
                x[:, :, k, half_rotary_dim:] = y1

                ## better for C++
                #for i0 in range(half_rotary_dim):
                #    i1 = i0 + half_rotary_dim
                #   xita = (inv_freq[i0] * cur_position_id)
                #    vcos = math.cos(xita)
                #    vsin = math.sin(xita)
                #    y0 = vcos * x[:, :, k, i0] - vsin * x[:, :, k, i1]
                #    y1 = vsin * x[:, :, k, i0] + vcos * x[:, :, k, i1]
                #    x[:, :, k, i0] = y0
                #    x[:, :, k, i1] = y1

        rope_embedd(query_states)
        rope_embedd(key_states)

        # kv_cache is a circular buffer, and tokens should be overwritten in word boundary
        # key_states, value_states = kv_cache.update_cache(self.layer_idx, key_states, value_states)
        #
        #  cache: [num_layers*2, B, H, length_max, S]
        #  slots: [q_len]   where ith query token should be placed cache
        #  
        #
        for k in range(q_len):
            kv_cache.cache[2*self.layer_idx + 0, :, :, kv_cache.slots[k], :] = key_states[:, :, k, :]
            kv_cache.cache[2*self.layer_idx + 1, :, :, kv_cache.slots[k], :] = value_states[:, :, k, :]

        # us beam_idx to gather(reorder kv cache), skipped in greedy case
        key_states = kv_cache.cache[2*self.layer_idx + 0, :, :, :kv_cache.cur_kv_len, :]
        value_states = kv_cache.cache[2*self.layer_idx + 1, :, :, :kv_cache.cur_kv_len, :]

        #kv_seq_len = kv_cache.cur_kv_len
        #kv_mask = kv_cache.mask[:kv_seq_len]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # mask out attn weight for kv-tokens whose [kv_cache_mask == 0]
        # attn_weights[:, :, :, kv_mask==0] = torch.finfo(torch.float32).min

        # apply causal mask, so:
        #    q-token[q_len-1] can use all kv-tokens
        #    q-token[q_len-2] can use all kv-tokens except the last one
        #    q-token[q_len-3] can use all kv-tokens except the last two
        #    q-token[k] can use all kv-tokens except the last (q_len - 1 - k)
        #    ....
        # [batch, num_heads, q_len ,kv_len] 

        #if self.layer_idx == 0:
        #    print(f"==={kv_cache.slots}")

        for k in range(q_len-1):
            pos = torch.arange(start=(k + 1), end=q_len, step=1, dtype = torch.int32)
            pos = kv_cache.slots[pos]

            if self.layer_idx == 0:
                print(f"k={k} pos={pos}")
            attn_weights[:, :, k, pos] = torch.finfo(torch.float32).min

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return attn_output

    def __repr__(self):
        return f"OP_mha(layer_idx:{self.layer_idx}, hidden_size:{self.hidden_size}, rotary_dims:{self.rotary_dims}, num_heads:{self.num_heads}, head_dim:{self.head_dim})"


#=========================================================================
# llmops provide functional kernels which are computational intensive,
# following python wrapper run on CPU will provide additional functions
# based on them:
#   - shape inference
#   - tensor memory management (constant weight & activations)
#   - kv-cache allocation & management
#
# these wrappers are also easy to manually re-implement in C++ for python-free
# deployment (they can be implemented in C++ class with same function and API
# with minimal effort). but python wrapper is best choice at developing stage.
#
def to_lt(a):
    if a is None:
        return llmops.empty(0)
    if type(a) is llmops.tensor:
        return a
    return llmops.tensor(a.detach().numpy())

def to_torch(a):
    return torch.from_numpy(a.numpy())

def llmop_iadd(a, b):
    llmops.iadd(to_lt(a), to_lt(b))

def llmop_imul(a, b):
    llmops.imul(to_lt(a), to_lt(b))

def llmop_trans(a, opname):
    llmops.itrans(to_lt(a), opname)

class llmop_fc_q4a(object):
    def __init__(self, linear):
        self.weight = llmops.offline_FC_quant_Q4A(to_lt(linear.weight))
        self.bias = to_lt(linear.bias) if linear.bias else None
        self.N = linear.weight.shape[0]
    
    def __call__(self, input):
        out = llmops.fc_Q4A(to_lt(input), self.weight, self.N)
        if self.bias:
            llmops.iadd(out, self.bias)
        return to_torch(out)

    def __repr__(self):
        return f"llmop_fc_q4a(weight: {self.weight.shape})"

class llmop_rmsnorm(object):
    def __init__(self, weight, variance_epsilon):
        self.weight = llmops.clone(to_lt(weight))
        self.eps = variance_epsilon

    def __call__(self, input):
        # variance = input.pow(2).mean(-1, keepdim=True)
        # input = input * torch.rsqrt(variance + variance_epsilon)
        # return weight * input.to(input_dtype)
        llmops.rmsnorm(to_lt(input), self.weight, self.eps)

    def __repr__(self):
        return f"llmop_rmsnorm(weight: {self.weight.shape}, eps: {self.eps})"

class llmop_embedding(object):
    def __init__(self, weight):
        self.weight = llmops.clone(to_lt(weight))

    def __call__(self, input_ids):
        # shape infer + output memory allocation are done explicitly as a separate step
        output = torch.empty(*input_ids.shape, self.weight.shape[1])
        # inference is done w/o shape-info
        llmops.embedding(llmops.tensor(output.numpy()), llmops.tensor(input_ids.numpy()), self.weight)
        return output

    def __repr__(self):
        return f"llmop_embedding(weight: {self.weight.shape})"


#=================================================================
# input_ids  : [batch, query_len]
# kv_cache   : [2 * n_layers, batch, n_head, max_kv_len, head_size]
# beam_table : [batch, max_kv_len]
# attn_mask  : [batch, query_len+past_len]
class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

    def load_from_HF(self, path, quant_type) -> None:
        print(f"load Tokenizer from {path}...")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = tokenizer.eos_token_id
        tokenizer.padding_side = "left"             # pad to left
        self.tokenizer = tokenizer

        print(f"load config/weight from HF model {path} ...")
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

        OP_fc = OP_fc_f32
        if quant_type == 'q8c':
            OP_fc = OP_fc_q8c
        if quant_type == 'q4a':
            OP_fc = OP_fc_q4a
        
        OP_fc = llmop_fc_q4a
        OP_rmsnorm = llmop_rmsnorm
        OP_embedding = llmop_embedding

        # the consts capture constant known at compile-time 
        self.op_dict = {
            'model.embed_tokens': OP_embedding(hf_model.model.embed_tokens.weight),
            'model.norm': OP_rmsnorm(hf_model.model.norm.weight, configs['rms_norm_eps']),
            'lm_head': OP_fc(hf_model.lm_head),
            'layers': [
                {
                    'input_layernorm': OP_rmsnorm(l.input_layernorm.weight, configs['rms_norm_eps']),
                    'post_attention_layernorm': OP_rmsnorm(l.post_attention_layernorm.weight, configs['rms_norm_eps']),
                    'self_attn.q_proj': OP_fc(l.self_attn.q_proj),
                    'self_attn.k_proj': OP_fc(l.self_attn.k_proj),
                    'self_attn.v_proj': OP_fc(l.self_attn.v_proj),
                    'self_attn.mha' : OP_mha(i, configs["rotary_dims"], configs["hidden_size"], configs["head_num"]),
                    'self_attn.o_proj': OP_fc(l.self_attn.o_proj),
                    'mlp.gate_proj': OP_fc(l.mlp.gate_proj),
                    'mlp.up_proj': OP_fc(l.mlp.up_proj),
                    'mlp.down_proj': OP_fc(l.mlp.down_proj),
                } for i, l in enumerate(hf_model.model.layers)
            ],
        }
        cost = time.time() - beg
        print(f'extracting done, cost {cost:.2f} seconds')

        self.configs = configs
        rope_base = 10000
        self.inv_freq = 1.0 / (rope_base ** (torch.arange(0, configs["rotary_dims"], 2).float().to("cpu") / configs["rotary_dims"]))
        #self.compute_graph_emitter(hf_model)

    def load(self, path) -> None:
        print(f"loading model from {path}...")
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def save(self, path):
        print(f"saving model to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def __repr__(self):
        ret = "Model configs :\n"
        for k, v in self.configs.items():
            ret += f'\t {k}: {v}\n'
        ret += "OPs:\n"
        ret += json.dumps(self.op_dict, indent=4, default=str)
        return ret

    # kv cache is not part of the model since it:
    #  - has non-const big tensor
    #  - is per session instead of per-model
    # it manages per-session resources, thus it's part of pipeline
    # and the best way to pass such resource into pipeline is inputs
    # we should set it as inputs instead of object
    def forward(self, input_ids, kv_cache):
        op_dict = self.op_dict

        hidden_states = op_dict['model.embed_tokens'](input_ids)

        for i, ops in enumerate(op_dict['layers']):
            input_layernorm = hidden_states.clone()
            ops['input_layernorm'](input_layernorm)

            q = ops['self_attn.q_proj'](input_layernorm)
            k = ops['self_attn.k_proj'](input_layernorm)
            v = ops['self_attn.v_proj'](input_layernorm)
            # q/k/v:[batch, seq_len, hiden_states]
            newq = ops["self_attn.mha"](q, k, v, kv_cache, self.inv_freq)
            attn_output = ops['self_attn.o_proj'](newq)

            llmop_iadd(attn_output, hidden_states)

            post_attention_layernorm = attn_output.clone()
            ops['post_attention_layernorm'](post_attention_layernorm)

            def mlp(states):
                # GLU_silu is used for up_projection
                gate = ops['mlp.gate_proj'](states)
                llmop_trans(gate, "silu")
                up_proj = ops['mlp.up_proj'](states)
                llmop_imul(up_proj, gate)
                # down projection
                down_proj = ops['mlp.down_proj'](up_proj)
                return down_proj

            hidden_states = mlp(post_attention_layernorm)

            llmop_iadd(hidden_states, attn_output)
            
        op_dict['model.norm'](hidden_states)

        logits = op_dict['lm_head'](hidden_states)    # q:[batch, seq_len, hiden_states]
        return logits

#=================================================================
# simple greedy pipeline using model_forward
def simple_chat_pipeline(model, org_prompt, max_kv_len, system_message, verbose):
    global inv_freq

    tokenizer = model.tokenizer
    streamer = TextStreamer(tokenizer)

    if max_kv_len > model.configs["max_position_embeddings"]:
        max_kv_len = model.configs["max_position_embeddings"]
    print(f"max_kv_len = {max_kv_len}")

    batch_size = 1
    kv_cache = KVCache(model.configs, batch_size, max_kv_len, verbose)
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
                logits = model.forward(input_ids, kv_cache)
                sys_msg = None

            if org_prompt:
                prompt = org_prompt
            else:
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
            logits = model.forward(input_ids, kv_cache)
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
                logits = model.forward(input_ids, kv_cache)

                second_tok_count += 1
                next_tokens = torch.argmax(logits, dim=-1).reshape(batch_size, 1)
                streamer.put(next_tokens)
            second_tok_latency = 0 if (second_tok_count == 0) else ((time.time() - second_tok_latency) / second_tok_count)
            streamer.end()
            if early_stop:
                print("\033[0;31m", early_stop)

            print("\033[0;90m", f" position_id: {kv_cache.position_id}  latency: {first_tok_latency*1e3:.2f} ms + {second_tok_latency*1e3:.2f}ms x {second_tok_count}  rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MB")
            print("\033[00m")
            if org_prompt:
                break

def main():
    parser = argparse.ArgumentParser('')
    
    # /home/tingqian/models/chinese-alpaca-2-1.3b-hf
    # C:/Users/tingqian/Syncplicity/LTQ/Models/chinese-alpaca-2-1.3b-hf
    parser.add_argument('-hf', '--hf_model', type=str)
    parser.add_argument('-m', '--model', type=str, default='saved_model.pkl')
    parser.add_argument('-q', '--quant', type=str)
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--sys', type=str, default=None)
    parser.add_argument('--kv-len', type=int, default=2048)
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('-d', '--device', default="cpu")
    parser.add_argument('prompt', type=str, nargs='?')
    
    args = parser.parse_args()

    model = Model()

    print(f" rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MiB")
    if args.hf_model:
        model.load_from_HF(args.hf_model, args.quant)
        if args.save:
            model.save('saved_model.pkl')
    else:
        model.load(args.model)

    print(f" rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MiB")
    print(model)

    print(f"to device {args.device} ... ")
    #model.to(args.device)

    simple_chat_pipeline(model, args.prompt, args.kv_len, args.sys, args.verbose)
