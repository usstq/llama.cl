import os, sys, time
import math
import torch
torch.set_default_dtype(torch.float32)
torch.set_flush_denormal(True)

if sys.platform == 'win32':
    os.add_dll_directory("C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/redist/intel64_win/compiler")
    os.add_dll_directory("C:/Program Files (x86)/Intel/oneAPI/compiler/2023.2.1/windows/bin")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import llmops
import numpy

def to_lt(a):
    if a is None:
        return llmops.empty(0)
    if type(a) is llmops.tensor:
        return a
    return llmops.tensor(a.detach().numpy())

def to_torch(a):
    return torch.from_numpy(numpy.array(a, copy=False))

def test_mha(past_kv_len, cur_seq_len):
    #  [B, qL, H*S]
    B = 1
    qL = cur_seq_len
    H = 32
    S = 128
    num_layers = 32
    rotary_dims = 64
    max_length = 2048
    
    query_states = torch.rand(B, qL, H*S)
    key_states = torch.rand(B, qL, H*S)
    value_states = torch.rand(B, qL, H*S)

    rope_base = 10000
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dims, 2).float().to("cpu") / rotary_dims))
    #inv_freq = inv_freq * 0
    #print(f"inv_freq={inv_freq}")

    # [2, B, H, max_length, S]
    kv_cache = torch.rand(2*num_layers, B, H, max_length, S)
    # [qL]
    kv_cache_slots = torch.arange(past_kv_len, past_kv_len + qL)

    position_id = past_kv_len

    lt_kv_cache = to_lt(kv_cache).clone()
    lt_out = to_lt(query_states).clone()
    lt_key = to_lt(key_states).clone()
    lt_value = to_lt(value_states).clone()

    repeats = 100
    t0 = time.time()
    for r in range(repeats):
        for layer_idx in range(num_layers):
            llmops.attention_rope(lt_out, lt_key, lt_value, to_lt(inv_freq), lt_kv_cache, to_lt(kv_cache_slots), position_id, layer_idx)
    t1 = time.time()
    
    latency = (t1-t0)/repeats
    print(f" latency   : {latency * 1e3 : .1f} ms")

    kv_size = 2*num_layers * B * H * (past_kv_len + qL) * S * kv_cache.element_size()
    print(f" bandwidth : {kv_size/latency/1e9 : .1f} GB/s  (1GB=1,000,000,000 Bytes)")

if __name__ == '__main__':
    test_mha(15, 1)
    test_mha(31, 1)
    test_mha(63, 1)
    test_mha(127, 1)
    test_mha(255, 1)
    test_mha(511, 1)
    test_mha(1023, 1)
    test_mha(2047, 1)