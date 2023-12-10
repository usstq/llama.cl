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

def test_mha(past_kv_len, cur_seq_len, repeats = 100):
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
    kv_cache_slots = torch.arange(past_kv_len, past_kv_len + qL, dtype=torch.int32)

    position_id = past_kv_len

    lt_kv_cache = to_lt(kv_cache).clone()
    lt_out = to_lt(query_states).clone()
    lt_key = to_lt(key_states).clone()
    lt_value = to_lt(value_states).clone()

    t0 = time.time()
    for r in range(repeats):
        for layer_idx in range(num_layers):
            llmops.attention_rope(lt_out, lt_key, lt_value, to_lt(inv_freq), lt_kv_cache, to_lt(kv_cache_slots), position_id, layer_idx)
    t1 = time.time()
    
    latency = (t1-t0)/(repeats * num_layers)
    kv_size_1layer = 2* B * H * (past_kv_len + qL) * S * kv_cache.element_size()

    print(f"{past_kv_len}+{qL}\t:  x{repeats * num_layers}  {latency * 1e3 : .1f} ms   q-size:{query_states.numel() * query_states.element_size()/1e6:.1f}MB  kv-size:{kv_size_1layer/1e6:.1f}MB  {kv_size_1layer/latency/1e9 : .1f} GB/s  (1GB=1,000,000,000 Bytes)")

def test_qk(mm_qk_kernel, qL, kvLen, repeats=100):
    B = 1
    H = 32
    S = 128
    num_layers = 32
    
    q = torch.rand(B, qL, H*S)
    kcache = torch.rand(B, H, kvLen, S)

    q2 = q.view(B, qL, H, S).permute(0,2,1,3) * (1/(S**0.5))

    # accuracy & warm-up
    ref = (q2 @ kcache.permute(0,1,3,2)).numpy()
    act = mm_qk_kernel(to_lt(q), to_lt(kcache)).numpy()

    assert numpy.allclose(ref, act)

    t0 = time.time()
    for r in range(repeats):
        mm_qk_kernel(to_lt(q), to_lt(kcache))
    t1 = time.time()
    latency = (t1-t0)/repeats

    MAdds_per_sec = (S * B * H * qL * kvLen)/latency
    print(f"{mm_qk_kernel.__name__:8} S:{S} qL: {qL:6}  kvLen: {kvLen:6}  {latency * 1e3 : 6.1f} ms  q:{q.numpy().nbytes/1e6:6.1f} MB  k:{kcache.numpy().nbytes/1e6:6.1f} MB   attn:{act.nbytes/1e6:6.1f} MB  MAdds: {MAdds_per_sec/1e9:.1f}G/s" )

def main():
    # warm-up
    test_mha(2047, 1, 1)
    test_mha(1023, 1, 10)
    test_mha(1024-8, 8, 10)
    return
    

    test_mha(31, 1, 10)
    test_mha(511, 1, 10)
    test_mha(1023, 1, 10)
    test_mha(2047, 1, 10)
    
    #test_mha(0, 32, 100)
    #test_mha(0, 64, 10)
    test_mha(0, 128, 2)
    test_mha(0, 256, 2)
    test_mha(0, 512, 2)
    test_mha(0, 1024, 2)


def main_qk(mm_qk_kernels):
    for ker in mm_qk_kernels:
        test_qk(ker, 8, 2048, 10)

    for ker in mm_qk_kernels:
        test_qk(ker, 32, 2048, 10)

    for ker in mm_qk_kernels:
        test_qk(ker, 1024, 2048, 10)

    for ker in mm_qk_kernels:
        test_qk(ker, 2048, 2048, 10)
    return

    test_qk(32, 32, 10)
    test_qk(128, 128, 10)
    test_qk(1024, 1024, 10)
    test_qk(2048, 2048, 10)

if __name__ == '__main__':
    #main()
    main_qk((llmops.mm_qk, llmops.mm_qk2, llmops.mm_qk42, llmops.mm_qk24))