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

def attention_rope(query_states, key_states, value_states, inv_freq, kv_cache, kv_cache_slots, position_id, layer_idx):
    num_heads = kv_cache.size(2)
    max_kv_length = kv_cache.size(3)
    head_dim = kv_cache.size(-1)

    bsz, q_len, _ = query_states.size()
    num_kv_heads = num_heads
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # q/k/v states : [batch, nHead, q_len, head_dim]
    #rope_embed(query_states, inv_freq, position_id)
    #rope_embed(key_states, inv_freq, position_id)

    llmops.rope_embed(to_lt(query_states), to_lt(inv_freq), position_id)
    llmops.rope_embed(to_lt(key_states), to_lt(inv_freq), position_id)

    # kv_cache is a circular buffer, and tokens should be overwritten in word boundary
    # key_states, value_states = kv_cache.update_cache(self.layer_idx, key_states, value_states)
    #
    #  cache: [num_layers*2, B, H, length_max, S]
    #  slots: [q_len]   where ith query token should be placed cache
    for k in range(q_len):
        kv_cache[2*layer_idx + 0, :, :, kv_cache_slots[k], :] = key_states[:, :, k, :]
        kv_cache[2*layer_idx + 1, :, :, kv_cache_slots[k], :] = value_states[:, :, k, :]
    
    # us beam_idx to gather(reorder kv cache), skipped in greedy case
    if position_id + q_len < kv_cache.size(3):
        key_states = kv_cache[2*layer_idx + 0, :, :, :(position_id + q_len), :]
        value_states = kv_cache[2*layer_idx + 1, :, :, :(position_id + q_len), :]
    else:
        key_states = kv_cache[2*layer_idx + 0, :, :,:,:]
        value_states = kv_cache[2*layer_idx + 1, :, :,:,:]

    #kv_seq_len = kv_cache.cur_kv_len
    #kv_mask = kv_cache.mask[:kv_seq_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    # apply causal mask, so:
    #    q-token[q_len-1] can use all kv-tokens
    #    q-token[q_len-2] can use all kv-tokens except the last one
    #    q-token[q_len-3] can use all kv-tokens except the last two
    #    q-token[k] can use all kv-tokens except the last (q_len - 1 - k)
    #    ....
    # [batch, num_heads, q_len ,kv_len] 
    for k in range(q_len-1):
        pos = torch.arange(start=(k + 1), end=q_len, step=1, dtype = torch.int32)
        pos = kv_cache_slots[pos]
        attn_weights[:, :, k, pos] = torch.finfo(torch.float32).min

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    return attn_output

def test_mha(kernel, past_kv_len, cur_seq_len, repeats, test_acc = False):
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

    note = ""
    if test_acc:
        ref = attention_rope(query_states, key_states, value_states, inv_freq, kv_cache, kv_cache_slots, position_id, 0)
        kernel(lt_out, lt_key, lt_value, to_lt(inv_freq), lt_kv_cache, to_lt(kv_cache_slots), position_id, 0)
        if numpy.allclose(ref.numpy(), lt_out.numpy()):
            note = " [allclose ok]"
        else:
            print(ref.numpy()[0,0:4,...])
            print(lt_out.numpy()[0,0:4,...])
            note = " [allclose failed]"

    t0 = time.time()
    for r in range(repeats):
        for layer_idx in range(num_layers):
            kernel(lt_out, lt_key, lt_value, to_lt(inv_freq), lt_kv_cache, to_lt(kv_cache_slots), position_id, layer_idx)
    t1 = time.time()
    
    latency = (t1-t0)/(repeats * num_layers)
    kv_size_1layer = 2* B * H * (past_kv_len + qL) * S * kv_cache.element_size()
    q_size_1layer = query_states.numel() * query_states.element_size()

    kvLen = past_kv_len + qL
    MAdds_per_sec = B * H * (S*qL*kvLen + qL*S*kvLen)/float(latency)

    length_info = f"{past_kv_len}+{qL}"
    print(f"{kernel.__name__:18} {length_info:8}  x{repeats * num_layers:5}  {latency * 1e3:7.3f} ms "
          f" q-size:{q_size_1layer/1e6:6.2f}MB  kv-size:{kv_size_1layer/1e6:6.2f}MB "
          f" MemBW: {(q_size_1layer + kv_size_1layer)/latency/1e9:6.1f} GB/s MAdds: {MAdds_per_sec/1e9:6.1f}G/s {note}")

class time_value:
    def __init__(self, t):
        self.t_sec = t
        self.t = t*1e3
        self.unit = "ms"
        if (self.t < 1):
            self.t *= 1e3
            self.unit = "us"
    def __float__(self):
        return self.t_sec

    def __repr__(self):
        return f"{self.t : 6.1f} {self.unit}"
    


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

    note = ""
    if not numpy.allclose(ref, act):
        note = " [allclose failed]"
        #print("ref=", ref)
        #print("act=", act)
        #assert numpy.allclose(ref, act)

    t0 = time.time()
    for r in range(repeats):
        mm_qk_kernel(to_lt(q), to_lt(kcache))
    t1 = time.time()
    latency = time_value((t1-t0)/repeats)

    MAdds_per_sec = (S * B * H * qL * kvLen)/float(latency)
    print(f"{mm_qk_kernel.__name__:10} S:{S} qL: {qL:6}  kvLen: {kvLen:6}  {latency}  q:{q.numpy().nbytes/1e6:6.1f} MB  k:{kcache.numpy().nbytes/1e6:6.1f} MB   attn:{act.nbytes/1e6:6.1f} MB  MAdds: {MAdds_per_sec/1e9:.1f}G/s {mm_qk_kernel.__name__:10} {note}" )

def main():
    
    # warm-up
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
    print("------------------------------------------------")
    for ker in mm_qk_kernels: test_qk(ker, 8, 2048, 10)
    print("------------------------------------------------")
    for ker in mm_qk_kernels: test_qk(ker, 32, 2048, 10)
    print("------------------------------------------------")
    for ker in mm_qk_kernels: test_qk(ker, 1024, 2048, 10)
    print("------------------------------------------------")
    for ker in mm_qk_kernels: test_qk(ker, 2048, 2048, 10)
    return

    test_qk(32, 32, 10)
    test_qk(128, 128, 10)
    test_qk(1024, 1024, 10)
    test_qk(2048, 2048, 10)

def main_mha(test_acc = True):
    #test_mha(llmops.attention_rope2, 0, 1024, 1, True); return
    #test_mha(llmops.attention_rope2, 0, 1025, 1, True)
    #test_mha(llmops.attention_rope2, 0, 1023, 1, True); return
    #test_mha(llmops.attention_rope2, 0, 1024, 2); return
    if test_acc:
        test_mha(llmops.attention_rope, 0, 32, 1, True)
        test_mha(llmops.attention_rope2, 30, 1, 1, True)
        test_mha(llmops.attention_rope2, 31, 1, 1, True)
        test_mha(llmops.attention_rope2, 0, 1024, 1, True)
        test_mha(llmops.attention_rope2, 0, 1022, 1, True)
        test_mha(llmops.attention_rope2, 0, 1023, 1, True)
        test_mha(llmops.attention_rope2, 0, 1025, 1, True)
        test_mha(llmops.attention_rope2, 0, 1026, 1, True)

    print("-------------------------------------")
    test_mha(llmops.attention_rope, 0, 1024, 2)
    test_mha(llmops.attention_rope, 0, 1025, 2)
    test_mha(llmops.attention_rope2, 0, 1024, 2)
    test_mha(llmops.attention_rope2, 0, 1025, 2)

    print("-------------------------------------")
    test_mha(llmops.attention_rope, 30, 1, 10)
    test_mha(llmops.attention_rope, 511, 1, 10)
    test_mha(llmops.attention_rope, 1023, 1, 10)
    test_mha(llmops.attention_rope, 2047, 1, 10)

    test_mha(llmops.attention_rope2, 30, 1, 10)
    test_mha(llmops.attention_rope2, 511, 1, 10)
    test_mha(llmops.attention_rope2, 1023, 1, 10)
    test_mha(llmops.attention_rope2, 2047, 1, 10)

if __name__ == '__main__':
    print("********* Note: 1GB=1,000,000,000 Bytes *********")
    #main()
    #main_qk((llmops.mm_qk, llmops.mm_qk42, llmops.mm_qk24, llmops.onednn_qk))
    #for ker in (llmops.mm_qk42, llmops.mm_qk81, llmops.onednn_qk): test_qk(ker, 8, 64, 10)
    #for ker in (llmops.mm_qk42, llmops.mm_qk81, llmops.onednn_qk): test_qk(ker, 16, 64, 10)
    #for ker in (llmops.mm_qk42, llmops.mm_qk81, llmops.onednn_qk): test_qk(ker, 256, 256, 1000)
    main_mha(False)