import os, sys, time
import torch
import math

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


def mytest_tensor():
    import psutil
    class C(object):
        def __getitem__(self, val):
            print(type(val), val)
    c = C()
    c[2, 3:-1]

    print(type(llmops))
    print(dir(llmops))
    a = llmops.empty(1,2,3,4)
    print(f"a.shape={a.shape} a.strides={a.strides} a.item_size={a.item_size} a.data={hex(a.data)}")
    
    na = numpy.array(a, copy=False)
    print(na)
    nb = a.numpy()
    print(nb)
    na[0,0] = 1
    print("============")
    print(na)
    print("============")
    print(a)

    print(f"rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MB")
    a = llmops.ones(1024, 1024, 512)
    print(f"rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MB")
    nb = a.numpy()
    print(f"rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MB")
    del a
    print(f"rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MB")
    del nb
    print(f"rss: {psutil.Process().memory_info().rss/(1024**2):.1f} MB")


    import pickle
    a = llmops.ones(3, 8)
    print(a)
    with open("_temp_test_tensor.pkl", 'wb') as f:
        pickle.dump({"a": a}, f)
    with open("_temp_test_tensor.pkl", 'rb') as f:
        di = pickle.load(f)
        print(di, di['a'])

def test_embedding():
    batch_size = 1
    seq_len = 7
    vocab_size = 3
    embedding_dim = 4096
    weight = torch.rand(vocab_size, embedding_dim)
    input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64)

    print("input=", input.numpy().dtype, input.dtype, input.shape, input)
    print("weight=", weight.dtype, weight.shape, weight)
    ref = torch.nn.functional.embedding(input, weight)
    print("ref=", ref.numpy())
    out = llmops.empty(batch_size, seq_len, embedding_dim)
    llmops.embedding(out, to_lt(input), to_lt(weight))
    print("out=", out.numpy())
    for i in range(seq_len):
        a = ref.numpy()[0,i,:]
        b = out.numpy()[0,i,:]
        if not numpy.allclose(a, b):
            print("not match!======", i, a, b)
    assert numpy.allclose(ref.numpy(), out.numpy())

def test_rmsnorm():
    batch_size = 1
    seq_len = 7
    states = 4096
    input = torch.rand(batch_size, seq_len, states)
    weight = torch.rand(states)
    eps = 1e-5
    variance = input.pow(2).mean(-1, keepdim=True)
    ref = weight * (input * torch.rsqrt(variance + eps))
    
    out = to_lt(input).clone()
    llmops.rmsnorm(out, to_lt(weight), eps)
    assert numpy.allclose(ref.numpy(), out.numpy())

def test_fc():
    B = 2
    M = 17
    N = 4096
    K = 4096
    #weight = torch.rand(N, K, dtype=torch.float32)

    # weight can be perfected re-constructed by i4 quantization
    # input can be perfected re-constructed by i8 quantization
    weight = torch.randint(-8, 8, (N, K), dtype=torch.float32)
    input = torch.randint(-127, 128, (B, M, K), dtype=torch.float32)
    for k in range(0, K, 32):
        weight[:, k] = -8
        weight[:, k+1] = 7
        input[:, :, k] = -127
        input[:, :, k+1] = 127
    weight = weight * (1/128)
    input = input * (1/128)
    print(f"weight: {weight.dtype} {weight.shape}")
    print(f"input: {input.dtype} {input.shape}")

    weight_q4a = llmops.offline_FC_quant_Q4A(to_lt(weight))
    weight_dq4a = llmops.offline_FC_dequant_Q4A(weight_q4a).numpy()

    ref = torch.nn.functional.linear(input, weight, None).numpy()
    out = llmops.fc_Q4A(to_lt(input), weight_q4a, N).numpy()

    weight = weight.numpy()
    print(f"weight     ={weight[0,:]}")
    print(f"weight_dq4a={weight_dq4a[0,:]}")

    CSTR = ""
    for k in range(K):
        CSTR += "T" if numpy.allclose(weight_dq4a[:,k], weight[:,k]) else "F"
    print(CSTR)

    assert numpy.allclose(weight_dq4a, weight)

    print(ref)
    print(out)
    for b in range(B):
        for m in range(M):
            print(f"{b} {m} amax abs diff = {numpy.amax(numpy.abs(ref[b, m, :] - out[b, m, :]))}  avg abs diff = {numpy.mean(numpy.abs(ref[b, m, :] - out[b, m, :]))}")
    assert numpy.allclose(ref, out)

    while False:
        CNT=1000
        t0 = time.time()
        for i in range(CNT):
            output = llmops.fc_Q4A(to_lt(input), b0, N)
        dt = (time.time() - t0)/CNT
        b0_bytesize = b0.numel() * b0.item_size
        print(f" {b0_bytesize/(1024**2):.1f} MB {dt*1e3:.1f} ms  {b0_bytesize/(1024**3)/dt:.1f} GB/s")

def test_softmax():
    def do_test(input):
        ref = torch.nn.functional.softmax(input, dim=-1)
        act = llmops.tensor(input.numpy()).clone()
        llmops.softmax(act)
        if not numpy.allclose(ref.numpy(), act.numpy()):
            print(ref.numpy())
            print(act.numpy())
            assert False

    do_test(torch.rand(7, 12, 3, 69, dtype=torch.float32) - 0.5)
    do_test(torch.tensor([0.66046,0.61789,-0.289624,0.799137,-0.715343,0.321021,1.47277,-3.40282e+38,], dtype=torch.float32))
    do_test(torch.tensor([0.66046,0.61789,], dtype=torch.float32))

def rope_embed(x, inv_freq, position_id):
    rotary_dims = inv_freq.size(0)*2
    half_rotary_dim = rotary_dims//2
    q_len = x.size(2)
    for k in range(q_len):
        xita = inv_freq * (position_id + k)
        vcos = torch.cos(xita)
        vsin = torch.sin(xita)
        x0 = x[:, :, k, :half_rotary_dim]
        x1 = x[:, :, k, half_rotary_dim:rotary_dims]
        y0 = vcos * x0 - vsin * x1
        y1 = vsin * x0 + vcos * x1
        x[:, :, k, :half_rotary_dim] = y0
        x[:, :, k, half_rotary_dim:rotary_dims] = y1

def test_rope():
    B = 10
    qL = 17
    H = 32
    S = 128
    rotary_dims = 64
    rope_base = 10000
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dims, 2).float().to("cpu") / rotary_dims))
    position_id = 16

    states = torch.rand(B, H, qL, S)

    ref = states.clone()
    rope_embed(ref, inv_freq, position_id)

    out = to_lt(states).clone()
    llmops.rope_embed(out, to_lt(inv_freq), position_id)

    ref = ref.numpy()
    out = out.numpy()
    for i in range(qL):
        print(i, numpy.amax(numpy.abs(ref[:,:,i,:] - out[:,:,i,:])))
    assert numpy.allclose(ref, out, atol=1e-5)

'''
void attention_rope(tensor q,          // [B, qL, H*S]
                    tensor k,          // [B, qL, H*S]
                    tensor v,          // [B, qL, H*S]
                    tensor inv_freq,   // [rotary_dims/2] for RoPE
                    tensor kv_cache,   // [2, B, H, max_length, S]
                    tensor kvc_slots,  // [qL]
                    int position_id,
                    int layer_id)
'''
def attention_rope(query_states, key_states, value_states, inv_freq, kv_cache, kv_cache_slots, position_id, layer_idx):
    num_heads = kv_cache.size(2)
    max_kv_length = kv_cache.size(3)
    head_dim = kv_cache.size(-1)
    # [half_ro_ndims]
    rotary_dims = inv_freq.size(0) * 2

    # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/models/llama/modeling_llama.py#L331
    # q    : B, qL, H*S
    # k/v  : B, kvL, H*S
    #
    # kv-cache layout : [layer, B, H, max_length, S]
    #
    # cache_slots  [kvL]          : gives kv positions in the cache for each token [0 ~ kvL) in k/v to push.
    # cache_gather [B, all-kvL]   : gives how to fetch from kv-cache to get present-kv
    #
    bsz, q_len, _ = query_states.size()
    num_kv_heads = num_heads
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    '''
    q2 = query_states.clone().contiguous()

    llmops.rope_embed(to_lt(query_states), to_lt(inv_freq), position_id)
    llmops.rope_embed(to_lt(q2), to_lt(inv_freq), position_id)

    print(query_states.numpy().strides)
    print(q2.numpy().strides)

    print(torch.allclose(q2, query_states))
    
    assert False
    '''

    # q/k/v states : [batch, nHead, q_len, head_dim]

    # derive total kv length from attn (has limitation)
    # apply_rotary_pos_emb to key_states/value_states    

    #rope_embed(query_states, inv_freq, position_id)
    #rope_embed(key_states, inv_freq, position_id)

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
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
        pos = kv_cache_slots[pos]
        attn_weights[:, :, k, pos] = torch.finfo(torch.float32).min

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
    return attn_output

def test_mha():
    #  [B, qL, H*S]
    cur_kv_len = 0
    B = 1
    qL = 8
    H = 32
    S = 128
    rotary_dims = 64
    max_length = 16
    
    query_states = torch.rand(B, qL, H*S)
    key_states = torch.rand(B, qL, H*S)
    value_states = torch.rand(B, qL, H*S)

    rope_base = 10000
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dims, 2).float().to("cpu") / rotary_dims))
    #inv_freq = inv_freq * 0
    print(f"inv_freq={inv_freq}")

    # [2, B, H, max_length, S]
    kv_cache = torch.rand(2, B, H, max_length, S)
    # [qL]
    kv_cache_slots = torch.arange(cur_kv_len, cur_kv_len + qL, dtype=torch.int32)

    position_id = cur_kv_len
    layer_idx = 0

    ref = attention_rope(query_states, key_states, value_states, inv_freq, kv_cache, kv_cache_slots, position_id, layer_idx).numpy()
    
    out = to_lt(query_states).clone()
    llmops.attention_rope(out, to_lt(key_states), to_lt(value_states), to_lt(inv_freq), to_lt(kv_cache), to_lt(kv_cache_slots), position_id, 0)
    out = out.numpy()

    print(ref.shape, out.shape)
    for i in range(qL):
        print(i, numpy.amax(numpy.abs(ref[:,i,:] - out[:, i, :])))
    assert numpy.allclose(ref, out)


if __name__ == '__main__':
    for i in range(10):
        llmops.syclmain()

    #test_softmax()
    #test_tensor()
    #test_fc()
