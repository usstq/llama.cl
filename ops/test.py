import os, sys, time
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path  + "./build/Release")
import llmops
import numpy
import torch

def to_lt(a):
    if a is None:
        return llmops.empty(0)
    if type(a) is llmops.tensor:
        return a
    return llmops.tensor(a.detach().numpy())

def to_torch(a):
    return torch.from_numpy(numpy.array(a, copy=False))

def test_embedding():
    batch_size = 2
    seq_len = 3
    vocab_size = 3
    embedding_dim = 4096
    weight = torch.rand(vocab_size, embedding_dim)
    input = torch.randint(0, vocab_size, (batch_size, seq_len))

    print("input=", input)
    print("weight=", weight)
    ref = torch.nn.functional.embedding(input, weight)
    print("ref=", ref)
    act = llmops.embedding(llmops.tensor(input.numpy()), llmops.tensor(weight.numpy()))
    print(act)
    print("test_embedding(): ", numpy.allclose(ref.numpy(), numpy.array(act, copy=False)))

def test_fc():
    N = 32
    K = 32
    weight = torch.rand(N, K, dtype=torch.float32)
    #weight = torch.randint(0, 10, (N, K), dtype=torch.float32) - 5
    input = torch.rand(1, 1, K, dtype=torch.float32) - 0.5

    b0 = llmops.offline_FC_quant_Q4A(to_lt(weight))
    b0_deq = to_torch(llmops.offline_FC_dequant_Q4A(b0))

    print("weight=")
    print(weight[0,:])
    print("b0_deq=")
    print(b0_deq[0,:])

    output = to_torch(llmops.fc_Q4A(to_lt(input), b0, N))*15

    output0 = torch.nn.functional.linear(input, weight, None)*15

    print(output[0, 0, -8:])
    print(output0[0, 0, -8:])
    print(f"amax abs diff = {(output0 - output).abs().amax()}  avg abs diff = {(output0 - output).abs().mean()}")

    while False:
        CNT=1000
        t0 = time.time()
        for i in range(CNT):
            output = llmops.fc_Q4A(to_lt(input), b0, N)
        dt = (time.time() - t0)/CNT
        b0_bytesize = b0.numel() * b0.item_size
        print(f" {b0_bytesize/(1024**2):.1f} MB {dt*1e3:.1f} ms  {b0_bytesize/(1024**3)/dt:.1f} GB/s")


test_fc()
sys.exit(0)

def test_tensor():
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
    na[0,0] = 1
    print("============")
    print(na)
    print("============")
    print(a)

