import os, sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path  + "./build/Debug")
import llmops
import numpy

def test_embedding():
    import torch
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

test_embedding()
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

