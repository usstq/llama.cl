import sys
import torch
import llext1 # llama_ext depends on torch, so torch must be loaded first

if False:
    a=torch.rand(2, 3)
    print(a)
    print(llext1.accTensor(a, 0, 1))
    sys.exit(0)

torch.set_num_threads(12)
torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=2000)

N = 3200
a0 = torch.rand(N, 4096, dtype=torch.float32)

b0, b1 = llext1.FC_quant_Q8C(a0)

q8c_blk = b0[0,0,:].view(8, 32*4)

a1 = a0.transpose(0, 1)

print(a1[:4, :32])
print(q8c_blk[0, :].view(32, 4).transpose(0, 1))
print(b1)