import sys
import torch
import llext1 # llama_ext depends on torch, so torch must be loaded first
import time

if False:
    a=torch.rand(2, 3)
    print(a)
    print(llext1.accTensor(a, 0, 1))
    sys.exit(0)

N = 4096
K = 4096
a0 = torch.rand(N, K, dtype=torch.float32)

b0, b1 = llext1.FC_quant_Q8C(a0)

input = torch.rand(1, 1, K, dtype=torch.float32)

for i in range(10):
    t0 = time.time()
    output = llext1.FC_evaluate_Q8C(input, b0, b1, N)
    dt = time.time() - t0
    print(f"{dt*1e3:.1f} ms {N*K/1e9/dt} GB/s")