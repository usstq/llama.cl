import sys
import torch
import llext1 # llama_ext depends on torch, so torch must be loaded first
import time

if False:
    a=torch.rand(2, 3)
    print(a)
    print(llext1.accTensor(a, 0, 1))
    sys.exit(0)

torch.set_num_threads(2)

N = 4096
K = 4096
weight = torch.rand(N, K, dtype=torch.float32)

b0, b1 = llext1.FC_quant_Q8C(weight)

input = torch.rand(1, 1, K, dtype=torch.float32)

output = llext1.FC_evaluate_Q8C(input, b0, b1, N)
output0 = torch.nn.functional.linear(input, weight, None)

print(f"amax abs diff = {(output0 - output).abs().amax()}  avg abs diff = {(output0 - output).abs().mean()}")

while True:
    CNT=1000
    t0 = time.time()
    for i in range(CNT):
        output = llext1.FC_evaluate_Q8C(input, b0, b1, N)
    dt = (time.time() - t0)/CNT
    b0_bytesize = b0.element_size() * b0.nelement()
    print(f" {b0_bytesize/(1024**2):.1f} MB {dt*1e3:.1f} ms  {b0_bytesize/(1024**3)/dt:.1f} GB/s")