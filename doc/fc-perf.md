
# machine 14900k: Raptor Lake

Up to 8 Raptor Cove performance cores (P-core):
 - CPU 0,2,4,6,8,10,12,14
 - CPU 1,3,5,7,9,11,13,15 (logical cores)
 - 2MB L2 cache for each P-core

Up to 16 Gracemont efficiency cores (E-core) in 4-core clusters
 - 4MB L2 cache per cluster

Up to 36 MB L3 cache
 - 3 MB per P-core
 - 3 MB per E-core cluster

# test1

```bash
numactl -C0,2,4,6,8,10,12,14 python ./ops/perf_ops.py

********* Note: 1GB=1,000,000,000 Bytes *********
**** test_fc  fc_Q4A   B,M,N,K=1,1,4096,40960   weight_q4a.shape=[4096, 40960]=>[128, 1280, 640] ****
 weight:104.9 MB 1.816 ms  Bound:57.8 GB/s  92.4 GMAdds/s  [allclose ok]
 weight:104.9 MB 1.788 ms  Bound:58.6 GB/s  93.8 GMAdds/s  [allclose ok]
 weight:104.9 MB 1.805 ms  Bound:58.1 GB/s  93.0 GMAdds/s  [allclose ok]
 weight:104.9 MB 1.794 ms  Bound:58.4 GB/s  93.5 GMAdds/s  [allclose ok]
**** test_fc  fc_Q4A   B,M,N,K=1,2,4096,40960   weight_q4a.shape=[4096, 40960]=>[128, 1280, 640] ****
 weight:104.9 MB 1.942 ms  Bound:54.0 GB/s  172.8 GMAdds/s  [allclose ok]
 weight:104.9 MB 1.925 ms  Bound:54.5 GB/s  174.4 GMAdds/s  [allclose ok]
 weight:104.9 MB 1.930 ms  Bound:54.3 GB/s  173.8 GMAdds/s  [allclose ok]
 weight:104.9 MB 1.923 ms  Bound:54.5 GB/s  174.5 GMAdds/s  [allclose ok]
**** test_fc  fc_Q4A   B,M,N,K=1,4,4096,40960   weight_q4a.shape=[4096, 40960]=>[128, 1280, 640] ****
 weight:104.9 MB 2.216 ms  Bound:47.3 GB/s  302.9 GMAdds/s  [allclose ok]
 weight:104.9 MB 2.155 ms  Bound:48.7 GB/s  311.4 GMAdds/s  [allclose ok]
 weight:104.9 MB 2.162 ms  Bound:48.5 GB/s  310.3 GMAdds/s  [allclose ok]
 weight:104.9 MB 2.164 ms  Bound:48.5 GB/s  310.2 GMAdds/s  [allclose ok]
**** test_fc  fc_Q4A   B,M,N,K=1,8,4096,40960   weight_q4a.shape=[4096, 40960]=>[128, 1280, 640] ****
 weight:104.9 MB 2.835 ms  Bound:37.0 GB/s  473.5 GMAdds/s  [allclose ok]
 weight:104.9 MB 2.794 ms  Bound:37.5 GB/s  480.4 GMAdds/s  [allclose ok]
 weight:104.9 MB 2.703 ms  Bound:38.8 GB/s  496.6 GMAdds/s  [allclose ok]
 weight:104.9 MB 2.685 ms  Bound:39.0 GB/s  499.8 GMAdds/s  [allclose ok]
```

each thread in fc_Q4A calculate output sub-matrix in uint of [K, 32], which is less than 2MB in int4.
so when B*M increase from 1 to 8, latency increased by (2.7-1.8)=0.9ms, and A matrix & sub-matrix B should be all in L2.
so 0.9ms should be the expected time took to run kernel for [7,K]x[K,M] on L2-hit cases, suppose L2 bandwidth can support
VNNI's throughput (CPU_freq * 2 * 32)*8 = 1638.4 GMAdds/s (suppose CPU is run at 3.2GHz), what we really saw is
(7*4096*40960/0.9e-3/1e9) = 1304 GMAdds/s, which is 20% slower than expected.
