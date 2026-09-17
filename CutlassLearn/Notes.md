# CutlassLearn Notes

Hardware: RTX 5080 (sm_120, Blackwell consumer) · CUDA 13.2 · WSL2

## Stage 0 — CudaBasics

### VectorAdd (N = 2^26, traffic = 3 × 256 MB)

| Kernel | Time | Bandwidth |
|---|---|---|
| V1 one-thread-per-elem | 0.950 ms | 847 GB/s |
| V2 grid-stride | 1.036 ms | 777 GB/s |

- RTX 5080 理论带宽 ~960 GB/s (GDDR7)，V1 已达 ~88%，说明纯 memory-bound kernel 下"一线程一元素"已足够。
- grid-stride 反而略慢：固定 grid 的尾部 block 处理多个元素，负载略不均。grid-stride 的价值在更复杂的 kernel（占用受限、需要控制 block 数）才体现。

### Reduction (N = 2^26, traffic = 256 MB)

| Kernel | Time | Bandwidth |
|---|---|---|
| V1 shared-mem tree | 0.593 ms | 453 GB/s |
| V2 warp shuffle | 0.417 ms | 644 GB/s |
| V3 grid-stride + shuffle | 0.324 ms | 828 GB/s |

- V1 慢的根因：树形归约每一轮都要 `__syncthreads`，最后 5 轮（≤32 元素）只有 1 个 warp 在干活，其余 warp 空等。
- V2 用 `__shfl_down_sync` 寄存器交换替代最后 5 轮 shared memory + barrier。
- V3 在 V2 基础上把 grid 从 26 万个 block 降到 672 个（SM 数 × 8），每线程 grid-stride 串行累加多个元素，消除 block 调度开销，达到理论带宽的 ~86%。
- 附带收益：V3 的累加顺序更接近 CPU 顺序，结果精度也是三者中最好的。

## Stage 1 — GemmFromScratch (M=N=K=4096, FP32)

| Kernel | Time | TFLOPS | % of cuBLAS |
|---|---|---|---|
| cuBLAS | 3.689 ms | 37.25 | — |
| V1 naive | 39.973 ms | 3.44 | 9.2% |

- V1 瓶颈分析：每个输出元素要从 global memory 读 2K 个 float，零数据复用。A 的每个元素被重复读 N 次、B 被读 M 次，全局流量 = 2·M·N·K·4B ≈ 550 GB，除以 40 ms ≈ 13.7 TB/s 的等效需求——远超显存带宽，靠 L2/L1 缓存才没更惨。
- 优化方向（V2）：把 tile 搬进 shared memory，让 A/B 的每个元素从 global 只读一次、在 smem 里被复用几十次。
