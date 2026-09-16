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
| V1 shared-mem tree | 0.548 ms | 489 GB/s |
| V2 warp shuffle | 0.371 ms | 722 GB/s |

- V1 慢的根因：树形归约每一轮都要 `__syncthreads`，最后 5 轮（≤32 元素）只有 1 个 warp 在干活，其余 warp 空等。
- V2 用 `__shfl_down_sync` 寄存器交换替代最后 5 轮 shared memory + barrier，带宽提升 ~48%。
- 遗留问题（阶段 1 前可思考）：两个版本都只用了"一线程一元素"，每个 block 读完 1KB 就结束，grid 高达 262144 个 block、block 调度开销占比不小；优化方向是每线程先串行累加多个元素（grid-stride）再进 block 归约。
