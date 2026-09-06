# GPU 共享与调度机制笔记

围绕"RTX 5080 上 WSL 跑 LLM + 前台跑 DX12 游戏如何共存"展开的 GPU 体系结构讨论。记录时间：2026-08-19。

## 1. 多负载共存：显存 vs SM/带宽

- **显存是静态账本**：加法式瓜分、不可超订（WDDM 换页对 CUDA 负载等于性能崩塌）。
  - 估算实例（RTX 5080 16GB）：Windows 桌面 ~0.5–1GB；3A 游戏 6–10GB；MiniCPM-o Q4_K_M 单实例 ≈ 6.4GB（权重 4.4 + mmproj 1 + KV 0.5 + compute buffer 0.5）
  - 结论：1 LLM 实例 + 游戏紧但可行；2 实例 + 游戏必然 OOM
- **SM 不能静态划分**：消费卡无 MIG，跨负载只能抢占式分时。LLM tg 是 memory-bound，游戏还抢共享的 960GB/s 显存带宽 → 实测互损比纯 SM 争抢更严重（LLM tg 可能 160→60–100 t/s，游戏帧率掉 20–40%）
- **最有效缓解手段：给游戏锁帧**（直接减少其 DMA buffer 提交频率，让出时间量子和带宽）
- **多 LLM 并发首选单实例 `--parallel N`**（KV cache 分 slot，省整份权重显存），优于多开进程

## 2. 三层并行机制对照（核心结论表）

| 层次 | 机制 | 并行性质 |
|---|---|---|
| 同一队列/context 内 | block/wave 在 SM 上空间共存 | 真并行（寄存器/shared mem/warp 槽够分时） |
| 同进程跨队列（图形 vs compute） | async compute：硬件多命令队列 + fence 编排 | 真并行，可精细控制 |
| CUDA 进程间 | MPS（Volta+）可空间共享 | 真并行（需显式开 MPS） |
| CUDA vs DX12 跨进程（如 LLM vs 游戏） | WDDM/HAGS 硬件调度器轮转 | **只能分时抢占，无 fence 可编排** |

## 3. CUDA 与图形管线的概念对应

| CUDA | 图形管线 | 对应精度 |
|---|---|---|
| thread | 一次 VS 顶点 / PS 片元调用 | 精确 |
| warp（32 线程锁步） | 打包成 32 宽 wave 执行的顶点/像素（PS 按 2×2 quad 组织） | 精确（同一套 SIMT 硬件机制） |
| thread block | Compute Shader 的 thread group | 精确 |
| thread block | VS/PS 的一批顶点/像素 | 仅近似——分组是硬件行为，程序不可见、无 shared memory、无 barrier |

关键差异：CUDA block 是程序员掌控的协作单元（`__shared__` + `__syncthreads()`）；VS/PS 调用间完全隔离。需要像素间协作的图形算法（tile-based 光照、屏幕空间效果）因此都用 Compute Shader 实现。

## 4. Compute Shader ≈ CUDA 执行模型

术语一一对应：grid↔Dispatch、block↔thread group（`[numthreads]`）、`blockIdx`↔`SV_GroupID`、`threadIdx`↔`SV_GroupThreadID`、`__shared__`↔`groupshared`、`__syncthreads()`↔`GroupMemoryBarrierWithGroupSync()`、warp↔wave（NV 32 / AMD 64）、裸指针↔UAV/RWStructuredBuffer。

- 硬件执行完全相同：group 独占驻留 SM、wave 锁步、occupancy 规则一致
- **CS 优势**：与渲染管线零成本互通（RT/深度/纹理/采样器硬件），适合帧内算法
- **CUDA 优势**：cooperative groups、dynamic parallelism、warp 原语更成熟、统一内存、cuBLAS/cuDNN 生态、跨平台
- 互相收敛：HLSL 有 wave intrinsics/WaveMatrix（≈WMMA），CUDA 有 cudaGraph
- 实践结论：帧内图形算法用 CS；纯计算/推理用 CUDA（llama.cpp 选 CUDA 的根本原因）

## 5. Async Compute 实现原理

**目标**：填掉图形管线气泡（状态切换空泡、shadow/后处理等低占用阶段），纯图形队列 SM 利用率典型只有 60–80%，收益约 5–15% 帧率。

**硬件基础**：
- 多命令处理器：AMD GCN 起 8×ACE；NVIDIA 1 图形队列 + 多 compute/copy 硬件通道（Pascal 后真硬件支持）
- SM 共存：图形 wave 与 compute thread group 同时驻留不同 SM（或同 SM 不同 warp 槽），无静态划分；图形突然吃满时 compute 排队等位或被抢占——opportunistic 填空，非带宽保证

**DX12 机制**：
```cpp
// 两条独立硬件队列
CreateCommandQueue(DIRECT)  → graphicsQueue
CreateCommandQueue(COMPUTE) → computeQueue
// fence 做跨队列 GPU 时间线同步（非 CPU 阻塞）
computeQueue->Signal(fence, v);
graphicsQueue->Wait(fence, v);   // 放在真正需要数据的最后一刻
```
- 同步粒度决定收益：Wait 越靠后重叠窗口越大；提交即 Wait 退化为串行
- 资源跨队列需 barrier，开销必须小于并行收益
- 典型帧编排：帧 N 后处理 ∥ 帧 N+1 shadow/geometry pass；物理/粒子/动画剔除塞 compute 队列
- Vulkan 对应：queue family + timeline semaphore

## 6. WDDM 抢占调度机制

**架构**：内核态 `dxgkrnl.sys` = VidMm（显存分页）+ VidSch（GPU 时间分配）。应用命令经 UMD 攒成 DMA buffer，由 VidSch/HAGS 决定何时上 GPU。

**抢占粒度演进**：

| 世代 | 粒度 | 效果 |
|---|---|---|
| WDDM 1.0 | DMA buffer 级 | 长 shader 卡死 GUI；TDR（~2s 强制重置 GPU）由此而生 |
| WDDM 1.3 | 图元级 | 长 pass 不再冻桌面 |
| WDDM 2.0 | 指令级 | 任意点抢占，保存/恢复全部流水线状态到显存 context save area |

**HAGS（硬件加速 GPU 调度，Win10 2004+，现代默认开）**：
- 传统：VidSch 在 CPU 内核态排队排序（有内核往返开销）
- HAGS：各进程队列直接映射给 GPU，**GPU 固件自行做指令级抢占轮转**（NV Pascal+ 硬件支持），OS 只管粗粒度优先级/配额

**调度输入**：进程优先级带（Windows 图形设置）、时间量子轮转（毫秒级，避免 save/restore 开销失控）、dwm.exe 永远最高实时优先级（所以游戏满载桌面仍可动）、TDR 兜底。

**两层调度器边界**：
- VidSch/HAGS：决定**哪个进程**上 GPU（跨进程共享只发生在这层）
- GPU 内部 warp 调度器：决定已上 GPU 的 block 如何在 SM 间摆放（每个租户的时间片内看到整张卡）

**对 WSL CUDA + 游戏场景的含义**：跨进程跨 API 无 fence 可用，可调手段只有 Windows 每应用 GPU 偏好和游戏锁帧。
