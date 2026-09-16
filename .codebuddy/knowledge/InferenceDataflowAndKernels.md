# 推理数据流带宽账本与算子框架笔记

推理时的显存带宽明细账、CUTLASS/Triton 算子框架、以及"图形管线为什么没有对应框架"的分析。记录时间：2026-08-19。实例基准：MiniCPM-o 2.6（7.61B，28 层，hidden 3584，28 Q 头/4 KV 头，head_dim 128，Q4_K_M 4.4GB，RTX 5080 带宽 960GB/s）。

## 1. 推理带宽账本（实测自洽）

### Decode（每生成 1 token）

| 项目 | 流量 | 说明 |
|---|---|---|
| 权重读取 | 4.4 GB | 恒定主导（8K ctx 时 ~82%） |
| KV cache 读取 | 1K:0.11GB / 8K:0.92GB / 32K:3.7GB / 128K:15GB | 公式：2×28层×4头×128维×ctx×2B；32K 后与权重同级，长上下文主瓶颈 |
| KV cache 写回 | 57 KB | 永远可忽略 |
| 算子间激活中转 | ~8 MB | <0.2%，且多数命中 L2 不落 DRAM |
| 注意力分数 | ≈0 | FlashAttention 分块 SRAM 消化，不物化 |

验证：8K ctx 总流量 5.3GB/token ÷ 6.3ms（159.8 t/s）≈ 840GB/s = 峰值 88% → decode 被带宽焊死。

### Prefill（N=512 一批）

- 权重**整批只读一遍**（4.4GB，与 N 无关 → pp 摊薄权重成本的原理）
- 激活中转 ~2GB 但 buffer 仅 3.7MB，高比例 L2 命中；KV 写入 29MB；注意力分数被 FA 压到 ≈0
- 实测带宽占用 ~85GB/s（峰值 9%）→ pp 是 compute-bound，量化不提速 pp

### 优化映射

| 战线 | 手段 |
|---|---|
| 权重（decode 恒定主导） | 权重量化、MoE（只读激活专家） |
| KV 读取（O(ctx) 线性） | GQA（已省 7×）、KV 量化（--cache-type-k q8_0）、滑窗注意力 |
| 注意力分数 O(ctx²) | FlashAttention（必选） |
| 激活中转 | kernel 融合（fused_rms_norm/RoPE）、L2 亲和 |

## 2. CUTLASS vs Triton

两者都是 kernel 开发框架，介于裸 CUDA C 与 cuBLAS 现成库之间。

**CUTLASS（NVIDIA，C++ 模板）**
- 层级积木：thread/warp/block/grid 四级 tiling 各自可替换；CuTe 用 Layout/Tensor 代数编译期推导地址与 swizzle
- 性能天花板 = 硬件峰值；FlashAttention-3、TensorRT-LLM、cuBLAS 均基于它；Blackwell 新特性（FP4、tcgen05）最先支持
- 代价：学习曲线陡、编译慢、开发以天计

**Triton（OpenAI，Python DSL）**
- block 级编程视角（tl.load 操作整个 tile），编译器自动处理线程分工、SRAM staging、swizzle
- 性能 80–95% 峰值，开发以小时计；跨 NVIDIA/AMD/Intel
- PyTorch torch.compile 后端、vLLM、xFormers 大量使用

选型：NVIDIA 极限性能/最新硬件特性 → CUTLASS；快速原型/跨硬件/PyTorch 集成 → Triton。社区现状："Triton 原型，CUTLASS 攻坚"。llama.cpp 例外——为兼容十余种后端选择手写 CUDA kernel + ggml 抽象层。

## 3. 图形管线为何没有 CUTLASS/Triton 对应物

对应关系：CUDA C ↔ HLSL/GLSL/MSL（语言层存在）；CUTLASS/Triton ↔ **空白**。

原因——负载结构不同：
- CUTLASS/Triton 优化的是**数据复用编排**（GEMM/attention 的 tiling、SRAM staging、喂 Tensor Core）
- 图形 shader 是**流式负载**：像素进来→采样→光照→写出，无跨线程数据复用、无 tile 编排；重活在固定功能硬件（光栅化器/ROP/纹理单元）
- shader 优化面只有寄存器压力/指令数/occupancy，由驱动编译器+手写 HLSL 解决
- 图形世界的性能框架在**引擎层**（合批、PSO 排序、tile-based deferred），相当于 CUDA 的 stream/graph 编排而非 kernel 内优化

最接近的存在：Slang 语言（泛型/模块/自动微分）、DX12 Work Graphs（≈dynamic parallelism）、shader permutation 系统（变体管理，非性能编排）。

**反证**：图形中唯一 GEMM 级负载是 DLSS/FSR4 的 AI 部分——各家均直接把预编译 CUDA/TensorRT kernel 塞进渲染管线，而非在 HLSL 世界重建框架。两世界边界不在硬件（SM 共享），而在负载类型决定的工具链：**流式负载出语言，复用型负载出框架**。
