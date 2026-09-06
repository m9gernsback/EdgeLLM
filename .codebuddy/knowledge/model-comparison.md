# 模型对比：MiniCPM-o 2.6 vs Qwen3-VL 4B/8B

> 记录时间：2026-08-19。数据来源：openbmb/MiniCPM-o-2_6 官方模型卡、Qwen3-VL Technical Report（arXiv:2511.21631）。
> 注意：Qwen3-VL 小模型的官方多模态详表以图片发布，未能逐项提取，下文多模态分数以报告正文可考数字为准。

## 参数量构成

**MiniCPM-o 2.6（总 8B，模块化拼装）**

| 模块 | 基础 | 参数量 |
|---|---|---|
| LLM 主干 | Qwen2.5-7B | 7B |
| 视觉编码器 | SigLIP-400M | 0.4B |
| 音频编码器 | Whisper-medium | 0.3B |
| 语音合成 | ChatTTS | 0.2B |

**Qwen3-VL 4B / 8B**：端到端 dense（8B 版含视觉塔共约 9B），原生 256K 上下文，无音频。

## 纯文本能力（Qwen3-VL 技术报告 Table 6，Instruct）

| Benchmark | MiniCPM-o 2.6（≈Qwen2.5-7B 水平） | Qwen3-VL-4B | Qwen3-VL-8B |
|---|---|---|---|
| MMLU-Pro | ~56 | 67.1 | 71.6 |
| GPQA | ~34 | 55.9 | 61.9 |
| AIME-25 | ~13 | 46.6 | 45.9 |
| LiveCodeBench v6 | ~14 | 37.9 | 39.3 |
| IFEval | — | 82.3 | 83.7 |
| BFCL-v3（工具调用） | — | 63.3 | 66.3 |

## 多模态能力

| Benchmark | MiniCPM-o 2.6 | Qwen3-VL-8B |
|---|---|---|
| MMBench-EN | 80.5 | 85.3（Thinking） |
| MMStar | 64.0 | 75.3（Thinking） |
| OCRBench | 897（发布时 <25B SOTA） | 未提取到确切值（官方定性称"极具竞争力"） |
| DocVQA | 93.5 | 未提取到确切值 |
| MMMU val | 50.4 | 未提取到确切值 |
| MathVista mini | 71.9 | 未提取到确切值 |
| Video-MME（无/有字幕） | 63.9 / 67.9 | 官方宣称 8B 接近 Qwen2.5-VL-72B 水平 |

MiniCPM-o 2.6 独有能力（Qwen3-VL 不具备）：
- 音频理解（ASR：AISHELL-1 CER 1.6；LibriSpeech WER 1.7）
- 端到端语音合成/全双工对话、语音克隆
- 实时流式视频理解（StreamingBench 66.0，开源 SOTA）

## 结论

1. **存在代差**：MiniCPM-o 2.6（2025-01）vs Qwen3-VL（2025 年底）。Qwen3-VL-4B 的文本分数已超 MiniCPM-o 的 7B 主干，视觉全面领先。
2. **MiniCPM-o 的差异化仅在音频**，且该优势在 llama.cpp 部署栈中无法兑现（llama.cpp 只支持其文本+图像）。要用音频必须 PyTorch 全栈部署。
3. **工程支持**：Qwen3-VL 是 llama.cpp 一等公民（`conversion/qwen3vl.py` 官方转换路径）；MiniCPM-o 2.6 需 legacy 脚本 + 手动修配置（见 MiniCpmO26DeployWorkflow.md 踩坑记录）。
4. **部署体积**：Qwen3-VL-4B Q4 ≈ 2.5GB，8B Q4 ≈ 5GB；MiniCPM-o 2.6 Q4_K_M = 4.4GB。RTX 5080 16GB 均可全量 offload。

## 选型建议（Workbuddy Harness 场景）

- **文本+图像场景：选 Qwen3-VL-8B（或 4B 求快）** —— 能力更强、转换省事、llama.cpp 原生支持
- **需要音频交互：才考虑 MiniCPM-o 2.6**，但须放弃 llama.cpp，改用 PyTorch/transformers 部署

---

## 附：GGUF 量化实现细节

### 分块量化结构

量化不是全局共用一套区间（离群值会压垮精度），而是每块独立记录缩放参数：

- **Q8_0**（对称）：每 32 权重一块 = 1 个 FP16 scale + 32 个 int8。解码：`w = scale × q`
- **Q4_K_M**（K-quants，非对称超级块）：每 256 权重一个超级块，内含 8 个 32 权重子块
  - 超级块存 1 个 FP16 super-scale + 1 个 FP16 super-min
  - 各子块的 scale/min 本身再压到 6 bit（"区间的区间"也被量化）
  - 256 个 int4 两两打包进 128 字节
  - 解码：`w = scale × q - min`（区间 [min, max] 映射到 [0,15]）
- K-quants 精度优于老款 Q4_0 的原因之一就是这套二级 scale 结构；实际约 4.8 bit/权重（scale 摊销）

### GPU 上的解码方式：融合 kernel，非整体解压

不会在显存里解压出完整 FP16 副本（否则显存省不下来）。llama.cpp CUDA kernel 做 dequant+matmul 融合：

```
显存只读 int4 + scales（带宽省在这里）→ 线程在寄存器就地解码出 FP16
→ 立即与激活乘加 → FP16 用完即弃，不落显存
```

显存占用 = 量化后大小；tg 阶段提速与体积缩减成正比（MiniCPM-o 实测 Q4_K_M 159.8 vs Q8_0 99.7 t/s）。

### 混合精度策略

Q4_K_M 的 "M" = Medium 混合：敏感层（embedding、attn v_proj、ffn_down 等）用 Q6_K，其余 Q4_K，由 llama-quantize 内置规则自动分配。故实测 4.91 BPW 略高于 4。可用 `--output-tensor-type` 自定义逐层档位做更细的体积/质量权衡。
