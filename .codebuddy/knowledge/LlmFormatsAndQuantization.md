# LLM 部署基础概念笔记

围绕 MiniCPM-o 2.6 部署流程中涉及的关键概念：数值格式、量化、模型文件格式、采样参数、GGUF 转换原理。

## 数值格式：BF16 vs FP16

| 格式 | 符号位 | 指数位 | 尾数位 | 特点 |
|---|---|---|---|---|
| FP16 (Half) | 1 | 5 | 10 | 范围小（±65504），精度高 |
| BF16 | 1 | 8 | 7 | 范围与 FP32 相同，精度低 |

- BF16 设计动机：训练痛点是梯度上/下溢（范围）而非尾数精度；与 FP32 同范围可免 loss scaling
- **GPU 原生支持**：NVIDIA 自 Ampere（RTX 30 系）起 Tensor Core 原生支持 BF16，吞吐与 FP16 相同，无性能损失。格式支持体现在计算单元数据通路，而非寄存器

## 量化：Q8_0 / Q4_K_M 等

- `Q8_0`：权重压到 8 bit 整数；`Q4_K_M`：约 4.5 bit，K 系列分块量化算法，M = Medium 混合精度（关键层 6 bit、普通层 4 bit）
- **为什么 GPU 没有对应计算单元却更快**：llama.cpp 是先把低 bit 整数解压回 FP16/BF16 再计算。加速来自**显存带宽**而非算力：
  - 自回归生成（tg）是 memory-bound——每生成 1 token 需把全部权重读一遍。Q4_K_M（4.4GB）比 Q8_0（7.6GB）少读 42% 数据 → 实测 tg 159.8 vs 99.7 t/s（RTX 5080，带宽 ~960GB/s）
  - 预填充（pp）是 compute-bound——权重只读一次，两者几乎相同（8532 vs 8501 t/s）
- 硬件趋势：INT8 Tensor Core（Turing 起）、FP8（Ada/Hopper 起）、FP4（Blackwell 原生）。K-quants 走解压路线是为兼容性和精度；NVFP4 等原生格式是未来方向

## 模型文件格式：safetensors vs GGUF

- **safetensors**：HF 生态通用权重存储，只有张量数据，需 PyTorch/transformers 加载，面向训练/研究
- **GGUF**：llama.cpp 专用，单文件打包权重 + tokenizer + 架构元数据 + 量化信息，支持 mmap 秒加载，面向本地推理
- 关系：safetensors 是原料，GGUF 是成品（convert → quantize 两步加工）

## GGUF 转换原理（convert_hf_to_gguf.py）

GGUF 文件 = 元数据 KV + 张量索引 + 张量数据三段。转换器做三个提取：

1. **架构元数据**：读 config.json 的 `architectures`，经 `conversion/__init__.py` 的 `TEXT_MODEL_MAP` 分发到 Model 类（如 Qwen2），该类声明 GGUF arch 名和 config 字段映射（`hidden_size → qwen2.embedding_length` 等）
2. **权重映射**：`gguf-py/gguf/tensor_mapping.py` 维护 HF 名→GGUF 规范名的模式映射表（`model.layers.{bid}.self_attn.q_proj → blk.{bid}.attn_q.weight`），遍历 safetensors 逐个改名 + dtype 转换后写入。拓扑知识内置在映射表里，不执行模型 PyTorch 代码——所以新架构必须等 llama.cpp 显式支持
3. **Tokenizer 提取**：词表和 merge 规则（`tokenizer.json` / `vocab.json+merges.txt`）序列化为 `tokenizer.ggml.*` 元数据，special tokens、chat_template 一并写入；运行时由 llama.cpp 内置 BPE 实现分词

MiniCPM-o 2.6 的特殊性：转换器只认标准架构，需先 surgery 拆出 LLM 部分并伪装成标准 Qwen2（改 model_type、删 auto_map），视觉塔/Resampler 走 legacy 专用脚本生成 mmproj GGUF。

## 采样参数

生成时模型输出词表概率分布，参数控制选词方式，作用顺序：temp → repeat-penalty → top-k → top-p → 归一化采样：

- **--temp 0.7**：调节分布平坦度，越低越保守
- **--repeat-penalty 1.05**：已出现词的概率除以该值，抑制复读循环；1.0 = 不惩罚
- **--top-k 100**：只在概率最高的前 100 个候选里挑
- **--top-p 0.8**（核采样）：累积概率达 0.8 即截断，集合大小自适应——模型确定时输出稳，不确定时输出多样

MiniCPM-o 官方推荐：temp 0.7 / top-p 0.8 / top-k 100 / repeat-penalty 1.05（已用于 llama-server 常驻配置）。
