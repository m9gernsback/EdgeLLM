# MiniCPM-o 2.6 本地部署工作流程

目标：下载 MiniCPM-o 2.6 BF16 → 本地量化为 GGUF → llama.cpp（CUDA）启动 OpenAI 兼容服务 → 测评 → 供 Workbuddy 等 Harness 接入。

> 已知限制：llama.cpp 仅支持 MiniCPM-o 2.6 的**文本 + 图像**能力，音频输入/输出不支持（官方文档 `LlamaCpp/docs/multimodal/minicpmo2.6.md`）。

## 环境基线（已确认）

- WSL2 (Ubuntu 24.04)，32 核 CPU，45GB RAM
- GPU：RTX 5080 16GB（Blackwell，sm_120），CUDA toolkit 13.2 位于 `/usr/local/cuda`
- 磁盘：`/mnt/e` 余 1.1TB；`~` 余 899GB
- Conda 环境：`MiniCPM`（Python 3.12，已装 huggingface_hub、cmake）

## 目录约定（PascalCase）

| 路径 | 内容 |
|---|---|
| `MiniCpmO26/Models/MiniCPM-o-2_6/` | HF 原始 BF16 权重（~16GB） |
| `MiniCpmO26/Gguf/` | 转换产物：BF16 GGUF、mmproj、量化版本 |
| `LlamaCpp/` | llama.cpp 源码与 `build/` 编译产物 |

## 流程步骤

### 1. 下载模型（后台进行中，task: mwTEna）
```bash
conda activate MiniCPM
hf download openbmb/MiniCPM-o-2_6 --local-dir MiniCpmO26/Models/MiniCPM-o-2_6
```

### 2. 编译 llama.cpp（CUDA）
```bash
conda activate MiniCPM   # cmake 在此环境
export PATH=/usr/local/cuda/bin:$PATH
cd LlamaCpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j32
```
产物：`build/bin/` 下 `llama-server`、`llama-bench`、`llama-quantize`、`llama-mtmd-cli`。

### 3. HF → GGUF 转换 + 量化
需先装转换依赖：`pip install -r LlamaCpp/requirements.txt`（torch/transformers 等，CPU 版 torch 即可）。

```bash
cd LlamaCpp
# 3.1 拆分：LLM 部分 → model/，投影器 → minicpmv.projector
python tools/mtmd/legacy-models/minicpmv-surgery.py -m ../MiniCpmO26/Models/MiniCPM-o-2_6
# 3.2 视觉编码器 + 投影器 → mmproj GGUF（o 2.6 用 --minicpmv_version 4）
python tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py \
  -m ../MiniCpmO26/Models/MiniCPM-o-2_6 \
  --minicpmv-projector ../MiniCpmO26/Models/MiniCPM-o-2_6/minicpmv.projector \
  --output-dir ../MiniCpmO26/Gguf/ --minicpmv_version 4
# 3.3 LLM 部分（Qwen2 架构）→ BF16 GGUF
python convert_hf_to_gguf.py ../MiniCpmO26/Models/MiniCPM-o-2_6/model \
  --outfile ../MiniCpmO26/Gguf/MiniCPM-o-2_6-BF16.gguf
# 3.4 量化
./build/bin/llama-quantize ../MiniCpmO26/Gguf/MiniCPM-o-2_6-BF16.gguf \
  ../MiniCpmO26/Gguf/MiniCPM-o-2_6-Q4_K_M.gguf Q4_K_M
```
量化档位：Q4_K_M（~4.9GB，主目标）；RTX 5080 余量够，可加做 Q8_0（~8.5GB）对比精度。

### 4. 启动 OpenAI 兼容服务
```bash
./build/bin/llama-server \
  -m ../MiniCpmO26/Gguf/MiniCPM-o-2_6-Q4_K_M.gguf \
  --mmproj ../MiniCpmO26/Gguf/mmproj-model-f16.gguf \
  -ngl 99 -c 8192 --host 0.0.0.0 --port 8080 \
  --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05
```
- `--host 0.0.0.0`：让 Windows 侧经 WSL2 localhost 转发或局域网访问
- `-ngl 99`：全部层 offload 到 GPU

### 5. 测评
- 功能：`curl http://localhost:8080/v1/chat/completions` 分别验证纯文本与图文（image_url base64）请求
- 性能：`llama-bench -m <gguf> -ngl 99` 记录 pp/tg tokens/s；对比 Q4_K_M vs Q8_0

### 6. Workbuddy 接入配置
- base_url：`http://localhost:8080/v1`（Windows 侧不通则用 WSL IP：`hostname -I`）
- api_key：任意占位（如 `sk-local`）
- model：`minicpm-o-2_6`（以 `/v1/models` 返回为准）

## 风险与备选

- 若 mmproj 量化报错：mmproj 保持 f16，只量化文本模型（官方做法）
- 若新版转换器行为变化：备选方案是直接下载官方已转换的 `openbmb/MiniCPM-o-2_6-gguf`，仅用本地 llama-quantize 做量化

---

## 执行结果（2026-08-18，已验证）

### 踩坑记录（转换阶段）

1. transformers 新版（4.48/4.57 均复现）对本地 trust_remote_code 模型拷贝 remote code 时会漏文件（`image_processing_minicpmv.py`、`tokenization_minicpmo_fast.py`）。解决：手动补拷到 `~/.cache/huggingface/modules/transformers_modules/MiniCPM-o-2_6/`。
2. surgery 拆出的 `model/` 目录 config 残留 `auto_map`（指向自定义代码）且 `model_type=minicpmo`，新版转换器拒绝加载。解决：删 `auto_map`、`model_type` 改 `qwen2`；`tokenizer_config.json` 同样删 `auto_map`、`tokenizer_class` 改 `Qwen2TokenizerFast`；删除 model/ 下三个自定义 .py 文件。**注意保留 chat_template**。

### 产物（`MiniCpmO26/Gguf/`）

| 文件 | 大小 | 用途 |
|---|---|---|
| MiniCPM-o-2_6-BF16.gguf | 15G | 基准/再量化母本 |
| MiniCPM-o-2_6-Q4_K_M.gguf | 4.4G | 主推部署档 |
| MiniCPM-o-2_6-Q8_0.gguf | 7.6G | 高精度对比档 |
| mmproj-model-f16.gguf | 997M | 视觉投影（保持 f16） |

### 测评（RTX 5080，-ngl 99，pp512/tg128）

| 量化 | pp (t/s) | tg (t/s) |
|---|---|---|
| Q4_K_M | 8532 | 159.8 |
| Q8_0 | 8501 | 99.7 |

功能验证：纯文本对话正常；图文对话正确识别形状/颜色并读出图中文字。

### Workbuddy / Harness 接入

服务常驻命令见上文第 4 步。接入参数：

- **API 类型**：OpenAI 兼容
- **base_url**：`http://localhost:8080/v1`（WSL2 localhost 转发通常可用；不通则用 `http://172.17.61.59:8080/v1`，IP 以 `hostname -I` 为准）
- **api_key**：任意占位，如 `sk-local`
- **model**：`minicpm-o-2_6`
- **能力**：文本 + 图像（image_url / base64）；不支持音频
