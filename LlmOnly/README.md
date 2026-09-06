# LLM-Only Deployment

On-device deployment of text-only large language models on edge devices (iPhone 15, Android).

## Frameworks

- **llama.cpp** — primary choice; GGUF models, runs on iOS (Metal) and Android, largest ecosystem.
- **MLX / Core ML** — Apple-only path with Neural Engine acceleration.
- **MLC LLM** — AOT-compiled, best GPU utilization, also reaches WebGPU.

## Candidate models (4-bit quantized)

- Qwen3 1.7B / 4B
- Gemma 3 1B / 4B
- Llama 3.2 1B / 3B
- Phi-4-mini

## Targets

- iPhone 15 (A16, 6 GB RAM) — 3–4B @ 4-bit fits comfortably.
- Android emulator — functional testing only (CPU fallback, no GPU/NPU).
