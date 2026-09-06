# LLM + Vision/Audio Deployment

On-device deployment of an LLM alongside vision and/or audio models (mixed model fleet).

## Frameworks

- **ONNX Runtime (onnxruntime-genai)** — one runtime for LLM + vision + audio; NNAPI/CoreML/QNN execution providers.
- **LiteRT (ex-TFLite) / LiteRT-LM** — Android-first, deepest NPU access.
- **Core ML** — Apple-only option covering all model types.
- Note: llama.cpp/MLC are LLM-only runtimes — not suitable here.

## Candidate models

- LLM: Qwen3 4B, Gemma 3 4B (quantized)
- Vision: MobileNet/EfficientNet classifiers, YOLO-nano detection
- Audio: Whisper-tiny/base (speech-to-text)

## Targets

- iPhone 15 (A16, 6 GB RAM)
- Android emulator — functional testing only (CPU fallback).
