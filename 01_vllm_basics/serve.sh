#!/usr/bin/env bash
# 以 OpenAI 兼容 API 的形式启动 vLLM 服务。
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --generation-config vllm
