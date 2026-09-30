#!/usr/bin/env bash
# 对照实验一次只启动一个服务；配置和日志保存到 results/。
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-baseline}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
TOKEN_BUDGET="${TOKEN_BUDGET:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# 沿用用户已跑通的国内模型源；本地目录也可作为 MODEL。
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"
flags=(--no-enable-prefix-caching --enable-chunked-prefill)
case "$PROFILE" in
  baseline) ;;
  serial) MAX_NUM_SEQS=1 ;;
  prefix) flags=(--enable-prefix-caching --enable-chunked-prefill) ;;
  wide-prefill) TOKEN_BUDGET=4096 ;;
  small-kv) flags+=(--kv-cache-memory-bytes 1073741824) ;;
  *) echo "用法: bash serve.sh {baseline|serial|prefix|wide-prefill|small-kv}" >&2; exit 2 ;;
esac
command -v vllm >/dev/null || { echo "请先激活服务器上的 vLLM 环境" >&2; exit 1; }
mkdir -p results
run_stamp="$(date +%Y%m%d-%H%M%S)-$$"
args=(serve "$MODEL" --served-model-name core-lab --host 127.0.0.1 --port "$PORT"
  --dtype half --generation-config vllm --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$TOKEN_BUDGET" --enable-prompt-tokens-details)
{
  vllm --version
  printf 'VLLM_USE_MODELSCOPE=%s\n' "$VLLM_USE_MODELSCOPE"
  printf '%q ' vllm "${args[@]}" "${flags[@]}"
  printf '\n'
  nvidia-smi || true
} | tee "results/server-${PROFILE}-${run_stamp}.config.txt"
vllm "${args[@]}" "${flags[@]}" 2>&1 | tee "results/server-${PROFILE}-${run_stamp}.log"
