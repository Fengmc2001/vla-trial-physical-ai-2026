#!/usr/bin/env bash
# The web process stays CPU-only. Explicitly enabled controls launch a separate
# bounded child, which must pass the ordinary execution/admission checks.
set -euo pipefail
VLA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$VLA_ROOT/.cache/tmp"
export GRADIO_TEMP_DIR="$VLA_ROOT/.cache/live-viewer"
mkdir -p "$TMPDIR" "$GRADIO_TEMP_DIR"
if ! command -v timeout >/dev/null 2>&1; then
  printf '%s\n' '需要系统 GNU timeout 以限制观看服务为两小时；未启动任何服务。' >&2
  exit 1
fi
cd "$VLA_ROOT"
exec timeout --signal=TERM --kill-after=10s 7200s "$VLA_ROOT/.venv/bin/python" -m vla_trial.live_ui --project-root "$VLA_ROOT"
