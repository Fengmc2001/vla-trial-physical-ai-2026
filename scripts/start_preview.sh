#!/usr/bin/env bash
set -euo pipefail
VLA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$VLA_ROOT/.cache/tmp"
export GRADIO_TEMP_DIR="$VLA_ROOT/.cache/gradio"
mkdir -p "$TMPDIR" "$GRADIO_TEMP_DIR"
exec "$VLA_ROOT/.venv/bin/python" -m vla_trial --project-root "$VLA_ROOT" ui --host 127.0.0.1 --port "${VLA_UI_PORT:-7866}"
