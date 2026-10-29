#!/usr/bin/env bash
set -euo pipefail
VLA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd "$VLA_ROOT"
"$VLA_ROOT/.venv/bin/python" -m vla_trial doctor
"$VLA_ROOT/.venv/bin/python" -m pytest -q
