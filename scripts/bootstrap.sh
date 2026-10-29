#!/usr/bin/env bash
# Install libraries in this project only. Never load a model or start a GPU job.
set -euo pipefail
VLA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$VLA_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export TMPDIR="$VLA_ROOT/.cache/tmp"
export UV_CACHE_DIR="$VLA_ROOT/.cache/uv"
mkdir -p "$TMPDIR"
VLA_PYTHON="${VLA_PYTHON:-python3.11}"
if [ ! -x .venv/bin/python ]; then
  "$VLA_PYTHON" -m venv .venv
fi
.venv/bin/python -c 'import sys, platform; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "This lock targets Linux x86_64"'
.venv/bin/python -m pip install --no-cache-dir 'uv==0.13.1'
VLA_REQUIREMENTS=requirements.txt
if [ -f requirements.lock.txt ]; then VLA_REQUIREMENTS=requirements.lock.txt; fi
nice -n 10 .venv/bin/uv pip install --python .venv/bin/python --torch-backend cu128 --no-cache -r "$VLA_REQUIREMENTS"
.venv/bin/uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/uv pip check --python .venv/bin/python
printf '%s\n' 'Dependencies installed. GPU, model and simulation have NOT been run.'
