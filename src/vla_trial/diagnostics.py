"""Metadata-only diagnostics. No torch/CUDA/Genesis/graphics imports or probes."""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "genesis-world", "gradio", "quadrants", "imageio-ffmpeg")


def model_metadata(path):
    if not path:
        return {"status": "NOT_CONFIGURED", "weights_loaded": False}
    root = Path(path).resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    names = sorted(set(index["weight_map"].values()))
    if any(Path(name).name != name for name in names):
        raise ValueError("Unexpected shard path in index")
    shards = [{"name": n, "exists": (root / n).is_file(), "bytes": (root / n).stat().st_size if (root / n).is_file() else 0} for n in names]
    return {"status": "METADATA_COMPLETE" if shards and all(s["exists"] for s in shards) else "MISSING_SHARDS",
            "model_type": config.get("model_type"), "architectures": config.get("architectures"),
            "revision": root.name, "shards": shards, "total_file_bytes": sum(s["bytes"] for s in shards),
            "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "weights_loaded": False, "weight_content_hash_checked": False}


def diagnose(config):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        model = model_metadata(config["model"]["path"])
    except (OSError, ValueError, KeyError) as error:
        model = {"status": "METADATA_ERROR", "detail": str(error), "weights_loaded": False}
    return {"phase": "PREPARATION_ONLY", "python": platform.python_version(), "host": platform.node(),
            "runtime_enabled": config["runtime"]["enabled"], "packages": packages, "model": model,
            "gpu_query_performed": False, "graphics_context_created": False, "simulation_run": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            "heavy_modules_loaded": sorted(n for n in ("torch", "genesis", "quadrants", "taichi", "cupy") if n in sys.modules)}
