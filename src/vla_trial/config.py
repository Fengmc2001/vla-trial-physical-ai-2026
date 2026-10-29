"""Private machine settings overlay a public, inert configuration."""
import json
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def merge(base, overlay):
    result = dict(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(root=None):
    root = Path(root) if root else project_root()
    config = json.loads((root / "configs/default.json").read_text())
    private = root / "configs/local.json"
    if private.exists():
        config = merge(config, json.loads(private.read_text()))
    return config
