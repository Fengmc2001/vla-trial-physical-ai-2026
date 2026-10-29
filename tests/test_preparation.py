import json
import subprocess
import sys
from pathlib import Path

from vla_trial.config import load_config, merge
from vla_trial.diagnostics import model_metadata


def test_default_is_inert():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/default.json").read_text())
    assert config["runtime"]["enabled"] is False
    assert config["runtime"]["approval_receipt"] == ""
    assert config["simulation"]["render_device_verified"] is False
    assert config["model"]["path"] == ""


def test_private_overlay_preserves_defaults():
    original = {"runtime": {"enabled": False}, "model": {"path": "", "local_files_only": True}}
    result = merge(original, {"model": {"path": "/existing/cache"}})
    assert result["model"]["local_files_only"] is True
    assert result["runtime"]["enabled"] is False
    assert original["model"]["path"] == ""


def test_no_configured_model_does_not_download():
    assert model_metadata("")["status"] == "NOT_CONFIGURED"


def test_missing_shard_is_not_reported_complete(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{"x":"missing.safetensors"}}')
    report = model_metadata(str(tmp_path))
    assert report["status"] == "MISSING_SHARDS"
    assert report["weights_loaded"] is False


def test_doctor_does_not_import_gpu_libraries():
    code = "from vla_trial.config import load_config; from vla_trial.diagnostics import diagnose; import json; print(json.dumps(diagnose(load_config())))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30)
    report = json.loads(result.stdout)
    assert report["heavy_modules_loaded"] == []
    assert report["gpu_query_performed"] is False
    assert report["graphics_context_created"] is False


def test_ui_is_loopback_only():
    result = subprocess.run([sys.executable, "-m", "vla_trial", "ui", "--host", "0.0.0.0"], capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_preview_import_guard_blocks_runtime_imports():
    code = '''
import sys
from vla_trial.ui import PreparationImportGuard
sys.meta_path.insert(0, PreparationImportGuard())
for name in ('torch', 'genesis', 'transformers', 'quadrants'):
    try:
        __import__(name)
    except ImportError as e:
        assert 'PREPARATION_ONLY' in str(e)
    else:
        raise AssertionError(name)
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_preview_fixture_matches_strict_schema():
    from vla_trial.preview import FIXTURES, fixture_plan
    from vla_trial.schema import validate_plan
    for language in FIXTURES:
        sample = fixture_plan(language)
        assert sample["metadata"]["source"] == "HAND_WRITTEN_SCHEMA_EXAMPLE_NOT_MODEL_OUTPUT"
        assert len(validate_plan(sample["plan"], protected_ids=["red_block"]).actions) == 3
