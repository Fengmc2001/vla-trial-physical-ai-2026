"""Create an offline source archive; never publish, include weights, or include machine paths."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
import os
import tempfile


PUBLIC_RUN_FILES = (
    "README.md", "status.json", "evaluation.json", "task-contract.json", "actions.json",
    "action-milestones.json", "initial.json", "final.json", "initial.png", "final.png",
    "trajectory.json", "executed-plan.json", "control-events.json", "events.jsonl",
    "source-manifest.json", "runtime-environment.json", "trial-summary.json", "simulation.mp4",
    "inference/status.json", "inference/events.jsonl", "inference/runtime-environment.json",
    "inference/source-manifest.json", "inference/plan.json", "inference/model-output.json",
)
V2_REQUIRED_FILES = set(PUBLIC_RUN_FILES) - {"trial-summary.json", "inference/source-manifest.json"}


def safe_project_file(root, path):
    """Reject escaping paths and symlinked ancestors as well as symlink files."""
    if not path.resolve(strict=True).is_relative_to(root):
        raise SystemExit("Export input escapes this project")
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"Unexpected source symlink: {relative}")
    if not path.is_file():
        raise SystemExit(f"Expected an export file: {relative}")


def read_evidence_json(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite evidence JSON")))


def check_public_json(value):
    private_keys = {"pid", "gpu_uuid", "gpu_uuids", "host", "hostname", "operator", "allocation_reference",
                    "admission_library", "model_path", "site_config", "approval_receipt",
                    "admission_output", "project_root", "cuda_visible_devices"}
    if isinstance(value, dict):
        if any(str(key).lower() in private_keys for key in value):
            raise SystemExit("Public run metadata still contains machine/operator fields")
        for child in value.values():
            check_public_json(child)
    elif isinstance(value, list):
        for child in value:
            check_public_json(child)
    elif isinstance(value, str):
        if re.search(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value):
            raise SystemExit("Public run metadata still contains a device UUID or email")


def verify_public_multistep(root, directory):
    """Refuse an incomplete or unscored folder named validated_multistep_run."""
    for name in V2_REQUIRED_FILES:
        path = directory / name
        if not path.exists():
            raise SystemExit(f"Incomplete public multistep bundle: {name}")
        safe_project_file(root, path)
    status = read_evidence_json(directory / "status.json")
    evaluation = read_evidence_json(directory / "evaluation.json")
    contract = read_evidence_json(directory / "task-contract.json")
    if (status.get("stage") != "full" or status.get("phase") != "completed"
            or status.get("model_used") is not True or status.get("video_finalized") is not True
            or status.get("test_fixture") or evaluation.get("success") is not True
            or evaluation.get("evaluation_complete") is not True or evaluation.get("contract") != contract
            or contract.get("schema_version") != 2):
        raise SystemExit("Public multistep bundle must be a completed independently successful real v2 run")
    sys.path.insert(0, str(root / "src"))
    from vla_trial.evaluation_multistep import evaluate_multistep_task
    from vla_trial.schema import parse_plan, validate_plan
    trajectory = read_evidence_json(directory / "trajectory.json")
    initial = read_evidence_json(directory / "initial.json")
    final = read_evidence_json(directory / "final.json")
    milestones = read_evidence_json(directory / "action-milestones.json")
    rescored = evaluate_multistep_task(contract, initial["object_positions"], trajectory, milestones)
    for key, value in rescored.items():
        if key != "evaluation_scope" and evaluation.get(key) != value:
            raise SystemExit(f"Public multistep score does not reproduce: {key}")
    if trajectory[0]["positions"] != initial["object_positions"] or trajectory[-1]["positions"] != final["object_positions"]:
        raise SystemExit("Public observation metadata and trajectory differ")
    options = {"object_ids": contract["active_object_ids"], "protected_ids": contract["protected_ids"],
               "max_actions": 12, "allow_repeated_objects": True}
    plan = validate_plan(read_evidence_json(directory / "inference/plan.json"), **options).to_dict()
    model = read_evidence_json(directory / "inference/model-output.json")
    executed = read_evidence_json(directory / "executed-plan.json")
    inference_status = read_evidence_json(directory / "inference/status.json")
    action_results = read_evidence_json(directory / "actions.json")
    if (parse_plan(model.get("raw_output"), **options).to_dict() != plan
            or executed.get("source") != "QWEN_REAL_INFERENCE" or executed.get("plan") != plan
            or model.get("instruction") != contract["instruction"] or status.get("instruction") != contract["instruction"]
            or model.get("image_sha256") != hashlib.sha256((directory / "initial.png").read_bytes()).hexdigest()):
        raise SystemExit("Public model input/output and executed plan do not agree")
    expected_actions = [action for action in plan["actions"] if action["action"] != "done"]
    if ([milestone["action"] for milestone in milestones] != expected_actions
            or not isinstance(action_results, list) or len(action_results) != len(expected_actions)
            or any(not isinstance(actual, dict) or any(actual.get(key) != value for key, value in expected.items())
                   for actual, expected in zip(action_results, expected_actions))
            or inference_status.get("stage") != "infer" or inference_status.get("phase") != "completed"
            or model.get("settings", {}).get("task_mode") != "multistep"):
        raise SystemExit("Public inference status and actual action sequence do not match the model plan")


def public_run_sources(root):
    files = []
    for folder in ("validated_run", "validated_multistep_run"):
        directory = root / "examples" / folder
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise SystemExit("Public evidence directory must be a real project directory")
        if folder == "validated_multistep_run":
            verify_public_multistep(root, directory)
        files.extend(directory / name for name in PUBLIC_RUN_FILES if (directory / name).exists())
    return files


def delivery_sources(root):
    """Optional reviewed final files only; never walk arbitrary output trees."""
    directory = root / "交付" / "最终提交"
    if not directory.is_dir() or directory.is_symlink():
        raise SystemExit("Reviewed final delivery directory is not ready")
    patterns = {"video": re.compile(r"PAI最終課題_([A-Za-z0-9_.-]{1,64})\.mp4\Z"),
                "report": re.compile(r"説明資料_PAI最終課題_([A-Za-z0-9_.-]{1,64})\.pdf\Z")}
    academic_directory = root / "交付" / "論文形式レポート"
    if academic_directory.is_symlink():
        raise SystemExit("Academic report directory must not be a symlink")
    selected = {}
    for kind, pattern in patterns.items():
        source_directory = academic_directory if kind == "report" and academic_directory.is_dir() else directory
        matches = [(path, pattern.fullmatch(path.name)) for path in source_directory.iterdir() if pattern.fullmatch(path.name)]
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one final {kind} with the course filename in the final delivery directory")
        selected[kind] = matches[0]
    if selected["video"][1].group(1) != selected["report"][1].group(1):
        raise SystemExit("Final video and PDF account names differ")
    return [(path, "delivery/" + path.name) for path, _ in selected.values()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/尝试VLA-源码草稿.zip")
    parser.add_argument("--include-delivery", action="store_true", help="Include the reviewed final MP4/PDF from the final delivery folder as delivery/; requires a complete public v2 bundle")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "dist"):
        raise SystemExit("Output must stay under this project's dist directory")
    output.parent.mkdir(exist_ok=True)
    sources = [root / "README.md", root / "pyproject.toml", root / "requirements.txt", root / "requirements-media.txt", root / "THIRD_PARTY_NOTICES.md", root / "configs/default.json"]
    sources.extend((root / "configs/tasks").glob("*.json"))
    sources.append(root / "configs/site.example.json")
    for directory in ("src", "tests"):
        sources.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".") and p.suffix in (".py", ".md"))
    # Development plans and draft reports contain internal status and unfinished
    # result fields. Export only the public instructions and marked template.
    sources.extend(root / "docs" / n for n in ("SUBMISSION_CHECKLIST.md", "REPORT_TEMPLATE_JA.md", "USER_GUIDE_ZH.md", "SOURCE_PROVENANCE.md", "EXPERIMENT_PROTOCOL.md"))
    if (root / "docs" / "ACADEMIC_REPORT_JA.md").is_file():
        sources.append(root / "docs" / "ACADEMIC_REPORT_JA.md")
    sources.extend(root / "scripts" / n for n in ("bootstrap.sh", "cpu_check.sh", "start_preview.sh", "start_live.sh", "export_source.py", "build_submission_video.py", "build_submission_report.py", "build_academic_report.py"))
    sources.extend(public_run_sources(root))
    sources.append(root / ".gitignore")
    if (root / "requirements.lock.txt").exists():
        sources.append(root / "requirements.lock.txt")
    checked = {}
    private_prefixes = (b"/nas/" + b"data_", b"/" + b"Users/", b"/" + b"home/")
    entries = [(path, str(path.relative_to(root))) for path in sorted(set(sources))]
    if args.include_delivery:
        if not (root / "examples/validated_multistep_run").is_dir():
            raise SystemExit("Final delivery inclusion requires the independently checked public v2 bundle first")
        entries.extend(delivery_sources(root))
    for path, archive_name in entries:
        safe_project_file(root, path)
        data = path.read_bytes()
        if any(prefix in data for prefix in private_prefixes):
            raise SystemExit(f"Private machine path found in {path.relative_to(root)}; review before exporting")
        if archive_name.startswith("examples/") and path.suffix in (".json", ".jsonl"):
            values = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()] if path.suffix == ".jsonl" else [read_evidence_json(path)]
            for value in values:
                check_public_json(value)
        if archive_name in checked:
            raise SystemExit("Duplicate archive path")
        checked[archive_name] = data
    handle, temporary = tempfile.mkstemp(prefix=".source-", suffix=".zip", dir=output.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in checked.items():
                archive.writestr(name, data)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(output)


if __name__ == "__main__":
    main()
