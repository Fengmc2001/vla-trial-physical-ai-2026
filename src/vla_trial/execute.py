"""Bounded course execution; deliberately separate from the CPU-only preview."""
from __future__ import annotations
import argparse
import fcntl
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Explicit bounded course execution; requires private site authorization")
    parser.add_argument("--stage", choices=("preview", "control", "full", "infer"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--task", default="configs/tasks/minimal_sort_ja.json")
    parser.add_argument("--wall-seconds", type=int, default=1500)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not 30 <= args.wall_seconds <= 1800:
        parser.error("wall-seconds must be in 30..1800")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", GRADIO_ANALYTICS_ENABLED="False")
    os.environ["QD_NUM_THREADS"] = "2"
    os.environ["NUMBA_NUM_THREADS"] = "2"
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[-2:])
    for key, subdir in (("XDG_CACHE_HOME", ".cache/xdg"), ("TMPDIR", ".cache/tmp"), ("QD_CACHE_DIR", ".cache/quadrants")):
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    from .config import load_config
    from .safety import authorize_runtime
    from .admission import release_startup_lock
    config = load_config(root)
    task_path = (root / args.task).resolve()
    if not task_path.is_relative_to(root / "configs/tasks"):
        raise ValueError("Task contract must be inside configs/tasks.")
    task = json.loads(task_path.read_text())
    multistep = task.get("task_mode") == "multistep"
    max_actions = 12 if multistep else 4
    site = json.loads((root / "configs/site.json").read_text())
    if site.get("authorization_date_jst") != datetime.now(timezone(timedelta(hours=9))).date().isoformat():
        raise RuntimeError("Site authorization is not for today; no unattended continuation.")
    (root / "outputs").mkdir(exist_ok=True)
    execution_lock = None
    if args.stage != "infer":
        execution_lock = (root / "outputs/course-execution.lock").open("a+")
        try:
            fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            execution_lock.close()
            raise RuntimeError("Another course scene is running; wait for it to finish. No queued retry.")
    run = args.run_dir.resolve() if args.run_dir else root / "outputs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + args.stage)
    if not run.is_relative_to(root / "outputs"):
        raise ValueError("Run directory must stay inside project outputs.")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "status.json").exists():
        raise ValueError("Run directory already contains evidence; choose a fresh directory.")
    (root / "outputs").mkdir(exist_ok=True)
    if args.stage != "infer":
        save(root / "outputs/live.json", {"run_dir": str(run.relative_to(root))})
    uuid = site["inference_gpu_uuid" if args.stage == "infer" else "simulation_gpu_uuid"]
    now = datetime.now(timezone.utc)
    save(run / "operator-receipt.json", {"authorized": True, "purpose": "vla-trial", "host": site["host"],
         "gpu_uuid": uuid, "operator": site["operator"], "allocation_reference": site["authorization_reference"],
         "issued_at": now.isoformat(), "expires_at": (now + timedelta(minutes=40)).isoformat()})
    runtime = {"enabled": True, "gpu_uuid": uuid, "approval_receipt": str(run / "operator-receipt.json"),
               "execution_profile": "bounded-course-run", "site_config": str(root / "configs/site.json"),
               "admission_output": str(run), "project_root": str(root), "uses_renderer": args.stage != "infer",
               "peak_budget_mib": 18432 if args.stage == "infer" else 6144,
               "allocator_cap_mib": 17300 if args.stage == "infer" else 4096}
    state = {"phase": "admission", "stage": args.stage, "pid": os.getpid(), "gpu_uuid": uuid,
             "instruction": args.instruction, "video_finalized": False, "started_at": now.isoformat()}
    save(run / "status.json", state)
    started_timer = time.monotonic()
    source_files = sorted((root / "src/vla_trial").glob("*.py")) + [root / "requirements.txt", root / "requirements.lock.txt"]
    save(run / "source-manifest.json", {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in source_files if path.is_file() and not path.name.startswith("._")})
    save(run / "task-contract.json", task)
    def phase(value, **details):
        state.update(phase=value, **details)
        save(run / "status.json", state)
        event = {"phase": value, "at": datetime.now(timezone.utc).isoformat(),
                 "wall_elapsed_s": round(time.monotonic() - started_timer, 3), **details}
        with (run / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
    def deadline(signum, frame):
        raise TimeoutError("Bounded course process reached its wall-time limit or was stopped.")
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(args.wall_seconds)
    resource = None
    watch = None
    child = None
    samples = []
    actions = []
    milestones = []
    try:
        grant = authorize_runtime(runtime)
        from importlib.metadata import version
        import torch
        save(run / "runtime-environment.json", {"python": sys.version.split()[0],
             "packages": {name: version(name) for name in ("torch", "torchvision", "genesis-world", "transformers", "gradio")},
             "admitted_gpu_name": torch.cuda.get_device_name(0), "resource_budget_mib": runtime["peak_budget_mib"],
             "physics_backend": "cpu" if args.stage != "infer" else None})
        from .resource_watch import ProcessResourceWatch
        watch = ProcessResourceWatch(uuid, runtime["peak_budget_mib"], run / "resource-watch.json",
                                     wall_seconds=args.wall_seconds).start()
        if args.stage == "infer":
            from .qwen import QwenPlanner
            from PIL import Image
            if not args.image or not args.instruction:
                raise ValueError("Inference requires an actual scene image and instruction.")
            phase("loading_model")
            settings = dict(config["model"], max_pixels=65536, max_new_tokens=512 if multistep else 192,
                            task_mode="multistep" if multistep else "standard", memory_cap_mib=runtime["allocator_cap_mib"],
                            active_object_ids=["red_block", "blue_block", "yellow_block"])
            resource = QwenPlanner(settings, grant)
            resource.load()
            release_startup_lock()
            phase("inferencing")
            try:
                with Image.open(args.image) as image:
                    plan = resource.plan(args.instruction, image.copy(), protected_ids=["red_block"], max_actions=max_actions)
            finally:
                save(run / "model-output.json", {"raw_output": getattr(resource, "last_raw_output", None),
                     "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(), "instruction": args.instruction,
                     "model_revision": Path(config["model"]["path"]).name, "settings": {k:v for k,v in settings.items() if k != "path"}})
            save(run / "plan.json", plan.to_dict())
            phase("completed", plan=plan.to_dict())
            return
        from .simulation import FrankaSimulation
        from .evaluation import load_contract, evaluate_task
        from .schema import validate_plan
        from PIL import Image
        if multistep:
            from .evaluation_multistep import load_multistep_contract, evaluate_multistep_task
            contract = load_multistep_contract(task_path)
        else:
            contract = load_contract(task_path)
        instruction = args.instruction or task["instruction"]
        phase("building_scene", instruction=instruction)
        sim_config = dict(config["simulation"], backend="cpu", active_object_ids=["red_block", "blue_block", "yellow_block"])
        if args.stage == "full":
            sim_config.update(width=960, height=720)
        if multistep:
            criteria = task["success_criteria"]
            block_size = float(criteria["block_height_m"])
            if abs(2 * float(criteria["block_half_width_m"]) - block_size) > 1e-9:
                raise ValueError("The implemented scene requires cubic blocks.")
            sim_config.update(max_video_frames=3200, block_size_m=block_size,
                              physics_substeps=4, noslip_iterations=5,
                              finger_extensions=True, grasp_offset_m=0.165,
                              pair_grasp_offset_m=0.165)
        environment = json.loads((run / "runtime-environment.json").read_text())
        environment["simulation_settings"] = {
            "dt_seconds": float(sim_config.get("dt", 0.01)),
            "physics_substeps": int(sim_config.get("physics_substeps", 1)),
            "noslip_iterations": int(sim_config.get("noslip_iterations", 0)),
            "friction_cone": "pyramidal", "block_size_m": float(sim_config.get("block_size_m", 0.04)),
            "grasp_force_n_per_finger": float(sim_config.get("grasp_force", 2.0)),
            "pair_grasp_force_n_per_finger": float(sim_config.get("pair_grasp_force", 4.0)),
            "finger_extensions": bool(sim_config.get("finger_extensions", False)),
            "grasp_offset_m": float(sim_config.get("grasp_offset_m", 0.11)),
            "pair_grasp_offset_m": float(sim_config.get("pair_grasp_offset_m", 0.125)),
        }
        save(run / "runtime-environment.json", environment)
        initial = None
        origin_step = 0
        def frame(rgb):
            temp = run / "latest.tmp.jpg"
            Image.fromarray(rgb).save(temp, format="JPEG", quality=85)
            temp.replace(run / "latest.jpg")
            if initial is not None:
                sample = {"elapsed_s": round((resource.step_count - origin_step) * resource.dt, 6), "positions": resource.object_positions()}
                if samples and sample["elapsed_s"] == samples[-1]["elapsed_s"]:
                    samples[-1] = sample
                else:
                    samples.append(sample)
        resource = FrankaSimulation(sim_config, grant, video_path=run / "simulation.mp4", on_frame=frame, max_runtime_seconds=900)
        resource.build()
        release_startup_lock()
        initial = resource.object_positions()
        origin_step = resource.step_count
        samples.append({"elapsed_s": 0.0, "positions": initial})
        resource.save_observation(run / "initial.png")
        frame(resource.frame())
        phase("scene_ready")
        if args.stage == "preview":
            resource._step(200)
        else:
            if args.stage == "full":
                phase("waiting_for_qwen")
                infer_dir = run / "inference"
                command = [sys.executable, "-m", "vla_trial.execute", "--stage", "infer", "--project-root", str(root),
                           "--run-dir", str(infer_dir), "--image", str(run / "initial.png"), "--instruction", instruction,
                           "--wall-seconds", "600", "--task", args.task]
                with (run / "inference.log").open("w") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    code = child.wait(timeout=620)
                if code:
                    raise RuntimeError(f"Qwen stage exited with {code}; see inference.log. No automatic retry.")
                payload = json.loads((infer_dir / "plan.json").read_text())
            else:
                payload = {"actions": [{"action": "pick_place", "object_id": "blue_block", "target_id": "left_tray"}, {"action": "done"}]}
                if multistep:
                    payload = {"actions": [
                        {"action": "pick_place", "object_id": "blue_block", "target_id": "left_tray"},
                        {"action": "pick_place", "object_id": "yellow_block", "target_id": "right_tray"},
                        {"action": "push_out", "object_id": "blue_block", "source_target_id": "left_tray"},
                        {"action": "stack", "object_id": "yellow_block", "support_id": "blue_block"},
                        {"action": "move_stack", "base_id": "blue_block", "top_id": "yellow_block", "target_id": "right_tray"},
                        {"action": "done"}]}
            plan = validate_plan(payload, object_ids=resource.active_object_ids, protected_ids=["red_block"],
                                 max_actions=max_actions, allow_repeated_objects=multistep)
            save(run / "executed-plan.json", {"source": "QWEN_REAL_INFERENCE" if args.stage == "full" else "CONTROLLER_VALIDATION_ONLY", "plan": plan.to_dict()})
            # Model startup wait is outside the movement time budget.
            resource.started_at = time.monotonic()
            for action in plan.actions:
                if action.action == "done":
                    break
                phase("moving", action=action.to_dict())
                milestone = {"action": action.to_dict(), "started_s": round((resource.step_count-origin_step)*resource.dt,6), "completed": False}
                milestones.append(milestone)
                save(run / "action-milestones.json", milestones)
                dispatch = {"pick_place": resource.pick_place, "push_out": resource.push_out,
                            "stack": resource.stack, "move_stack": resource.move_stack}
                arguments = {k:v for k,v in action.to_dict().items() if k != "action"}
                result = dispatch[action.action](**arguments)
                result["action"] = action.action
                actions.append(result)
                resource._step(50)
                frame(resource.frame())
                milestone.update(completed=True, ended_s=round((resource.step_count-origin_step)*resource.dt,6))
                save(run / "action-milestones.json", milestones)
                save(run / "actions.json", actions)
            resource._step(100)
        frame(resource.frame())
        resource.save_observation(run / "final.png")
        evaluation = evaluate_multistep_task(contract, initial, samples, milestones) if multistep else evaluate_task(contract, initial, samples)
        if args.stage in ("preview", "control"):
            evaluation = {"evaluation_scope": "camera_validation" if args.stage == "preview" else "controller_validation",
                          "success": None, "whole_task_success": None,
                          "note": "Calibration only; no AI inference was used in this run.",
                          "standard_contract_comparison": evaluation}
        elif instruction != task["instruction"]:
            evaluation = {"evaluation_scope": "custom_task_unscored", "success": None,
                          "whole_task_success": None, "instruction": instruction,
                          "note": "Custom instruction was executed; fixed standard placements cannot score its meaning.",
                          "standard_contract_comparison": evaluation}
        else:
            evaluation["evaluation_scope"] = "standard_task"
        save(run / "evaluation.json", evaluation)
        phase("completed", evaluation=evaluation, model_used=args.stage == "full")
    except BaseException as error:
        phase("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        closed = False
        try:
            if resource is not None:
                if hasattr(resource, "control_events"):
                    save(run / "control-events.json", resource.control_events)
                resource.close()
            closed = True
        except BaseException as cleanup_error:
            state.update(phase="failed", cleanup_error=f"{type(cleanup_error).__name__}: {cleanup_error}")
            raise
        finally:
            release_startup_lock()
            if watch is not None:
                watch.close()
                state["resource_peak_measured_mib"] = watch.record["peak_measured_mib"]
            signal.alarm(0)
            if samples:
                save(run / "trajectory.json", samples)
            state.update(video_finalized=closed and (run / "simulation.mp4").is_file(), finished_at=datetime.now(timezone.utc).isoformat())
            save(run / "status.json", state)
            if execution_lock is not None:
                fcntl.flock(execution_lock, fcntl.LOCK_UN)
                execution_lock.close()


if __name__ == "__main__":
    main()
