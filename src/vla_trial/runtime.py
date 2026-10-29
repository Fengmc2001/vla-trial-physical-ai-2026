"""Safe configuration and prepared future execution; imports only stdlib/core."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .safety import authorize_runtime
from .schema import DEFAULT_OBJECTS, DEFAULT_TARGETS, TaskPlan, validate_instruction, validate_plan


# Host-specific paths belong only in the private local configuration.
DEFAULT_MODEL_PATH = ""
_DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {"enabled": False, "gpu_uuid": "", "approval_receipt": "", "max_actions": 6, "max_model_rounds": 3, "max_runtime_seconds": 180},
    "model": {"path": DEFAULT_MODEL_PATH, "device": "cuda:0", "dtype": "bfloat16", "max_new_tokens": 256, "max_pixels": 262144},
    "simulation": {"dt": 0.01, "width": 640, "height": 480, "fps": 10, "max_video_frames": 1800, "render_device_verified": False},
    "paths": {"output_dir": "outputs"},
    "scene": {"protected_ids": ["red_block"]},
}


@dataclass(frozen=True)
class RuntimeConfig:
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(_DEFAULT_CONFIG))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        if not isinstance(value, Mapping):
            raise ValueError("Configuration must be an object.")
        result = copy.deepcopy(_DEFAULT_CONFIG)
        for section, section_value in value.items():
            if section in result:
                if not isinstance(section_value, Mapping):
                    raise ValueError(f"Configuration section {section} must be an object.")
                result[section].update(copy.deepcopy(dict(section_value)))
            else:
                result[section] = copy.deepcopy(section_value)
        if type(result["runtime"]["enabled"]) is not bool:
            raise ValueError("runtime.enabled must be a JSON boolean.")
        limits = (
            ("runtime", "max_actions", 1, 16), ("runtime", "max_model_rounds", 1, 5),
            ("runtime", "max_runtime_seconds", 1, 3600), ("simulation", "width", 128, 1280),
            ("simulation", "height", 128, 720), ("simulation", "fps", 1, 30),
            ("simulation", "max_video_frames", 1, 5400), ("model", "max_new_tokens", 16, 1024),
            ("model", "max_pixels", 4096, 1048576),
        )
        for section, key, lower, upper in limits:
            item = result[section][key]
            if type(item) is not int or not lower <= item <= upper:
                raise ValueError(f"{section}.{key} must be an integer in [{lower}, {upper}].")
        dt = result["simulation"]["dt"]
        if type(dt) not in (float, int) or not 0.001 <= dt <= 0.02:
            raise ValueError("simulation.dt must be between 0.001 and 0.02 seconds.")
        if not isinstance(result["scene"]["protected_ids"], list):
            raise ValueError("scene.protected_ids must be a list of object IDs.")
        validate_plan({"actions": [{"action": "done"}]}, protected_ids=result["scene"]["protected_ids"])
        return cls(result)

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    @property
    def runtime(self) -> dict[str, Any]:
        return self.data["runtime"]

    @property
    def model(self) -> dict[str, Any]:
        return self.data["model"]

    @property
    def simulation(self) -> dict[str, Any]:
        return self.data["simulation"]

    @property
    def paths(self) -> dict[str, Any]:
        return self.data["paths"]

    @property
    def scene(self) -> dict[str, Any]:
        return self.data["scene"]

    @property
    def enabled(self) -> bool:
        return self.runtime["enabled"]


@dataclass
class RunResult:
    plans: list[TaskPlan]
    actions: list[dict[str, Any]]
    final_frame: Any
    video_path: str
    status: str
    constraints_satisfied: bool


class Runtime:
    def __init__(self, config: RuntimeConfig | Mapping[str, Any], project_root: str | Path):
        self.config = config if isinstance(config, RuntimeConfig) else RuntimeConfig.from_mapping(config)
        self.project_root = Path(project_root).resolve()

    def status(self) -> dict[str, Any]:
        """Configuration-only status: deliberately no GPU query or model import."""
        return {
            "mode": "safe-preparation",
            "runtime_enabled_in_config": self.config.enabled,
            "real_execution_available": False,
            "hardware_validation": "not performed; GPU/renderer/model have not been started",
            "graphics_binding": "unverified; cannot be enabled by a config boolean",
            "model_path": self.config.model["path"],
            "model_local_files_only": True,
            "model_loaded": False,
            "simulation_started": False,
            "camera_frame_available": False,
            "objects": [obj.object_id for obj in DEFAULT_OBJECTS],
            "targets": [target.target_id for target in DEFAULT_TARGETS],
        }

    def preflight(self) -> dict[str, Any]:
        result = self.status()
        model_path = self.config.model["path"]
        result["model_directory_exists"] = bool(
            isinstance(model_path, str) and model_path
            and Path(model_path).is_absolute() and Path(model_path).is_dir()
        )
        result["blockers"] = [
            "runtime.enabled is false" if not self.config.enabled else "explicit runtime flag is enabled, but does not establish resource permission",
            "a fresh host/GPU-specific operator receipt and live idle checks are required",
            "EGL/CUDA physical device binding is not yet implemented and validated",
            "real model inference, grasp control, rendering and video export remain untested",
        ]
        return result

    def run(
        self, instruction: str, *, protected_ids: Iterable[str] | None = None,
        on_frame: Callable[[Any], None] | None = None,
    ) -> RunResult:
        # This must precede any ML/simulator import, object creation or output write.
        grant = authorize_runtime(self.config.runtime)
        instruction = validate_instruction(instruction)
        protected = tuple(self.config.scene["protected_ids"] if protected_ids is None else protected_ids)
        validate_plan({"actions": [{"action": "done"}]}, protected_ids=protected)
        from .qwen import QwenPlanner
        from .simulation import FrankaSimulation

        output_base = (self.project_root / self.config.paths["output_dir"]).resolve()
        if not output_base.is_relative_to(self.project_root):
            raise ValueError("Output directory must stay inside this project.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = output_base / stamp
        output_dir.mkdir(parents=True, exist_ok=False)
        video_path = output_dir / "simulation.mp4"
        planner = None
        simulation = None
        plans: list[TaskPlan] = []
        actions: list[dict[str, Any]] = []
        status = "round_limit_reached"
        constraints_satisfied = True
        failure: BaseException | None = None
        cleanup_errors: list[dict[str, str]] = []
        result = None
        try:
            planner = QwenPlanner(self.config.model, grant)
            simulation = FrankaSimulation(
                self.config.simulation, grant, video_path=video_path, on_frame=on_frame,
                max_runtime_seconds=self.config.runtime["max_runtime_seconds"],
            )
            simulation.build()
            initial_positions = simulation.object_positions()
            remaining = self.config.runtime["max_actions"] - 1
            for _ in range(self.config.runtime["max_model_rounds"]):
                plan = planner.plan(
                    instruction, simulation.frame(), protected_ids=protected,
                    max_actions=min(remaining + 1, self.config.runtime["max_actions"]),
                )
                plans.append(plan)
                if len(plan.actions) == 1:
                    status = "model_reported_done_unverified"
                    break
                for action in plan.actions:
                    if action.action == "done":
                        break
                    actions.append(simulation.pick_place(action.object_id, action.target_id))
                    remaining -= 1
                    positions = simulation.object_positions()
                    if any(sum((a - b) ** 2 for a, b in zip(initial_positions[key], positions[key])) > 0.01 ** 2 for key in protected):
                        constraints_satisfied = False
                        raise RuntimeError("A protected object moved over 1cm; stopping the run.")
                if remaining == 0:
                    status = "action_limit_reached"
                    break
            result = RunResult(plans, actions, simulation.frame(), str(video_path), status, constraints_satisfied)
        except BaseException as exc:
            status = "failed"
            failure = exc
        finally:
            # Attempt each cleanup even if the other raises; preserve the
            # original execution failure and record secondary cleanup errors.
            for label, resource in (("simulation", simulation), ("planner", planner)):
                if resource is None:
                    continue
                try:
                    resource.close()
                except BaseException as exc:
                    cleanup_errors.append({"resource": label, "type": type(exc).__name__, "message": str(exc)})
                    if failure is None:
                        failure = exc
                    status = "failed"
            audit = {
                "status": status, "instruction": instruction, "protected_ids": list(protected),
                "plans": [plan.to_dict() for plan in plans], "actions": actions,
                "constraints_satisfied": constraints_satisfied,
                "whole_task_success": None,
                "execution_failed": failure is not None,
                "error": None if failure is None else {"type": type(failure).__name__, "message": str(failure)},
                "cleanup_errors": cleanup_errors,
                "evidence_limit": "Model done is not task success; no expected-goal scoring contract has been supplied.",
                "controller_uses_simulator_coordinates": True,
            }
            try:
                temporary_log = output_dir / "run.json.tmp"
                temporary_log.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary_log.replace(output_dir / "run.json")
            except BaseException as exc:
                if failure is None:
                    failure = exc
                elif hasattr(failure, "add_note"):
                    failure.add_note(f"Additionally could not write run.json: {type(exc).__name__}: {exc}")
        if failure is not None:
            raise failure.with_traceback(failure.__traceback__)
        return result
