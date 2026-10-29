"""Independent, CPU-only task scoring from a predeclared goal contract.

This module never accepts a model plan as the expected answer. Protection is
checked against every supplied telemetry sample; it cannot certify unobserved
motion between samples. Use simulation time, not model-inference wall time.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """The independently specified scoring contract is invalid."""


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number, not a string or boolean")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{field} must be finite" + (" and positive" if positive else ""))
    return value


def _vector(value: Any, field: str, *, positive: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three coordinates")
    return tuple(_number(x, field, positive=positive) for x in value)


@dataclass(frozen=True)
class TaskContract:
    """Validated contract whose goal placements come from the experimenter."""

    data: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskContract":
        try:
            if not isinstance(value, Mapping):
                raise ValueError("Contract must be an object")
            allowed = {"schema_version", "task_id", "language", "instruction", "expected_placements", "protected_ids", "ignored_ids", "targets", "success_criteria"}
            if set(value) != allowed or value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
                raise ValueError("Use the complete version 1 task-contract schema")
            data = copy.deepcopy(dict(value))
            for field in ("task_id", "language", "instruction"):
                if not isinstance(data[field], str) or not data[field].strip():
                    raise ValueError(f"{field} must be a nonempty string")
            goals, targets = data["expected_placements"], data["targets"]
            if not isinstance(goals, dict) or not goals or not isinstance(targets, dict) or not targets:
                raise ValueError("At least one expected placement and target are required")
            for field in ("protected_ids", "ignored_ids"):
                items = data[field]
                if not isinstance(items, list) or any(not isinstance(x, str) or not x for x in items) or len(items) != len(set(items)):
                    raise ValueError(f"{field} must be a list of unique nonempty IDs")
            groups = [set(goals), set(data["protected_ids"]), set(data["ignored_ids"])]
            if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("Goal, protected and ignored object sets must be disjoint")
            for object_id, target_id in goals.items():
                if not isinstance(object_id, str) or not object_id or not isinstance(target_id, str) or target_id not in targets:
                    raise ValueError("Each expected object must reference an existing target")
            for target_id, target in targets.items():
                if not isinstance(target_id, str) or not target_id or not isinstance(target, dict) or set(target) != {"label", "center", "center_tolerance"}:
                    raise ValueError("Target requires label, center and center_tolerance")
                if not isinstance(target["label"], str) or not target["label"]:
                    raise ValueError("Target label must be a nonempty string")
                target["center"] = list(_vector(target["center"], "target.center"))
                target["center_tolerance"] = list(_vector(target["center_tolerance"], "target.center_tolerance", positive=True))
            criteria = data["success_criteria"]
            required = {"stable_duration_seconds", "min_stable_samples", "max_stable_position_spread_m", "protected_max_displacement_m", "max_sample_gap_seconds"}
            if not isinstance(criteria, dict) or set(criteria) != required:
                raise ValueError("Incomplete success criteria")
            minimum = criteria["min_stable_samples"]
            if type(minimum) is not int or not 3 <= minimum <= 10000:
                raise ValueError("min_stable_samples must be an integer in [3,10000]")
            for key in required - {"min_stable_samples"}:
                criteria[key] = _number(criteria[key], key, positive=True)
            if criteria["max_sample_gap_seconds"] > criteria["stable_duration_seconds"]:
                raise ValueError("Sampling gaps cannot exceed the stability duration")
        except (ValueError, TypeError, KeyError) as exc:
            raise ContractError(str(exc)) from exc
        return cls(data)

    @classmethod
    def load(cls, path: str | Path) -> "TaskContract":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def load_contract(path: str | Path) -> TaskContract:
    return TaskContract.load(path)


def evaluate_task(
    contract: TaskContract | Mapping[str, Any],
    initial_positions: Mapping[str, Any],
    position_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return strict JSON-compatible evidence, including a boolean success.

    Each sample is ``{"elapsed_s": simulation_time, "positions": {id:[x,y,z]}}``.
    Include the initial frame at elapsed_s=0 and a terminal frame. Times must be
    strictly increasing; replace the last sample rather than duplicating its
    timestamp. Missing/invalid telemetry fails closed. Ignored objects need not
    be present, but any supplied coordinates must still be finite.
    """
    task = TaskContract.from_mapping(contract.to_dict() if isinstance(contract, TaskContract) else contract).data
    criteria = task["success_criteria"]
    goals = task["expected_placements"]
    required_ids = set(goals) | set(task["protected_ids"])
    result: dict[str, Any] = {
        "task_id": task["task_id"], "success": False, "whole_task_success": False,
        "evaluation_complete": False, "scoring_source": "predeclared_contract_not_model_output",
        "protection_scope": "all_recorded_samples_relative_to_initial_positions",
        "contract": task, "violations": [], "goal_results": {}, "protected_results": {},
        "sample_count": 0, "stable_window": None,
    }

    def violation(code: str, **detail: Any) -> None:
        result["violations"].append({"code": code, **detail})

    def read_positions(value: Any, where: str) -> dict[str, tuple[float, float, float]]:
        if not isinstance(value, Mapping) or not required_ids <= set(value):
            raise ValueError(f"{where}: all goal and protected objects must be present")
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError(f"{where}: object IDs must be nonempty strings")
        return {key: _vector(position, f"{where}.{key}") for key, position in value.items()}

    try:
        initial = read_positions(initial_positions, "initial_positions")
        if not isinstance(position_samples, (list, tuple)) or not 1 <= len(position_samples) <= 100000:
            raise ValueError("Provide 1..100000 ordered telemetry samples")
        samples = []
        for index, sample in enumerate(position_samples):
            if not isinstance(sample, Mapping) or "elapsed_s" not in sample or "positions" not in sample:
                raise ValueError(f"sample[{index}] requires elapsed_s and positions")
            elapsed = _number(sample["elapsed_s"], f"sample[{index}].elapsed_s")
            if elapsed < 0 or (samples and elapsed <= samples[-1][0]):
                raise ValueError("Sample times must be nonnegative and strictly increasing")
            samples.append((elapsed, read_positions(sample["positions"], f"sample[{index}].positions")))
    except (ValueError, TypeError, OverflowError) as exc:
        violation("invalid_telemetry", detail=str(exc))
        return result

    result["sample_count"] = len(samples)
    if abs(samples[0][0]) > 1e-9:
        violation("missing_initial_sample", first_elapsed_s=samples[0][0])
    gaps = [later[0] - earlier[0] for earlier, later in zip(samples, samples[1:])]
    if gaps and max(gaps) > criteria["max_sample_gap_seconds"] + 1e-9:
        violation("telemetry_gap", largest_gap_seconds=max(gaps))
    for object_id in task["protected_ids"]:
        maximum = max(math.dist(initial[object_id], positions[object_id]) for _, positions in samples)
        if not math.isfinite(maximum):
            violation("nonfinite_derived_metric", object_id=object_id)
            return result
        passed = maximum <= criteria["protected_max_displacement_m"] + 1e-9
        result["protected_results"][object_id] = {"max_displacement_m": maximum, "limit_m": criteria["protected_max_displacement_m"], "passed": passed}
        if not passed:
            violation("protected_object_moved", object_id=object_id, max_displacement_m=maximum)

    end = samples[-1][0]
    cutoff = end - criteria["stable_duration_seconds"]
    preceding = [index for index, (elapsed, _) in enumerate(samples) if elapsed <= cutoff + 1e-9]
    stable = samples[preceding[-1]:] if preceding else []
    window_complete = bool(stable) and len(stable) >= criteria["min_stable_samples"]
    result["stable_window"] = {"start_elapsed_s": stable[0][0] if stable else None, "end_elapsed_s": end, "sample_count": len(stable), "complete": window_complete}
    if not window_complete:
        violation("insufficient_stability_evidence")

    for object_id, target_id in goals.items():
        target = task["targets"][target_id]

        def inside(position: tuple[float, float, float]) -> bool:
            return all(abs(value - center) <= tolerance + 1e-9 for value, center, tolerance in zip(position, target["center"], target["center_tolerance"]))

        final = samples[-1][1][object_id]
        final_inside = inside(final)
        points = [positions[object_id] for _, positions in stable]
        # The bounding-box diagonal conservatively bounds any pairwise motion,
        # without a quadratic loop over densely sampled trajectories.
        spread = math.hypot(*(max(p[axis] for p in points) - min(p[axis] for p in points) for axis in range(3))) if points else None
        if spread is not None and not math.isfinite(spread):
            violation("nonfinite_derived_metric", object_id=object_id)
            return result
        stable_inside = window_complete and all(inside(p) for p in points)
        stable_position = window_complete and spread <= criteria["max_stable_position_spread_m"] + 1e-9
        passed = final_inside and stable_inside and stable_position
        result["goal_results"][object_id] = {
            "target_id": target_id, "final_position": list(final), "final_inside_target": final_inside,
            "stable_inside_target": stable_inside, "stable_position_spread_upper_bound_m": spread,
            "stable_position": stable_position, "passed": passed,
        }
        if not final_inside:
            violation("goal_not_reached", object_id=object_id, target_id=target_id)
        elif window_complete and not stable_inside:
            violation("goal_not_held_for_stability_window", object_id=object_id)
        if window_complete and not stable_position:
            violation("object_not_stable", object_id=object_id)

    result["evaluation_complete"] = True
    result["success"] = result["whole_task_success"] = not result["violations"]
    return result
