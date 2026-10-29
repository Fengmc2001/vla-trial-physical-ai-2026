"""Independent multistep scoring from trajectory and executed-action timestamps.

The contract, not a model plan, defines the required milestones. Position
telemetry can establish sampled geometry and rule out a lifted 'push', but does
not prove contact forces or absence of unobserved motion between samples.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evaluation import ContractError, _number, _vector


_EPS = 1e-9
_FIELDS = {"schema_version", "task_id", "task_mode", "language", "instruction", "interpretation",
           "active_object_ids", "protected_ids", "base_id", "top_id", "initial_placements",
           "push_source_target_id", "final_target_id", "targets", "success_criteria"}
_CRITERIA = {"stable_duration_seconds", "min_stable_samples", "max_stable_position_spread_m",
             "protected_max_displacement_m", "max_sample_gap_seconds", "block_half_width_m",
             "block_height_m", "outside_tray_clearance_m", "ground_center_z_m",
             "ground_center_z_tolerance_m", "stack_xy_tolerance_m", "stack_z_tolerance_m",
             "push_max_center_rise_m", "push_min_planar_displacement_m"}


def validate_multistep_contract(value: Mapping[str, Any]) -> dict:
    try:
        if not isinstance(value, Mapping) or set(value) != _FIELDS or type(value["schema_version"]) is not int or value["schema_version"] != 2:
            raise ValueError("Use the complete version 2 multistep contract.")
        task = copy.deepcopy(dict(value))
        if task["task_mode"] != "multistep":
            raise ValueError("Version 2 requires multistep task mode.")
        for field in ("task_id", "language", "instruction", "interpretation"):
            if not isinstance(task[field], str) or not task[field].strip():
                raise ValueError(f"{field} must be nonempty text")
        for field in ("active_object_ids", "protected_ids"):
            items = task[field]
            if not isinstance(items, list) or any(not isinstance(x, str) or not x for x in items) or len(items) != len(set(items)):
                raise ValueError(f"{field} must contain unique object IDs")
        active, protected = set(task["active_object_ids"]), set(task["protected_ids"])
        base, top = task["base_id"], task["top_id"]
        if not isinstance(base, str) or not isinstance(top, str) or base == top or not {base, top} <= active or not protected <= active or {base, top} & protected:
            raise ValueError("Stack members must be distinct active and unprotected objects.")
        targets = task["targets"]
        if not isinstance(targets, dict) or not targets:
            raise ValueError("Target geometry is required.")
        for target_id, target in targets.items():
            if not isinstance(target_id, str) or not target_id or not isinstance(target, dict) or set(target) != {"label", "center", "center_tolerance", "surface_half_size_xy"}:
                raise ValueError("Invalid target geometry schema.")
            target["center"] = list(_vector(target["center"], "center"))
            target["center_tolerance"] = list(_vector(target["center_tolerance"], "center_tolerance", positive=True))
            half = target["surface_half_size_xy"]
            if not isinstance(half, (list, tuple)) or len(half) != 2:
                raise ValueError("Tray footprint needs two half sizes.")
            target["surface_half_size_xy"] = [_number(x, "tray half size", positive=True) for x in half]
        placements = task["initial_placements"]
        if not isinstance(placements, dict) or set(placements) != {base, top} or any(target not in targets for target in placements.values()):
            raise ValueError("Initial placements must independently specify both stack members.")
        if task["push_source_target_id"] != placements[base] or task["final_target_id"] not in targets:
            raise ValueError("Push source and final target must agree with the declared targets.")
        criteria = task["success_criteria"]
        if not isinstance(criteria, dict) or set(criteria) != _CRITERIA:
            raise ValueError("Incomplete multistep success criteria.")
        if type(criteria["min_stable_samples"]) is not int or not 3 <= criteria["min_stable_samples"] <= 10000:
            raise ValueError("At least three stable samples are required.")
        for key in _CRITERIA - {"min_stable_samples"}:
            criteria[key] = _number(criteria[key], key, positive=True)
        if criteria["max_sample_gap_seconds"] > criteria["stable_duration_seconds"]:
            raise ValueError("Sampling gap cannot exceed stability duration.")
        return task
    except (ValueError, TypeError, KeyError) as error:
        raise ContractError(str(error)) from error


def load_multistep_contract(path: str | Path) -> dict:
    return validate_multistep_contract(json.loads(Path(path).read_text()))


def evaluate_multistep_task(
    contract: Mapping[str, Any], initial_positions: Mapping[str, Any],
    position_samples: Sequence[Mapping[str, Any]],
    action_milestones: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check ordered execution, geometry, push height and intact stack transport.

    Telemetry uses ``{"elapsed_s": time, "positions": {id:[x,y,z]}}``.
    Each actual primitive supplies ``{"action": action.to_dict(),
    "started_s": time, "ended_s": time, "completed": True}``. Include at least
    0.5 simulation seconds of settling before each ended_s and keep sampling
    during every motion. Do not create milestones from the planned actions.
    """
    task = validate_multistep_contract(contract)
    criteria = task["success_criteria"]
    base, top = task["base_id"], task["top_id"]
    required = set(task["active_object_ids"])
    gap = criteria["max_sample_gap_seconds"]
    result = {"task_id": task["task_id"], "evaluation_scope": "ordered_multistep_task",
              "success": False, "whole_task_success": False, "evaluation_complete": False,
              "scoring_source": "predeclared_contract_and_executed_trajectory_not_model_plan",
              "contract": task, "sample_count": 0, "action_count": 0,
              "protection_scope": "all_recorded_samples_relative_to_initial_positions",
              "telemetry_limit": "sampled positions do not prove contact forces or unobserved motion",
              "violations": [], "milestone_results": {}, "protected_results": {}}

    def fail(code, **details):
        result["violations"].append({"code": code, **details})

    def read_positions(value):
        if not isinstance(value, Mapping) or not required <= set(value):
            raise ValueError("Missing required object position.")
        return {key: _vector(point, f"position.{key}") for key, point in value.items()}

    try:
        initial = read_positions(initial_positions)
        if not isinstance(position_samples, (list, tuple)) or not position_samples:
            raise ValueError("No trajectory samples.")
        samples = []
        for entry in position_samples:
            if not isinstance(entry, Mapping):
                raise ValueError("Invalid sample.")
            when = _number(entry["elapsed_s"], "sample time")
            positions = read_positions(entry["positions"])
            if when < 0 or (samples and (when <= samples[-1][0] or when - samples[-1][0] > gap + _EPS)):
                raise ValueError("Times must increase strictly with bounded gaps.")
            samples.append((when, positions))
        if abs(samples[0][0]) > _EPS or any(math.dist(initial[key], samples[0][1][key]) > 1e-6 for key in required):
            raise ValueError("The first sample must match initial positions at time zero.")
    except (ValueError, TypeError, KeyError) as error:
        fail("invalid_trajectory", detail=str(error))
        return result
    result["sample_count"] = len(samples)
    times = [when for when, _ in samples]
    for object_id in task["protected_ids"]:
        peak = max(math.dist(initial[object_id], positions[object_id]) for _, positions in samples)
        if not math.isfinite(peak):
            fail("nonfinite_derived_metric", object_id=object_id)
            return result
        passed = peak <= criteria["protected_max_displacement_m"] + _EPS
        result["protected_results"][object_id] = {"maximum_displacement_m": peak, "passed": passed}
        if not passed:
            fail("protected_object_moved", object_id=object_id, maximum_displacement_m=peak)
    try:
        if isinstance(action_milestones, (list, tuple)):
            result["action_count"] = len(action_milestones)
        if not isinstance(action_milestones, (list, tuple)) or len(action_milestones) != 5:
            raise ValueError("Exactly five executed manipulation milestones are required.")
        milestones = []
        for entry in action_milestones:
            if not isinstance(entry, Mapping) or not {"action", "started_s", "ended_s", "completed"} <= set(entry):
                raise ValueError("Incomplete executed-action milestone.")
            start, end = _number(entry["started_s"], "action start"), _number(entry["ended_s"], "action end")
            if start < 0 or end <= start or end > times[-1] + _EPS or (milestones and start < milestones[-1]["ended_s"] - _EPS):
                raise ValueError("Executed action intervals are out of order or outside the trajectory.")
            if entry["completed"] is not True or not isinstance(entry["action"], Mapping):
                raise ValueError("A manipulation did not complete.")
            milestones.append({"action": dict(entry["action"]), "started_s": start, "ended_s": end})
    except (ValueError, TypeError, KeyError) as error:
        fail("invalid_execution_milestones", detail=str(error))
        return result
    result["action_count"] = len(milestones)
    expected_sort = [{"action": "pick_place", "object_id": object_id, "target_id": target_id}
                     for object_id, target_id in task["initial_placements"].items()]
    first_two = [milestone["action"] for milestone in milestones[:2]]
    expected_tail = [
        {"action": "push_out", "object_id": base, "source_target_id": task["push_source_target_id"]},
        {"action": "stack", "object_id": top, "support_id": base},
        {"action": "move_stack", "base_id": base, "top_id": top, "target_id": task["final_target_id"]},
    ]
    if not all(first_two.count(action) == 1 for action in expected_sort) or [m["action"] for m in milestones[2:]] != expected_tail:
        fail("wrong_executed_action_order_or_arguments")
        result["evaluation_complete"] = True
        return result

    def points_during(start, end):
        left = max(0, bisect_right(times, start + _EPS) - 1)
        right = bisect_right(times, end + _EPS)
        points = samples[left:right]
        if not points or start - points[0][0] > gap + _EPS or end - points[-1][0] > gap + _EPS:
            return []
        return points

    def stable_window(end, *, not_before=0.0):
        start = end - criteria["stable_duration_seconds"]
        if start < not_before - _EPS:
            return []
        points = points_during(start, end)
        return points if len(points) >= criteria["min_stable_samples"] else []

    def spread(points, object_id):
        if not points:
            return None
        vectors = [positions[object_id] for _, positions in points]
        return math.hypot(*(max(p[axis] for p in vectors) - min(p[axis] for p in vectors) for axis in range(3)))

    def stable(points, objects):
        return bool(points) and all(spread(points, obj) <= criteria["max_stable_position_spread_m"] + _EPS for obj in objects)

    def inside(position, target_id):
        target = task["targets"][target_id]
        return all(abs(position[axis] - target["center"][axis]) <= target["center_tolerance"][axis] + _EPS for axis in range(3))

    def outside_source(position):
        target = task["targets"][task["push_source_target_id"]]
        return any(abs(position[axis] - target["center"][axis]) >= target["surface_half_size_xy"][axis] + criteria["block_half_width_m"] + criteria["outside_tray_clearance_m"] - _EPS for axis in (0, 1))

    def on_ground(position):
        return abs(position[2] - criteria["ground_center_z_m"]) <= criteria["ground_center_z_tolerance_m"] + _EPS

    def stack_geometry(positions):
        return (math.hypot(positions[top][0] - positions[base][0], positions[top][1] - positions[base][1]) <= criteria["stack_xy_tolerance_m"] + _EPS
                and abs((positions[top][2] - positions[base][2]) - criteria["block_height_m"]) <= criteria["stack_z_tolerance_m"] + _EPS)

    def record(name, passed, **details):
        result["milestone_results"][name] = {"passed": bool(passed), **details}
        if not passed:
            fail("milestone_not_satisfied", milestone=name)

    for index, milestone in enumerate(milestones[:2]):
        action = milestone["action"]
        points = stable_window(milestone["ended_s"], not_before=milestone["started_s"])
        passed = stable(points, [action["object_id"]]) and all(inside(p[action["object_id"]], action["target_id"]) for _, p in points)
        record(f"initial_place_{action['object_id']}", passed, ended_s=milestone["ended_s"], stable_samples=len(points))
    initial_sort = stable_window(milestones[1]["ended_s"], not_before=milestones[1]["started_s"])
    sorted_both = stable(initial_sort, [base, top]) and all(all(inside(p[obj], target) for obj, target in task["initial_placements"].items()) for _, p in initial_sort)
    record("initial_sort", sorted_both, ended_s=milestones[1]["ended_s"])

    push = milestones[2]
    segment = points_during(push["started_s"], push["ended_s"])
    settled = stable_window(push["ended_s"], not_before=push["started_s"])
    rise = max((p[base][2] for _, p in segment), default=math.inf) - segment[0][1][base][2] if segment else None
    planar_distance = math.hypot(segment[-1][1][base][0] - segment[0][1][base][0], segment[-1][1][base][1] - segment[0][1][base][1]) if segment else None
    if segment and not all(math.isfinite(value) for value in (rise, planar_distance)):
        fail("nonfinite_derived_metric", milestone="push_blue_outside_a")
        return result
    pushed = (bool(segment) and inside(segment[0][1][base], task["push_source_target_id"])
              and rise <= criteria["push_max_center_rise_m"] + _EPS
              and planar_distance >= criteria["push_min_planar_displacement_m"] - _EPS
              and stable(settled, [base, top])
              and all(outside_source(p[base]) and on_ground(p[base]) and inside(p[top], task["initial_placements"][top]) for _, p in settled))
    record("push_blue_outside_a", pushed, ended_s=push["ended_s"], maximum_center_rise_m=rise, planar_displacement_m=planar_distance)

    stacking = milestones[3]
    settled = stable_window(stacking["ended_s"], not_before=stacking["started_s"])
    stacked = stable(settled, [base, top]) and all(stack_geometry(p) and outside_source(p[base]) and on_ground(p[base]) for _, p in settled)
    record("yellow_stacked_on_blue", stacked, ended_s=stacking["ended_s"], stable_samples=len(settled))

    transport = milestones[4]
    segment = points_during(transport["started_s"], transport["ended_s"])
    intact = bool(segment) and all(stack_geometry(p) for _, p in segment)
    record("stack_preserved_during_transport", intact, sampled_instants=len(segment),
           started_s=transport["started_s"], ended_s=transport["ended_s"])
    settled = stable_window(times[-1], not_before=transport["started_s"])
    final_target = task["final_target_id"]
    target = task["targets"][final_target]
    final_pass = stable(settled, [base, top]) and all(
        stack_geometry(p) and inside(p[base], final_target)
        and all(abs(p[top][axis] - target["center"][axis]) <= target["center_tolerance"][axis] + _EPS for axis in (0, 1))
        for _, p in settled)
    record("intact_stack_in_b", final_pass, stable_samples=len(settled), final_base_position=list(samples[-1][1][base]), final_top_position=list(samples[-1][1][top]))
    result["evaluation_complete"] = True
    result["success"] = result["whole_task_success"] = not result["violations"]
    return result
