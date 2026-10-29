"""Small, strict robot action vocabulary. No ML or simulation imports."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class PlanValidationError(ValueError):
    """The model output is not an admissible robot plan."""


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    label: str
    color: tuple[float, float, float, float]
    position: tuple[float, float, float]


@dataclass(frozen=True)
class Target:
    target_id: str
    label: str
    position: tuple[float, float, float]
    size: tuple[float, float, float] = (0.18, 0.18, 0.01)


DEFAULT_OBJECTS = (
    SceneObject("red_block", "red block", (0.9, 0.12, 0.12, 1.0), (0.42, -0.10, 0.02)),
    SceneObject("blue_block", "blue block", (0.12, 0.28, 0.9, 1.0), (0.42, 0.10, 0.02)),
    SceneObject("green_block", "green block", (0.12, 0.7, 0.25, 1.0), (0.56, -0.10, 0.02)),
    SceneObject("yellow_block", "yellow block", (0.95, 0.8, 0.10, 1.0), (0.56, 0.10, 0.02)),
)
DEFAULT_TARGETS = (
    Target("left_tray", "left tray (+world Y, fixed robot coordinates)", (0.50, 0.30, 0.005)),
    Target("right_tray", "right tray (-world Y, fixed robot coordinates)", (0.50, -0.30, 0.005)),
)


@dataclass(frozen=True)
class Action:
    action: str
    object_id: str | None = None
    target_id: str | None = None
    source_target_id: str | None = None
    support_id: str | None = None
    base_id: str | None = None
    top_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        fields = ACTION_FIELDS.get(self.action)
        if fields is None:
            raise PlanValidationError("Unknown action kind.")
        return {"action": self.action, **{field: getattr(self, field) for field in fields}}


@dataclass(frozen=True)
class TaskPlan:
    actions: tuple[Action, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [action.to_dict() for action in self.actions]}


ACTION_FIELDS = {
    "pick_place": ("object_id", "target_id"),
    "push_out": ("object_id", "source_target_id"),
    "stack": ("object_id", "support_id"),
    "move_stack": ("base_id", "top_id", "target_id"),
    "done": (),
}
OBJECT_ARGUMENTS = frozenset({"object_id", "support_id", "base_id", "top_id"})
TARGET_ARGUMENTS = frozenset({"target_id", "source_target_id"})


def validate_instruction(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise PlanValidationError("Instruction must be a nonempty string.")
    if len(instruction) > 4000:
        raise PlanValidationError("Instruction exceeds 4000 characters.")
    if "\x00" in instruction:
        raise PlanValidationError("Instruction contains a NUL character.")
    return instruction.strip()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_plan(
    payload: Mapping[str, Any],
    *,
    object_ids: Iterable[str] = tuple(obj.object_id for obj in DEFAULT_OBJECTS),
    target_ids: Iterable[str] = tuple(target.target_id for target in DEFAULT_TARGETS),
    protected_ids: Iterable[str] = (),
    max_actions: int = 6,
    allow_repeated_objects: bool = False,
) -> TaskPlan:
    """Validate exact schemas; a final done is a model signal, not proof of success."""
    known_objects, known_targets, protected = set(object_ids), set(target_ids), set(protected_ids)
    if not protected <= known_objects:
        raise PlanValidationError("Protected object IDs must belong to the scene.")
    if type(max_actions) is not int or not 1 <= max_actions <= 12:
        raise PlanValidationError("max_actions must be an integer from 1 to 12.")
    if type(allow_repeated_objects) is not bool:
        raise PlanValidationError("allow_repeated_objects must be a boolean.")
    if not isinstance(payload, Mapping) or set(payload) != {"actions"}:
        raise PlanValidationError("Plan must contain exactly one key: actions.")
    raw_actions = payload["actions"]
    if not isinstance(raw_actions, list) or not 1 <= len(raw_actions) <= max_actions:
        raise PlanValidationError("Invalid action count (including final done).")
    actions: list[Action] = []
    moved: set[str] = set()
    for index, entry in enumerate(raw_actions):
        if not isinstance(entry, dict) or not isinstance(entry.get("action"), str):
            raise PlanValidationError("Each action must be an object with a string action.")
        kind = entry["action"]
        if kind == "done":
            if set(entry) != {"action"} or index != len(raw_actions) - 1:
                raise PlanValidationError("done must be last and accept no arguments.")
            actions.append(Action("done"))
            continue
        allowed_kinds = set(ACTION_FIELDS) if allow_repeated_objects else {"pick_place", "done"}
        if kind not in allowed_kinds or set(entry) != {"action", *ACTION_FIELDS[kind]}:
            raise PlanValidationError("Action kind or exact argument schema is not enabled for this task mode.")
        objects = []
        for field in ACTION_FIELDS[kind]:
            value = entry[field]
            known = known_objects if field in OBJECT_ARGUMENTS else known_targets
            if not isinstance(value, str) or value not in known:
                raise PlanValidationError(f"Unknown {field}.")
            if field in OBJECT_ARGUMENTS:
                if value in protected:
                    raise PlanValidationError(f"Protected object may not be manipulated or used as support: {value}")
                objects.append(value)
        if len(objects) != len(set(objects)):
            raise PlanValidationError("An object cannot be its own support or both members of a stack.")
        if not allow_repeated_objects and any(object_id in moved for object_id in objects):
            raise PlanValidationError("Each object may move at most once per plan.")
        moved.update(objects)
        actions.append(Action(**entry))
    if actions[-1].action != "done":
        raise PlanValidationError("Plan must terminate with done.")
    return TaskPlan(tuple(actions))


def parse_plan(text: str, **validation_options: Any) -> TaskPlan:
    """Parse JSON only; never eval, execute code, or follow model-provided paths."""
    if not isinstance(text, str) or len(text) > 16384:
        raise PlanValidationError("Model output must be a JSON string of at most 16384 characters.")
    try:
        payload = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PlanValidationError("Model output is not strict JSON.") from exc
    return validate_plan(payload, **validation_options)
