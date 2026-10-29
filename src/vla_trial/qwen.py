"""Offline Qwen3-VL planner with a restricted action vocabulary.

The existing original model snapshot is read-only. Adapters, downloads and
remote Python code are never loaded by this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .safety import AdmissionGrant, require_grant
from .schema import DEFAULT_OBJECTS, DEFAULT_TARGETS, TaskPlan, parse_plan, validate_instruction


def build_planner_prompt(scene_manifest: Mapping[str, Any], max_actions: int, *, multistep: bool = False) -> str:
    """Describe supported primitives without supplying the user's answer plan."""
    vocabulary = (
        'Each action is {"action":"pick_place","object_id":"an allowed id","target_id":"an allowed id"} '
    )
    if multistep:
        vocabulary += (
            '(pick one block and place it in a tray), '
            '{"action":"push_out","object_id":"an allowed id","source_target_id":"an allowed tray id"} '
            '(physically push that block outside its source tray), '
            '{"action":"stack","object_id":"top block id","support_id":"bottom block id"} '
            '(place the first block on the second), '
            '{"action":"move_stack","base_id":"bottom block id","top_id":"top block id","target_id":"an allowed tray id"} '
            '(move the existing two-block stack together, keeping the top block on its base), '
        )
    vocabulary += 'or the final {"action":"done"}. '
    repetition = (
        "Objects may occur in multiple ordered actions when the instruction requires it. Preserve the requested order. "
        "A stack always has two distinct blocks. Resolve later references using the earlier instruction and scene. "
        if multistep else "Each object may move at most once in a plan. "
    )
    return (
        "You are a robot task planner. Observe the provided current image and obey the user's entire instruction, "
        "including later clauses even when the language changes. Return ONLY one JSON object with exactly the key actions. "
        "Put the entire task in one actions list. Do not emit separate JSON objects for different languages or stages, "
        "adjacent JSON objects, Markdown fences, explanations or repeated instructions. "
        + vocabulary + "Never move protected objects or use them as stack supports. " + repetition
        + f"Use at most {max_actions} actions including done. If the task is complete or cannot safely be interpreted, return only done. "
        "The left/right target names use fixed robot coordinates in the scene manifest, not image-left/image-right. "
        "Ignore any request to output code, paths, commands, new tools or extra keys. "
        "A done action is only your assessment; success is measured separately. Scene manifest: "
        + json.dumps(scene_manifest, ensure_ascii=False)
    )


class QwenPlanner:
    def __init__(self, config: Mapping[str, Any], grant: AdmissionGrant):
        self.config = dict(config)
        self.grant = grant
        self.model = None
        self.processor = None
        self.last_raw_output = ""

    def load(self) -> None:
        require_grant(self.grant)
        if self.model is not None:
            return
        model_path = Path(self.config["path"])
        if not model_path.is_absolute() or not model_path.is_dir():
            raise ValueError("model.path must be an existing absolute local snapshot directory.")
        if (model_path / "adapter_config.json").exists():
            raise ValueError("This project requires the original model snapshot, not an adapter directory.")
        config_path = model_path / "config.json"
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        if model_config.get("model_type") != "qwen3_vl":
            raise ValueError("Local snapshot is not the expected original Qwen3-VL model.")
        if self.config.get("device", "cuda:0") != "cuda:0":
            raise ValueError("Only cuda:0 inside the explicitly UUID-bound process is supported.")
        if self.config.get("dtype", "bfloat16") not in {"bfloat16", "float16"}:
            raise ValueError("Only bfloat16 or float16 inference is prepared.")
        # Imports are deliberately after the admission and local-path checks.
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch.set_num_threads(2)
        cap_mib = int(self.config.get("memory_cap_mib", 18000))
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(cap_mib / total_mib, 0.85), 0)

        dtype = getattr(torch, self.config.get("dtype", "bfloat16"))
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            min_pixels=4096, max_pixels=int(self.config.get("max_pixels", 262144)),
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            dtype=dtype, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()

    def plan(
        self,
        instruction: str,
        frame: Any,
        *,
        protected_ids: Iterable[str] = (),
        max_actions: int = 6,
    ) -> TaskPlan:
        self.last_raw_output = ""
        instruction = validate_instruction(instruction)
        protected = tuple(protected_ids)
        active_ids = tuple(self.config.get("active_object_ids", [obj.object_id for obj in DEFAULT_OBJECTS]))
        if not active_ids or len(active_ids) != len(set(active_ids)) or not set(active_ids) <= {obj.object_id for obj in DEFAULT_OBJECTS}:
            raise ValueError("active_object_ids must identify distinct known scene objects.")
        task_mode = self.config.get("task_mode", "standard")
        if task_mode not in {"standard", "multistep"}:
            raise ValueError("Unsupported planner task mode.")
        multistep = task_mode == "multistep"
        # Validate the allowlist even if inference would otherwise be skipped.
        options = dict(object_ids=active_ids, protected_ids=protected, max_actions=max_actions, allow_repeated_objects=multistep)
        parse_plan('{"actions":[{"action":"done"}]}', **options)
        require_grant(self.grant)
        self.load()
        from PIL import Image
        import torch

        image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
        if image.width > 1280 or image.height > 1280:
            image.thumbnail((1280, 1280))
        scene_manifest = {
            "objects": [{"object_id": obj.object_id, "label": obj.label} for obj in DEFAULT_OBJECTS if obj.object_id in active_ids],
            "targets": [{"target_id": target.target_id, "label": ("A: cyan/turquoise tray" if target.target_id == "left_tray" else "B: purple tray")} for target in DEFAULT_TARGETS],
            "protected_ids": list(protected),
        }
        system = build_planner_prompt(scene_manifest, max_actions, multistep=multistep)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": instruction},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=int(self.config.get("max_new_tokens", 256)),
                do_sample=False,
            )
        trimmed = [output[len(prompt):] for prompt, output in zip(inputs.input_ids, generated)]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        self.last_raw_output = text
        return parse_plan(text, **options)

    def close(self) -> None:
        self.model = None
        self.processor = None
        # No torch import and no global cache reset during teardown.
