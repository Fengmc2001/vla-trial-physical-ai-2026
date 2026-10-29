#!/usr/bin/env python3
"""Assemble a Japanese course video from one complete, genuine full run.

No simulator, model, network, GPU, upload, or substitute scene is invoked here.
Requires Pillow and local FFmpeg/ffprobe. The entire source recording is kept,
at 1x simulation time or a clearly labelled 2x. Version 2 multistep tasks aim
for roughly 65–75 seconds, keeping all motion even if that requires more time.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH, HEIGHT, FPS = 1280, 720, 30
BG = "#102335"
INK = "#F4F6F8"
MUTED = "#BECBD5"
ACCENT = "#66D8D0"
IMAGE_BOX = (32, 112, 736, 552)
STAGE_DURATIONS = {"task": 15.0, "model": 15.0, "result": 15.0, "limits": 10.0}
MULTISTEP_DURATIONS = {"task": 6.0, "model": 5.0, "result": 5.0, "limits": 4.0}
MULTISTEP_ROW_Y = (236, 304, 372, 440, 508)


class EvidenceError(ValueError):
    pass


def load_json(path: Path, maximum: int = 8 * 1024 * 1024):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise EvidenceError(f"Missing, linked, or oversized evidence: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(command: list[str], *, timeout: int = 1800):
    completed = subprocess.run(command, check=False, capture_output=True,
                               text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if completed.returncode:
        raise RuntimeError(f"{Path(command[0]).name} failed:\n{completed.stderr[-5000:]}")
    return completed


def probe(path: Path, ffprobe: str) -> dict:
    result = run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
                  "-of", "json", str(path)])
    value = json.loads(result.stdout)
    streams = [stream for stream in value["streams"] if stream.get("codec_type") == "video"]
    if len(streams) != 1:
        raise EvidenceError("Exactly one video stream is required.")
    stream = streams[0]
    rate = float(Fraction(stream["avg_frame_rate"]))
    count = int(stream["nb_read_frames"])
    if not 0 < rate <= 60 or count <= 0:
        raise EvidenceError("Video frame rate/count is invalid.")
    return {"duration_s": count / rate, "fps": rate, "frames": count,
            "width": int(stream["width"]), "height": int(stream["height"]),
            "codec": stream.get("codec_name"), "pixel_format": stream.get("pix_fmt")}


def load_evidence(directory: Path, ffprobe: str) -> dict:
    required = ["initial.png", "initial.json", "final.png", "final.json", "simulation.mp4",
                "status.json", "events.jsonl", "evaluation.json", "trajectory.json",
                "executed-plan.json", "inference/plan.json", "inference/model-output.json"]
    for name in required:
        path = directory / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise EvidenceError(f"Complete full-run evidence is required: {name}")
    status = load_json(directory / "status.json")
    if status.get("stage") != "full" or status.get("video_finalized") is not True:
        raise EvidenceError("This must be a finalized full run, not a camera/controller preview.")
    if status.get("phase") not in {"completed", "failed"} or status.get("model_used") is False:
        raise EvidenceError("Run has no complete AI-execution evidence.")
    if status.get("phase") == "completed" and status.get("model_used") is not True:
        raise EvidenceError("Completed run must explicitly record actual model use.")
    evaluation = load_json(directory / "evaluation.json")
    contract = evaluation.get("contract", {})
    multistep = contract.get("schema_version") == 2
    if evaluation.get("evaluation_scope") not in ({"standard_task", "ordered_multistep_task"} if multistep else {"standard_task"}):
        raise EvidenceError("This assembler requires the independently scored standard task.")
    expected_source = "predeclared_contract_and_executed_trajectory_not_model_plan" if multistep else "predeclared_contract_not_model_output"
    if evaluation.get("evaluation_complete") is not True or evaluation.get("scoring_source") != expected_source:
        raise EvidenceError("Independent scoring is incomplete or has the wrong source.")
    placement_key = "initial_placements" if multistep else "expected_placements"
    if (contract.get(placement_key) != {"blue_block": "left_tray", "yellow_block": "right_tray"}
            or contract.get("protected_ids") != ["red_block"]):
        raise EvidenceError("This video layout is specific to blue→A, yellow→B, with red protected.")
    milestones = None
    if multistep:
        required += ["task-contract.json", "action-milestones.json"]
        saved_contract = load_json(directory / "task-contract.json")
        if saved_contract != contract:
            raise EvidenceError("Run task contract differs from the independently scored contract.")
        if (contract.get("base_id"), contract.get("top_id"), contract.get("push_source_target_id"), contract.get("final_target_id")) != ("blue_block", "yellow_block", "left_tray", "right_tray"):
            raise EvidenceError("Multistep layout requires blue base, yellow top, push outside A, final stack in B.")
        milestones = load_json(directory / "action-milestones.json")
    initial_meta = load_json(directory / "initial.json")
    final_meta = load_json(directory / "final.json")
    trajectory = load_json(directory / "trajectory.json", maximum=32 * 1024 * 1024)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vla_trial.evaluation import evaluate_task
    from vla_trial.schema import parse_plan, validate_plan
    if multistep:
        from vla_trial.evaluation_multistep import evaluate_multistep_task
        recomputed = evaluate_multistep_task(contract, initial_meta["object_positions"], trajectory, milestones)
    else:
        recomputed = evaluate_task(contract, initial_meta["object_positions"], trajectory)
    for key, value in recomputed.items():
        if key == "evaluation_scope" and multistep:
            continue  # execute.py may name the exact selected contract standard_task.
        if evaluation.get(key) != value:
            raise EvidenceError(f"Saved independent score disagrees with telemetry: {key}")
    instruction = status.get("instruction")
    if instruction != evaluation["contract"]["instruction"] or not isinstance(instruction, str):
        raise EvidenceError("Instruction and independent task contract differ.")
    if len(instruction) > 400 or re.search(r"https?://|\b[\w.+-]+@[\w.-]+\.[A-Za-z]+|/(?:nas|Users|home)/", instruction):
        raise EvidenceError("Task instruction needs public-content review before rendering.")
    model = load_json(directory / "inference/model-output.json")
    if model.get("instruction") != instruction or model.get("image_sha256") != digest(directory / "initial.png"):
        raise EvidenceError("Inference input does not match this run's actual initial image/instruction.")
    plan = load_json(directory / "inference/plan.json")
    plan_options = {"object_ids": ["red_block", "blue_block", "yellow_block"],
                    "protected_ids": ["red_block"], "max_actions": 12 if multistep else 4,
                    "allow_repeated_objects": multistep}
    checked = validate_plan(plan, **plan_options).to_dict()
    raw_checked = parse_plan(model["raw_output"], **plan_options).to_dict()
    if checked != plan or raw_checked != plan:
        raise EvidenceError("Saved model JSON and parsed action plan disagree.")
    executed = load_json(directory / "executed-plan.json")
    if executed.get("source") != "QWEN_REAL_INFERENCE" or executed.get("plan") != plan:
        raise EvidenceError("Executed plan was not the recorded model plan.")
    if multistep and (len(milestones) != 5 or [m["action"] for m in milestones] != [a for a in plan["actions"] if a["action"] != "done"]):
        raise EvidenceError("Actual five executed milestones differ from the recorded model actions.")
    events_path = directory / "events.jsonl"
    if events_path.stat().st_size > 8 * 1024 * 1024:
        raise EvidenceError("Event log is unexpectedly large.")
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    phases = [event.get("phase") for event in events]
    expected_phases = ["building_scene", "scene_ready", "waiting_for_qwen", "moving", status["phase"]]
    cursor = 0
    for phase in phases:
        if cursor < len(expected_phases) and phase == expected_phases[cursor]:
            cursor += 1
    if cursor != len(expected_phases):
        raise EvidenceError("Ordered real scene, inference, action and terminal events are required.")
    elapsed = [event.get("wall_elapsed_s") for event in events]
    if any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in elapsed):
        raise EvidenceError("Event wall times are invalid.")
    if any(a > b for a, b in zip(elapsed, elapsed[1:])):
        raise EvidenceError("Event sequence is not chronological.")
    for name in ("initial.png", "final.png"):
        with Image.open(directory / name) as image:
            image.verify()
    raw = probe(directory / "simulation.mp4", ffprobe)
    sim_end = float(final_meta["simulation_seconds"])
    sim_start = float(initial_meta["simulation_seconds"])
    if not 0 < sim_start < sim_end or abs(raw["duration_s"] - sim_end) > 1 / raw["fps"] + 0.02:
        raise EvidenceError("Recorded video duration does not match final simulation time.")
    if abs(trajectory[-1]["elapsed_s"] - (sim_end - sim_start)) > 1 / raw["fps"] + 0.02:
        raise EvidenceError("Trajectory and camera final times disagree.")
    if trajectory[-1]["positions"] != final_meta["object_positions"]:
        raise EvidenceError("Final camera metadata does not match the scored final positions.")
    finger_extensions = False
    runtime_path = directory / "runtime-environment.json"
    if runtime_path.exists():
        runtime_environment = load_json(runtime_path)
        simulation_settings = runtime_environment.get("simulation_settings", {})
        if not isinstance(simulation_settings, dict):
            raise EvidenceError("Recorded simulation settings are invalid.")
        finger_extensions = simulation_settings.get("finger_extensions", False)
        if type(finger_extensions) is not bool:
            raise EvidenceError("Recorded finger-extension flag must be a boolean.")
        required.append("runtime-environment.json")
    successful = status["phase"] == "completed" and evaluation.get("whole_task_success") is True
    return {"status": status, "evaluation": evaluation, "plan": plan, "model": model,
            "instruction": instruction, "events": events, "raw": raw, "successful": successful,
            "multistep": multistep, "milestones": milestones,
            "finger_extensions": finger_extensions,
            "simulation_origin_s": sim_start, "final_positions": final_meta["object_positions"],
            "hashes": {name: digest(directory / name) for name in required}}


def choose_timing(evidence: dict) -> tuple[int, dict]:
    raw_seconds = evidence["raw"]["duration_s"]
    if evidence["multistep"]:
        speed = 1 if raw_seconds + sum(MULTISTEP_DURATIONS.values()) <= 70 else 2
        motion = raw_seconds / speed
        context = min(20.0, max(10.0, 70 - motion))
        durations = {name: seconds * context / 20 for name, seconds in MULTISTEP_DURATIONS.items()}
        extra = max(0, 65 - motion - context)
        for name, fraction in (("task", 0.4), ("model", 0.3), ("result", 0.3)):
            durations[name] += extra * fraction
    else:
        durations = dict(STAGE_DURATIONS)
        speed = 1 if raw_seconds + sum(durations.values()) <= 180 else 2
        durations["result"] += max(0, 60 - raw_seconds / speed - sum(durations.values()))
    if raw_seconds / speed + sum(durations.values()) > 180:
        raise EvidenceError("The complete recording cannot fit 3 minutes even at 2x; no actions will be cut.")
    if evidence["raw"]["fps"] * speed > FPS:
        raise EvidenceError("Source rate is too high to preserve every frame at the chosen playback speed.")
    return speed, durations


def action_label(action: dict) -> str:
    """Japanese display text derived only from a validated actual model action."""
    objects = {"blue_block": "青", "yellow_block": "黄"}
    targets = {"left_tray": "A", "right_tray": "B"}
    kind = action["action"]
    if kind == "pick_place":
        return f"{objects[action['object_id']]}を {targets[action['target_id']]} へ"
    if kind == "push_out":
        return f"{objects[action['object_id']]}を {targets[action['source_target_id']]} の外へ押す"
    if kind == "stack":
        return f"{objects[action['object_id']]}を{objects[action['support_id']]}の上に積む"
    if kind == "move_stack":
        return f"{objects[action['base_id']]}＋{objects[action['top_id']]}を積んだまま {targets[action['target_id']]} へ"
    raise EvidenceError("Only actual manipulation actions may appear in the action table.")


def find_font() -> Path:
    for path in Path("/System/Library/Fonts").glob("*.ttc"):
        name = unicodedata.normalize("NFC", path.name)
        if name == "ヒラギノ角ゴシック W3.ttc":
            return path
    for candidate in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                      "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError("A Japanese font is required. Pass --font with a local Japanese TTF/TTC.")


class Cards:
    def __init__(self, font: Path, account: str):
        self.font_path = font
        self.account = account
        self.prompt_font_name = font.name

    def font(self, size: int):
        return ImageFont.truetype(str(self.font_path), size=size)

    def font_for_exact_text(self, text: str, size: int):
        """Use a font covering every visible character of the original prompt."""
        candidates = (self.font_path, Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
                      Path("/System/Library/Fonts/STHeiti Light.ttc"),
                      Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
        for path in candidates:
            if not path.is_file():
                continue
            font = ImageFont.truetype(str(path), size=size)
            replacement = font.getmask(chr(0x10FFFF))
            missing_signature = (replacement.size, bytes(replacement))
            supported = True
            for char in set(text):
                if char.isspace():
                    continue
                glyph = font.getmask(char)
                if (glyph.size, bytes(glyph)) == missing_signature:
                    supported = False
                    break
            if supported:
                self.prompt_font_name = path.name
                return font
        raise EvidenceError("No local font covers every original prompt character; refusing missing-glyph boxes.")

    def wrapped(self, draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int],
                width: int, size: int = 28, color: str = INK, spacing: int = 12,
                bottom: int = 670, font_override=None) -> int:
        font = font_override or self.font(size)
        lines = []
        for paragraph in text.split("\n"):
            line = ""
            for char in paragraph:
                if line and draw.textlength(line + char, font=font) > width:
                    lines.append(line)
                    line = char
                else:
                    line += char
            lines.append(line)
        y = xy[1]
        for line in lines:
            if y + size > bottom:
                raise EvidenceError("Card text does not fit; shorten or explicitly revise the layout.")
            draw.text((xy[0], y), line, font=font, fill=color)
            y += size + spacing
        return y

    def base(self, title: str):
        image = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        draw.text((32, 26), title, font=self.font(34), fill=INK)
        draw.rounded_rectangle((1000, 26, 1248, 70), radius=12, fill="#264455")
        draw.text((1016, 34), "シミュレーション", font=self.font(24), fill=ACCENT)
        draw.line((32, 88, 1248, 88), fill="#436174", width=2)
        draw.text((32, 688), f"PAI最終課題_{self.account}", font=self.font(17), fill=MUTED)
        return image, draw

    def scene(self, canvas: Image.Image, source: Path, box: tuple[int, int, int, int] = IMAGE_BOX):
        x, y, width, height = box
        with Image.open(source) as image:
            image = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))

    def create(self, directory: Path, evidence: dict, work: Path, speed: int):
        if evidence["multistep"]:
            self.create_multistep(directory, evidence, work, speed)
            return
        image, draw = self.base("01  言葉で指示する卓上整理")
        self.scene(image, directory / "initial.png")
        y = self.wrapped(draw, "実際の初期カメラ画像", (816, 120), 408, size=24, color=ACCENT)
        y = self.wrapped(draw, evidence["instruction"], (816, y + 20), 408, size=29, spacing=13)
        self.wrapped(draw, "トレー A：青緑色\nトレー B：紫色\n赤いブロック：動かさない", (816, max(y + 28, 465)), 408, size=24, spacing=14)
        image.save(work / "task.png")

        image, draw = self.base("02  画像と言葉から動作を選ぶ")
        draw.text((32, 110), "Qwen3-VL-8B-Instruct の実出力（整形のみ）", font=self.font(27), fill=ACCENT)
        code = json.dumps(evidence["plan"], ensure_ascii=False, indent=2)
        # No key renaming or manual action substitution: display the file's JSON.
        mono_path = Path("/System/Library/Fonts/Monaco.ttf")
        for size in range(25, 18, -1):
            code_font = ImageFont.truetype(str(mono_path if mono_path.exists() else self.font_path), size=size)
            lines = code.splitlines()
            if max(draw.textlength(line, font=code_font) for line in lines) <= 820 and len(lines) * (size + 5) <= 500:
                break
        else:
            raise EvidenceError("Actual model JSON is too large for the legible 15-second card.")
        draw.multiline_text((42, 160), code, font=code_font, fill=INK, spacing=5)
        self.wrapped(draw, "入力\n実カメラ画像＋指示文\n\n出力\n対象と移動先の JSON\n\n搬送の制御には\nシミュレータ座標を使用", (900, 176), 330, size=25, spacing=12)
        image.save(work / "model.png")

        image, draw = self.base("03  指示に基づく搬送")
        draw.rectangle((IMAGE_BOX[0], IMAGE_BOX[1], IMAGE_BOX[0] + IMAGE_BOX[2], IMAGE_BOX[1] + IMAGE_BOX[3]), fill="#1A1A1A")
        speed_text = "等速（シミュレーション時間）" if speed == 1 else "2倍速（シミュレーション時間）"
        y = self.wrapped(draw, speed_text, (816, 126), 408, size=27, color=ACCENT)
        y = self.wrapped(draw, "動作録画を最初から最後まで連続再生", (816, y + 20), 408, size=28)
        y = self.wrapped(draw, "青緑色トレー：A\n紫色トレー：B\n赤いブロック：保護対象", (816, y + 32), 408, size=25)
        self.wrapped(draw, "初期化・AIの待ち時間は省略。\n映像の時間と、計算にかかった実時間は異なります。", (816, y + 28), 408, size=22, color=MUTED, spacing=10)
        image.save(work / "motion.png")

        image, draw = self.base("04  位置の記録から独立に検証")
        self.scene(image, directory / "final.png")
        verdict = "独立検証：成功" if evidence["successful"] else "独立検証：未達成"
        y = self.wrapped(draw, verdict, (816, 122), 408, size=32,
                         color=ACCENT if evidence["successful"] else "#FFBB8C")
        evaluation = evidence["evaluation"]
        for object_id, label in (("blue_block", "青いブロック → A"), ("yellow_block", "黄色いブロック → B")):
            goal = evaluation["goal_results"].get(object_id)
            if not isinstance(goal, dict):
                raise EvidenceError("Missing independently checked object goal.")
            y = self.wrapped(draw, f"{label}\n{'達成' if goal.get('passed') is True else '未達成'}", (816, y + 18), 408, size=25, spacing=8)
        red = evaluation["protected_results"].get("red_block")
        if not isinstance(red, dict) or not math.isfinite(float(red["max_displacement_m"])):
            raise EvidenceError("Missing independently measured protected-object result.")
        millimeters = 1000 * red["max_displacement_m"]
        y = self.wrapped(draw, f"赤：最大移動量 {millimeters:.2f} mm\n{'保護条件を達成' if red.get('passed') is True else '保護条件は未達成'}", (816, y + 18), 408, size=24, spacing=8)
        stable = evaluation["contract"]["success_criteria"]["stable_duration_seconds"]
        self.wrapped(draw, f"終端 {stable:g} 秒の安定性も確認。\nこの動画に示す 1 試行の結果。", (816, y + 22), 408, size=21, color=MUTED, spacing=8)
        image.save(work / "result.png")

        image, draw = self.base("05  構成と今回の範囲")
        y = self.wrapped(draw, "VLM が対象を選択 → IK・PD 制御で実行", (48, 125), 1160, size=37, color=ACCENT)
        y = self.wrapped(draw, "制御にはシミュレータの物体座標を使用。\n新しい VLA の学習や、実機への転移は行っていません。", (48, y + 28), 1160, size=30, spacing=18)
        y = self.wrapped(draw, "既存の Qwen3-VL-8B-Instruct と Genesis を使用。\nロボット：Franka Panda ／ 制御参考：Genesis 公式 grasp tutorial", (48, y + 38), 1160, size=27, spacing=17)
        self.wrapped(draw, "動画は実際のカメラ記録。モデルの「done」と独立検証を区別。\n出典・実行方法・検証条件は提出コードとレポートに記載。", (48, y + 34), 1160, size=25, color=MUTED, spacing=14)
        image.save(work / "limits.png")

    def create_multistep(self, directory: Path, evidence: dict, work: Path, speed: int):
        """Compact cards for the actual five-action run, with no invented plan."""
        actions = [action for action in evidence["plan"]["actions"] if action["action"] != "done"]
        if len(actions) != 5:
            raise EvidenceError("The compact action table requires five actual manipulation actions.")
        image, draw = self.base("01  日・英・中の連続指示")
        self.scene(image, directory / "initial.png", box=(32, 134, 560, 420))
        draw.text((32, 582), "実カメラ画像　A：青緑色 ／ B：紫色", font=self.font(22), fill=MUTED)
        draw.text((640, 112), "実際に入力した指示（原文）", font=self.font(24), fill=ACCENT)
        prompt_font = self.font_for_exact_text(evidence["instruction"], 22)
        y = self.wrapped(draw, evidence["instruction"], (640, 154), 608, size=22, spacing=7,
                         bottom=470, font_override=prompt_font)
        y = max(y + 18, 456)
        draw.text((640, y), "課題の解釈（要約）", font=self.font(23), fill=ACCENT)
        self.wrapped(draw, "青→A・黄→B → 青を A の外へ押す → 黄を青の上へ → 積んだまま B へ。赤は全工程で動かさない。", (640, y + 40), 608, size=25, spacing=10)
        image.save(work / "task.png")

        image, draw = self.base("02  VLM が選んだ 5 段階の操作")
        draw.text((32, 112), "Qwen3-VL-8B-Instruct の実 JSON 出力から表示（内容変更なし）", font=self.font(25), fill=ACCENT)
        for index, action in enumerate(actions):
            y = 174 + index * 82
            draw.rounded_rectangle((32, y, 1248, y + 68), radius=9, fill="#1B354A")
            draw.text((50, y + 17), f"{index + 1:02d}", font=self.font(27), fill=ACCENT)
            draw.text((114, y + 17), action["action"], font=self.font(26), fill=INK)
            self.wrapped(draw, action_label(action), (440, y + 18), 780, size=27, spacing=4, bottom=y + 66)
        draw.text((32, 622), "末尾 done はモデルの終了宣言。成功は実行後の独立検証で判定。", font=self.font(24), fill=MUTED)
        image.save(work / "model.png")

        image, draw = self.base("03  5 段階の操作を連続再生")
        draw.rectangle((IMAGE_BOX[0], IMAGE_BOX[1], IMAGE_BOX[0] + IMAGE_BOX[2], IMAGE_BOX[1] + IMAGE_BOX[3]), fill="#1A1A1A")
        draw.text((816, 112), "2倍速" if speed == 2 else "等速", font=self.font(38), fill=ACCENT)
        draw.text((816, 162), "シミュレーション時間を基準に再生", font=self.font(22), fill=INK)
        draw.text((816, 200), "A：青緑色　B：紫色　赤：保護対象", font=self.font(19), fill=MUTED)
        for index, action in enumerate(actions):
            self.wrapped(draw, f"{index + 1}. {action_label(action)}", (816, MULTISTEP_ROW_Y[index]), 408,
                         size=24, spacing=6, bottom=MULTISTEP_ROW_Y[index] + 62)
        self.wrapped(draw, "動作は全編連続。初期化・AI待ちを省略。\n計算にかかった実時間とは異なります。", (816, 602), 408, size=18, spacing=5)
        image.save(work / "motion.png")

        image, draw = self.base("04  全工程を位置記録で検証")
        self.scene(image, directory / "final.png")
        verdict = "独立検証：成功" if evidence["successful"] else "独立検証：未達成"
        draw.text((816, 118), verdict, font=self.font(30), fill=ACCENT if evidence["successful"] else "#FFBB8C")
        milestones = evidence["evaluation"]["milestone_results"]
        checks = (("initial_sort", "初期配置"), ("push_blue_outside_a", "青を A の外へ"),
                  ("yellow_stacked_on_blue", "黄を青の上へ"),
                  ("stack_preserved_during_transport", "搬送中の積層維持"), ("intact_stack_in_b", "2個を B に配置"))
        y = 176
        for key, label in checks:
            result = milestones.get(key)
            if not isinstance(result, dict) or type(result.get("passed")) is not bool:
                raise EvidenceError(f"Missing independently checked multistep milestone: {key}")
            self.wrapped(draw, f"{label}：{'達成' if result['passed'] else '未達成'}", (816, y), 408, size=23, spacing=4)
            y += 39
        red = evidence["evaluation"]["protected_results"].get("red_block")
        if not isinstance(red, dict) or not math.isfinite(float(red["maximum_displacement_m"])):
            raise EvidenceError("Missing measured red-block displacement.")
        red_mm = red["maximum_displacement_m"] * 1000
        self.wrapped(draw, f"赤の最大移動：{red_mm:.2f} mm\n保護条件：{'達成' if red.get('passed') is True else '未達成'}", (816, 391), 408, size=23, spacing=8)
        positions = evidence["final_positions"]
        base, top = positions["blue_block"], positions["yellow_block"]
        horizontal = 1000 * math.hypot(top[0] - base[0], top[1] - base[1])
        vertical = 1000 * (top[2] - base[2])
        self.wrapped(draw, f"最終位置の中心差\n水平 {horizontal:.2f} mm ／ 上下 {vertical:.2f} mm", (816, 486), 408, size=21, spacing=9)
        self.wrapped(draw, "全記録の赤の位置と各工程を確認。\nこの動画に示す 1 試行の結果。", (816, 594), 408, size=20, color=MUTED, spacing=9)
        image.save(work / "result.png")

        image, draw = self.base("05  構成と範囲")
        draw.text((48, 154), "VLM ＋ 従来制御（新規 VLA 学習なし）", font=self.font(39), fill=ACCENT)
        self.wrapped(draw, "制御にはシミュレータ座標を使用。実機への転移は未検証。", (48, 254), 1160, size=30, spacing=12)
        robot = "Panda＋延長指（物理衝突あり）" if evidence.get("finger_extensions") is True else "Franka Panda"
        self.wrapped(draw, f"Qwen3-VL-8B-Instruct ／ Genesis ／ {robot}\n制御参考：Genesis 公式 grasp tutorial", (48, 372), 1160, size=27, spacing=20)
        draw.text((48, 545), "出典・コード・検証条件は提出レポートに記載。", font=self.font(26), fill=MUTED)
        image.save(work / "limits.png")


def encoder_args() -> list[str]:
    return ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "2",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-map_metadata", "-1", "-movflags", "+faststart"]


def assemble(directory: Path, output: Path, account: str, font: Path, ffmpeg: str, ffprobe: str):
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("Choose a new .mp4 output; existing files are never overwritten.")
    if output.is_relative_to(directory):
        raise ValueError("Keep the submission export outside the immutable input run directory.")
    evidence = load_evidence(directory, ffprobe)
    speed, durations = choose_timing(evidence)
    motion_duration = evidence["raw"]["duration_s"] / speed
    planned_seconds = motion_duration + sum(durations.values())
    if evidence["multistep"] and planned_seconds > 75:
        print(f"Complete five-action recording requires {planned_seconds:.1f}s at {speed}x; "
              "keeping all motion and disclosing the longer duration.", file=sys.stderr)
    work = output.parent / (output.stem + "_assets")
    work.mkdir(parents=True, exist_ok=False)
    cards = Cards(font, account)
    cards.create(directory, evidence, work, speed)
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-filter_threads", "2", "-filter_complex_threads", "2"]
    segments = []
    timeline = []
    position = 0.0
    for index, name in enumerate(("task", "model", "motion", "result", "limits")):
        target = work / f"{index:02d}_{name}.mp4"
        if name == "motion":
            x, y, width, height = IMAGE_BOX
            filters = (f"[0:v]setpts=(PTS-STARTPTS)/{speed},scale={width}:{height}:"
                       f"force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                       f"fps={FPS}[robot];[1:v][robot]overlay={x}:{y}:shortest=1")
            if evidence["multistep"]:
                for index, milestone in enumerate(evidence["milestones"]):
                    start = (evidence["simulation_origin_s"] + milestone["started_s"]) / speed
                    end = (evidence["simulation_origin_s"] + milestone["ended_s"]) / speed
                    filters += (f",drawbox=x=804:y={MULTISTEP_ROW_Y[index] - 8}:w=432:h=62:"
                                f"color=0x66D8D0@0.9:t=2:enable='between(t,{start:.6f},{end:.6f})'")
            filters += ",format=yuv420p[v]"
            command = base + ["-threads", "2", "-i", str(directory / "simulation.mp4"),
                              "-loop", "1", "-framerate", str(FPS), "-i", str(work / "motion.png"),
                              "-filter_complex", filters, "-map", "[v]"] + encoder_args() + [str(target)]
            duration = motion_duration
        else:
            duration = durations[name]
            command = base + ["-loop", "1", "-framerate", str(FPS), "-i", str(work / f"{name}.png"),
                              "-frames:v", str(round(duration * FPS))] + encoder_args() + [str(target)]
        run(command)
        measured = probe(target, ffprobe)
        if abs(measured["duration_s"] - duration) > 1 / FPS + 0.002:
            raise EvidenceError(f"Encoded {name} duration does not match the uncut timeline.")
        if name == "motion" and measured["frames"] < evidence["raw"]["frames"]:
            raise EvidenceError("Motion segment lost source frames; export rejected.")
        segments.append(target.name)
        timeline.append({"stage": name, "start_s": round(position, 6),
                         "end_s": round(position + measured["duration_s"], 6),
                         "duration_s": measured["duration_s"],
                         "source": "simulation.mp4" if name == "motion" else f"{name}.png",
                         "playback_speed_vs_simulation_time": speed if name == "motion" else None})
        position += measured["duration_s"]
    (work / "concat.txt").write_text("".join(f"file '{name}'\n" for name in segments))
    temporary_output = work / "assembled.mp4"
    run(base + ["-f", "concat", "-safe", "1", "-i", str(work / "concat.txt"), "-c", "copy",
                "-map_metadata", "-1", "-metadata", f"title=PAI最終課題_{account}",
                "-movflags", "+faststart", str(temporary_output)])
    final = probe(temporary_output, ffprobe)
    if (final["width"], final["height"]) != (WIDTH, HEIGHT) or not 60 <= final["duration_s"] <= 180:
        raise EvidenceError("Final video dimensions or 1–3 minute duration requirement failed.")
    if abs(final["duration_s"] - position) > 1 / FPS + 0.002:
        raise EvidenceError("Concatenated duration differs from the complete source timeline.")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2",
         "-i", str(temporary_output), "-f", "null", "-"], timeout=1800)
    manifest = {"schema_version": 1, "account": account, "title": f"PAI最終課題_{account}",
                "video": output.name, "video_sha256": digest(temporary_output), "video_info": final,
                "independent_task_success": evidence["successful"], "timeline": timeline,
                "source_video": evidence["raw"], "source_evidence_sha256": evidence["hashes"],
                "raw_recording_cut": False, "playback_speed_vs_simulation_time": speed,
                "task_schema_version": evidence["evaluation"]["contract"]["schema_version"],
                "task_id": evidence["evaluation"]["contract"]["task_id"],
                "finger_extensions": evidence["finger_extensions"],
                "original_prompt_font": cards.prompt_font_name,
                "requested_duration_approx_seconds": 65 if evidence["multistep"] else None,
                "omitted": "Initialization and model waiting wall time; no motion footage removed.",
                "limitations": "VLM action selection plus conventional IK/PD; privileged simulator coordinates; no VLA training.",
                "publication_performed": False}
    if evidence["multistep"]:
        manifest["displayed_actions"] = [{"action": milestone["action"], "display_label": action_label(milestone["action"]),
             "source_simulation_start_s": evidence["simulation_origin_s"] + milestone["started_s"],
             "source_simulation_end_s": evidence["simulation_origin_s"] + milestone["ended_s"]}
             for milestone in evidence["milestones"]]
    (work / "timeline.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary_output.rename(output)
    print(json.dumps({"video": str(output), "timeline": str(work / "timeline.json"),
                      "duration_s": final["duration_s"], "simulation_playback_speed": speed,
                      "independent_task_success": evidence["successful"]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Complete actual full-run directory, copied from the simulator")
    parser.add_argument("--output", type=Path, required=True, help="New 1280x720 MP4 outside the input directory")
    parser.add_argument("--account", required=True, help="OmniCampus account name only, not a personal name")
    parser.add_argument("--font", type=Path, help="Local Japanese TTF/TTC; defaults to Hiragino or Noto CJK")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.account):
        parser.error("Use an account name containing 1–64 letters, digits, underscores, hyphens or periods.")
    for tool in (args.ffmpeg, args.ffprobe):
        if not Path(tool).is_file():
            parser.error(f"Required local executable is missing: {Path(tool).name}")
    try:
        try:
            os.nice(10)
        except (AttributeError, PermissionError):
            # Some macOS sandboxes reject niceness changes; FFmpeg still uses
            # only two codec/filter threads and never requests GPU encoding.
            pass
        assemble(args.run_dir.resolve(strict=True), args.output.resolve(), args.account,
                 args.font.resolve(strict=True) if args.font else find_font(), args.ffmpeg, args.ffprobe)
    except (EvidenceError, ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"Submission export stopped: {error}") from error


if __name__ == "__main__":
    main()
