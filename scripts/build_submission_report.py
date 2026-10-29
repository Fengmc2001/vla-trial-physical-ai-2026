#!/usr/bin/env python3
"""Build the Japanese course PDF from one completed, genuine full-run record.

Run with a Python environment containing reportlab, Pillow and pypdf.
Optional trial summary schema (task_id is required for a version 2 task):
{"task_id":"multistep_stack_mixed_v1",
 "formal":{"total":1,"success":1,"failure":0,"unscored":0},
 "development":{"failed_runs":2,"notes_ja":["調整内容の説明"]}}
No model, simulator, GPU or external service is started by this script.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape


class EvidenceError(ValueError):
    pass


PRIVATE_TEXT = r"(?:/(?:Users|home|nas)/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})"


def read_json(path: Path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 24_000_000:
        raise EvidenceError(f"Missing, unsafe or oversized evidence: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(EvidenceError("Nonfinite JSON value")))


def require(condition: bool, message: str):
    if not condition:
        raise EvidenceError(message)


def seconds(start: str, end: str) -> float:
    a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
    require(a.tzinfo is not None and b.tzinfo is not None, "Timestamps must have time zones")
    result = (b - a).total_seconds()
    require(math.isfinite(result) and result >= 0, "Invalid timestamp order")
    return result


def safe_text(value: str) -> str:
    require(isinstance(value, str) and len(value) <= 4000, "Invalid public text")
    require(not re.search(PRIVATE_TEXT, value), "Public text contains a private path or email address")
    return value


def events(path: Path) -> list[dict]:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 4_000_000, "Missing or oversized events.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(bool(records), "Event log is empty")
    previous = -1.0
    for record in records:
        value = float(record["wall_elapsed_s"])
        require(math.isfinite(value) and value >= previous, "Event clock is not monotonic")
        datetime.fromisoformat(record["at"])
        previous = value
    return records


def phase_time(records: list[dict], name: str) -> str:
    matches = [row for row in records if row.get("phase") == name]
    require(bool(matches), f"Missing phase event: {name}")
    return matches[0]["at"]


def action_description_ja(action: dict, labels: dict[str, str]) -> str:
    """Describe the recorded primitive without assuming every action is a pick."""
    label = lambda key: labels.get(action[key], action[key])
    kind = action.get("action", "pick_place")
    if kind == "pick_place":
        return f"{label('object_id')}を{label('target_id')}へ移す"
    if kind == "push_out":
        return f"{label('object_id')}を{label('source_target_id')}の外へ押し出す"
    if kind == "stack":
        return f"{label('object_id')}を{label('support_id')}の上に積む"
    if kind == "move_stack":
        return f"{label('base_id')}を下、{label('top_id')}を上に保って、2個を一緒に{label('target_id')}へ運ぶ"
    raise EvidenceError("Unrecognized executed primitive in report")


def simulation_settings_ja(environment: dict, contract: dict) -> str:
    """Describe recorded numerical settings, never fill in controller defaults."""
    settings = environment.get("simulation_settings")
    require(isinstance(settings, dict), "Missing recorded multistep simulation settings")
    for key in ("dt_seconds", "block_size_m", "grasp_force_n_per_finger", "pair_grasp_force_n_per_finger"):
        value = settings.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                f"Invalid recorded simulation setting: {key}")
    require(type(settings.get("physics_substeps")) is int and 1 <= settings["physics_substeps"] <= 8,
            "Invalid recorded physics substeps")
    require(type(settings.get("noslip_iterations")) is int and 0 <= settings["noslip_iterations"] <= 10,
            "Invalid recorded no-slip iteration count")
    require(settings.get("friction_cone") == "pyramidal",
            "Unexpected friction-cone model; review report wording before use")
    require(math.isclose(settings["block_size_m"], contract["success_criteria"]["block_height_m"], rel_tol=0, abs_tol=1e-9),
            "Recorded cube size differs from the evaluated task contract")
    if "finger_extensions" in settings:
        require(type(settings["finger_extensions"]) is bool, "Invalid recorded finger-extension flag")
    offsets = ("grasp_offset_m", "pair_grasp_offset_m")
    has_offsets = settings.get("finger_extensions") is True or any(key in settings for key in offsets)
    if has_offsets:
        for key in offsets:
            value = settings.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    f"Invalid recorded grasp offset: {key}")
    line = (
        f"物理設定: 時間刻み {settings['dt_seconds']:g} 秒、サブステップ {settings['physics_substeps']}、"
        f"no-slip反復 {settings['noslip_iterations']}、摩擦円錐 pyramidal、立方体の一辺 {settings['block_size_m'] * 1000:g} mm。"
        f"指当たりの把持力は通常 {settings['grasp_force_n_per_finger']:g} N、2個同時 {settings['pair_grasp_force_n_per_finger']:g} N。"
    )
    if has_offsets:
        line += (f"手先の高さオフセットは通常 {settings['grasp_offset_m'] * 1000:g} mm、"
                 f"2個同時 {settings['pair_grasp_offset_m'] * 1000:g} mm。")
    return line


def recorded_finger_geometry(control: list, environment: dict) -> dict | None:
    """Read extension geometry from execution events, not the current helper."""
    if environment.get("simulation_settings", {}).get("finger_extensions") is not True:
        return None
    matches = [row for row in control if isinstance(row, dict) and row.get("stage") == "physical_finger_extensions"]
    require(len(matches) == 1, "Missing or ambiguous executed finger-extension geometry")
    record = matches[0]
    dimensions = record.get("pad_size_m")
    require(isinstance(dimensions, list) and len(dimensions) == 3
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in dimensions),
            "Invalid executed finger-pad dimensions")
    mass = record.get("added_mass_kg_per_finger")
    require(type(mass) in (int, float) and math.isfinite(mass) and mass > 0, "Invalid executed finger-pad mass")
    require(record.get("manipulated_objects_attached") is False, "Extension record must preserve independent manipulated objects")
    return {"pad_size_m": dimensions, "added_mass_kg_per_finger": mass}


@dataclass
class Evidence:
    run: Path
    status: dict
    evaluation: dict
    actions: list
    plan: dict
    model: dict
    environment: dict
    inference_environment: dict
    timing: dict
    summary: dict | None
    manifest_digest: str
    video_seconds: float | None
    milestones: list | None = None
    multistep: bool = False
    finger_geometry: dict | None = None


def load_evidence(run: Path, summary_path: Path | None = None, video: Path | None = None) -> Evidence:
    run = run.resolve(strict=True)
    status = read_json(run / "status.json")
    require(status.get("stage") == "full" and status.get("phase") == "completed", "A completed full run is required; preview/control/failed runs cannot produce this report")
    require(status.get("model_used") is True and status.get("video_finalized") is True, "Full run lacks completed AI/video evidence")
    require(not status.get("test_fixture"), "Synthetic fixtures cannot produce a submission report")
    evaluation = read_json(run / "evaluation.json")
    contract_version = evaluation.get("contract", {}).get("schema_version")
    require(type(contract_version) is int and contract_version in (1, 2), "Unsupported evaluation contract version")
    multistep = contract_version == 2
    valid_scope = {"standard_task", "ordered_multistep_task"} if multistep else {"standard_task"}
    require(evaluation.get("evaluation_scope") in valid_scope and evaluation.get("evaluation_complete") is True, "A complete independently scored task is required")
    require(type(evaluation.get("success")) is bool and evaluation.get("whole_task_success") == evaluation["success"], "Invalid success evidence")
    trajectory = read_json(run / "trajectory.json")
    initial = read_json(run / "initial.json")
    final = read_json(run / "final.json")
    actions = read_json(run / "actions.json")
    control = read_json(run / "control-events.json")
    executed = read_json(run / "executed-plan.json")
    plan = read_json(run / "inference/plan.json")
    model = read_json(run / "inference/model-output.json")
    milestones = read_json(run / "action-milestones.json") if multistep else None
    inference_status = read_json(run / "inference/status.json")
    environment = read_json(run / "runtime-environment.json")
    inference_environment = read_json(run / "inference/runtime-environment.json")
    manifest = read_json(run / "source-manifest.json")
    require(isinstance(manifest, dict) and bool(manifest) and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in manifest.values()), "Invalid source manifest")
    require(inference_status.get("stage") == "infer" and inference_status.get("phase") == "completed", "Inference did not complete")
    require(executed.get("source") == "QWEN_REAL_INFERENCE" and executed.get("plan") == plan, "Executed plan is not the saved model plan")
    require(isinstance(actions, list) and isinstance(control, list) and bool(control), "Missing action/control evidence")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vla_trial.schema import parse_plan, validate_plan
    active_ids = evaluation["contract"].get("active_object_ids", ["red_block", "blue_block", "yellow_block"])
    validation_options = {"object_ids": active_ids, "protected_ids": evaluation["contract"]["protected_ids"],
                          "max_actions": 12 if multistep else 6, "allow_repeated_objects": multistep}
    validated_plan = validate_plan(plan, **validation_options).to_dict()
    require(parse_plan(model.get("raw_output"), **validation_options).to_dict() == validated_plan, "Saved plan differs from the actual raw model output")
    if multistep:
        require(model.get("settings", {}).get("task_mode") == "multistep", "Inference did not enable the recorded multistep task mode")
    expected_actions = [action for action in validated_plan["actions"] if action["action"] != "done"]
    require(len(actions) == len(expected_actions) and len(actions) > 0, "Executed action count does not match model plan")
    for actual, expected in zip(actions, expected_actions):
        require(isinstance(actual, dict) and all(actual.get(key, "pick_place" if key == "action" and not multistep else None) == value for key, value in expected.items()), "Executed action kind or arguments differ from the model plan")
    if multistep:
        require(isinstance(milestones, list) and len(milestones) == len(expected_actions), "Missing executed multistep milestones")
        for milestone, expected in zip(milestones, expected_actions):
            require(milestone.get("action") == expected and milestone.get("completed") is True, "Model plan and completed execution milestones differ")
    instruction = safe_text(status["instruction"])
    require(instruction == model.get("instruction") == evaluation["contract"]["instruction"], "Instruction and scoring contract differ")
    require(re.fullmatch(r"[0-9a-f]{40}", model.get("model_revision", "")) is not None, "Missing exact model revision")
    require(hashlib.sha256((run / "initial.png").read_bytes()).hexdigest() == model.get("image_sha256"), "Qwen input image hash does not match initial.png")
    from PIL import Image as PILImage
    for name in ("initial.png", "final.png"):
        with PILImage.open(run / name) as image:
            require(image.format == "PNG" and 32 <= image.width <= 4096 and 32 <= image.height <= 4096, "Invalid actual scene image")
            image.verify()
    require(initial.get("evidence", "").startswith("actual simulator camera") and final.get("evidence", "").startswith("actual simulator camera"), "Image metadata is not simulator observation evidence")
    require(trajectory[0]["positions"] == initial["object_positions"] and trajectory[-1]["positions"] == final["object_positions"], "Image and trajectory coordinates differ")
    if multistep:
        from vla_trial.evaluation_multistep import evaluate_multistep_task
        rescored = evaluate_multistep_task(evaluation["contract"], initial["object_positions"], trajectory, milestones)
        score_keys = ("success", "whole_task_success", "evaluation_complete", "violations", "milestone_results", "protected_results", "sample_count", "action_count")
    else:
        from vla_trial.evaluation import evaluate_task
        rescored = evaluate_task(evaluation["contract"], initial["object_positions"], trajectory)
        score_keys = ("success", "whole_task_success", "evaluation_complete", "violations", "goal_results", "protected_results", "sample_count", "stable_window")
    for key in score_keys:
        require(rescored[key] == evaluation[key], f"Independent recomputation differs: {key}")
    if multistep:
        required_milestones = {"initial_place_blue_block", "initial_place_yellow_block", "initial_sort",
                               "push_blue_outside_a", "yellow_stacked_on_blue",
                               "stack_preserved_during_transport", "intact_stack_in_b"}
        require(required_milestones <= set(evaluation["milestone_results"]), "Not every required multistep milestone could be evaluated")
    parent_events, inference_events = events(run / "events.jsonl"), events(run / "inference/events.jsonl")
    simulation_duration = float(trajectory[-1]["elapsed_s"])
    require(math.isfinite(simulation_duration) and simulation_duration > 0, "Invalid simulated duration")
    with PILImage.open(run / "initial.png") as image:
        image_width, image_height = image.size
    timing = {"wall": seconds(status["started_at"], status["finished_at"]),
              "model_wait": seconds(phase_time(parent_events, "waiting_for_qwen"), phase_time(parent_events, "moving")),
              "generation": seconds(phase_time(inference_events, "inferencing"), phase_time(inference_events, "completed")),
              "simulation": simulation_duration,
              "width": image_width, "height": image_height}
    for key, record in (("simulation_peak_mib", status), ("inference_peak_mib", inference_status)):
        value = record.get("resource_peak_measured_mib")
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0, "Missing measured process GPU-memory peak")
        timing[key] = value
    require(environment.get("physics_backend") == "cpu", "Unexpected physics backend; update report wording before use")
    for env in (environment, inference_environment):
        require(isinstance(env.get("packages"), dict) and "torch" in env["packages"], "Missing measured runtime versions")
        safe_text(env.get("admitted_gpu_name", ""))
    require("genesis-world" in environment["packages"] and "transformers" in inference_environment["packages"], "Incomplete library version evidence")
    if multistep:
        simulation_settings_ja(environment, evaluation["contract"])
    finger_geometry = recorded_finger_geometry(control, environment) if multistep else None
    summary = read_json(summary_path) if summary_path else None
    if summary is not None:
        if multistep or "task_id" in summary:
            require(summary.get("task_id") == evaluation["contract"]["task_id"], "Trial summary must refer only to this task, not combine earlier task versions")
        formal, development = summary["formal"], summary["development"]
        require(set(formal) == {"total", "success", "failure", "unscored"} and all(type(v) is int and v >= 0 for v in formal.values()), "Invalid formal-trial counts")
        require(formal["total"] >= 1 and formal["total"] == formal["success"] + formal["failure"] + formal["unscored"], "Formal trial counts do not sum")
        require(formal["success" if evaluation["success"] else "failure"] >= 1, "Summary excludes the selected trial's outcome")
        require(type(development["failed_runs"]) is int and development["failed_runs"] >= 0, "Invalid development count")
        require(isinstance(development["notes_ja"], list) and len(development["notes_ja"]) <= 8, "Invalid development notes")
        for note in development["notes_ja"]:
            safe_text(note)
    video_seconds = None
    if video is not None:
        probe = shutil.which("ffprobe")
        require(probe is not None, "ffprobe is required to verify the supplied submission video")
        result = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video.resolve(strict=True))], check=True, capture_output=True, text=True, timeout=30)
        video_seconds = float(json.loads(result.stdout)["format"]["duration"])
        require(math.isfinite(video_seconds) and 60 <= video_seconds <= 180, "Submission video must be 60-180 seconds")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return Evidence(run, status, evaluation, actions, plan, model, environment, inference_environment, timing, summary, digest, video_seconds, milestones=milestones, multistep=multistep, finger_geometry=finger_geometry)


def build_pdf(data: Evidence, output: Path, account: str, *, font_path: Path | None = None, fixture_banner: str | None = None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

    require(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", account) is not None, "Use only the OmniCampus account identifier")
    candidates = [font_path] if font_path else [Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), Path("/usr/share/fonts/truetype/ipaexg.ttf"), Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")]
    selected_font = next((path for path in candidates if path is not None and path.is_file()), None)
    require(selected_font is not None, "Supply --font with a Japanese TrueType font; the PDF must embed readable glyphs")
    pdfmetrics.registerFont(TTFont("ReportJapanese", str(selected_font)))
    font, navy, muted, teal = "ReportJapanese", colors.HexColor("#17354D"), colors.HexColor("#52616D"), colors.HexColor("#157979")
    body = ParagraphStyle("body", fontName=font, fontSize=9.2 if data.multistep else 9.6,
                          leading=14.3 if data.multistep else 15.7, spaceAfter=7, wordWrap="CJK", textColor=navy)
    small = ParagraphStyle("small", parent=body, fontSize=8.1, leading=12, spaceAfter=4, textColor=muted)
    title = ParagraphStyle("title", parent=body, fontSize=18, leading=26, spaceAfter=13)
    heading = ParagraphStyle("heading", parent=body, fontSize=12.2, leading=19, spaceBefore=8, spaceAfter=7, textColor=teal)
    caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER)
    story = []
    width = A4[0] - 84

    def p(text, style=body):
        return Paragraph(escape(safe_text(text)).replace("\n", "<br/>"), style)

    def add(text, style=body):
        story.append(p(text, style))

    def table(rows, column_widths=None, header=False):
        formatted = [[p(str(cell), small) for cell in row] for row in rows]
        tab = Table(formatted, colWidths=column_widths or [width * .32, width * .68], hAlign="LEFT")
        settings = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#DDE5E9")), ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F1F6F7"), colors.white])]
        if header:
            settings += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCECED"))]
        tab.setStyle(TableStyle(settings)); story.append(tab); story.append(Spacer(1, 7))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E4E8")); canvas.line(42, 39, A4[0] - 42, 39)
        canvas.setFont(font, 8); canvas.setFillColor(muted)
        canvas.drawString(42, 27, "Physical AI応用1 最終課題 | " + account)
        canvas.drawRightString(A4[0] - 42, 27, str(doc.page))
        if fixture_banner:
            canvas.setFillColor(colors.red); canvas.setFont(font, 10)
            canvas.drawCentredString(A4[0] / 2, A4[1] - 25, fixture_banner)
        canvas.restoreState()

    dt = datetime.fromisoformat(data.status["started_at"]).astimezone(timezone(timedelta(hours=9)))
    add("混合言語指示による\n多段階ブロック操作ロボット" if data.multistep else "視覚言語モデルによる\n指示追従型卓上整理ロボット", title)
    add(f"OmniCampus: {account}    実行日: {dt:%Y年%m月%d日} (JST)", small)
    add("1. 課題の概要と目的", heading)
    if data.multistep:
        add("視覚言語モデル（VLM）に実際の場面画像と日本語・英語・中国語を含む一つの指示を与え、操作の順序を計画させる。分類に続く押し出し、積み重ね、まとめての運搬までを、物理シミュレータ内の接触動作につなぎ、各段階を独立に評価することを目的とした。")
    else:
        add("画像と日本語の指示を理解する視覚言語モデル（VLM）を、物理シミュレータ内のロボット制御につなぐ。対象と移動先の選択をAIに任せ、実行可能な操作だけを制御器へ渡すことで、指示から実際の接触動作までを記録することを目的とした。")
    add("使用した指示", small)
    table([[data.status["instruction"]]], [width])
    if data.multistep:
        add("Franka Panda、赤・青・黄のブロック、青緑のトレーAと紫のトレーBを配置した。指示は、①青をA・黄をBへ分類、②青をAの外へ押し出す、③黄を青の上へ積む、④積層を保って2個を一緒にBへ運ぶ、と解釈した。赤は全工程で動かさない。")
    else:
        add("Franka Panda、赤・青・黄のブロック、青緑のトレーAと紫のトレーBを配置した。青をA、黄をBへ移し、赤は動かさない。入力と初期配置を固定した小規模な検証である。")
    add("2. AIと制御の接続", heading)
    add("実際のカメラ画像 + " + ("混合言語指示" if data.multistep else "日本語指示") + " → QwenのJSON行動案 → 対象・操作の検査 → 逆運動学と関節制御 → 位置履歴による成否判定")
    if data.multistep:
        add("Qwen3-VL-8B-Instructに操作語彙（pick_place / push_out / stack / move_stack）を提示し、回答計画は与えない。モデルが選んだ対象・支持物・目的地を検査し、赤を動かす操作や赤を支持物にする操作を拒否する。追加学習は行わず推論のみを用いた。")
    else:
        add("Qwen3-VL-8B-Instructは対象IDと移動先IDを出力する。許可外のID、不正な形式、赤いブロックへの操作を実行前に拒否する。追加学習は行わず、事前学習済みモデルの推論を用いた。")
    image_width = (width - 12) / 2
    image_height = image_width * data.timing["height"] / data.timing["width"]
    images = Table([[Image(str(data.run / "initial.png"), width=image_width, height=image_height), Image(str(data.run / "final.png"), width=image_width, height=image_height)], [p("実行前: モデルへ渡した実際の画像", caption), p("実行後: 同じ試行の最終状態", caption)]], colWidths=[width / 2, width / 2])
    images.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 0)]))
    story.append(images)
    story.append(PageBreak())

    add("3. 実行結果と判定方法", heading)
    success = "条件達成" if data.evaluation["success"] else "条件未達成"
    add(f"提示する正式試行の判定: {success}", title)
    add("以下の値は保存された位置履歴から再計算した。モデルのdone出力や動画の見た目だけで成功を判定していない。")
    labels = {"blue_block": "青いブロック", "yellow_block": "黄色いブロック", "red_block": "赤いブロック", "left_tray": "A", "right_tray": "B"}
    red = data.evaluation["protected_results"]["red_block"]
    criteria = data.evaluation["contract"]["success_criteria"]
    if data.multistep:
        values = data.evaluation["milestone_results"]
        titles = [("initial_place_blue_block", "青をAへ"), ("initial_place_yellow_block", "黄をBへ"),
                  ("initial_sort", "分類の完了"), ("push_blue_outside_a", "青をA外へ押す"),
                  ("yellow_stacked_on_blue", "黄を青の上へ"), ("stack_preserved_during_transport", "運搬中の積層"),
                  ("intact_stack_in_b", "積層したままB内へ")]
        rows = [["段階", "位置履歴からの確認", "判定"]]
        for key, label in titles:
            value = values[key]
            if key == "push_blue_outside_a":
                detail = f"上昇 {value['maximum_center_rise_m']*1000:.1f} mm / 水平 {value['planar_displacement_m']*1000:.1f} mm"
            elif key == "stack_preserved_during_transport":
                detail = f"{value['sampled_instants']} 時点で上下関係を検査"
            elif key == "initial_sort":
                detail = "2対象が各トレー内で安定"
            else:
                detail = f"終端の安定窓 {criteria['stable_duration_seconds']:.1f} 秒を検査"
            rows.append([label, detail, "達成" if value["passed"] else "未達成"])
        table(rows, [width*.29, width*.54, width*.17], header=True)
        final = values["intact_stack_in_b"]
        base_position, top_position = final["final_base_position"], final["final_top_position"]
        add("最終位置 (m): 青 [" + ", ".join(f"{x:.4f}" for x in base_position) + "]、黄 [" + ", ".join(f"{x:.4f}" for x in top_position) + "]。", small)
        red_peak = red["maximum_displacement_m"]
        add(f"赤の最大変位 {red_peak*1000:.3f} mm（許容 {criteria['protected_max_displacement_m']*1000:.1f} mm）。押し出し中の青の中心上昇は {criteria['push_max_center_rise_m']*1000:.0f} mm以下、終了時はブロック全体がAの外にあることを要求した。")
        add(f"積層条件は水平ずれ {criteria['stack_xy_tolerance_m']*1000:.0f} mm以下、高さ差 {criteria['block_height_m']*1000:.0f}±{criteria['stack_z_tolerance_m']*1000:.0f} mm。運搬の開始から終了まで確認し、終端は {criteria['stable_duration_seconds']:.1f} 秒以上安定、位置変動上限 {criteria['max_stable_position_spread_m']*1000:.0f} mm以下を要求した。", small)
        table([["時間（仮想 / 実時間）", f"{data.timing['simulation']:.2f} / {data.timing['wall']:.2f} 秒"],
               ["VLM工程待ち / 生成時間", f"{data.timing['model_wait']:.2f} / {data.timing['generation']:.2f} 秒"],
               ["最終積層の水平ずれ / 高さ差", f"{math.hypot(top_position[0]-base_position[0],top_position[1]-base_position[1])*1000:.2f} / {(top_position[2]-base_position[2])*1000:.2f} mm"],
               ["記録された位置サンプル", f"{data.evaluation['sample_count']} 点 / 最大許容間隔 {criteria['max_sample_gap_seconds']:.2f} 秒"]])
    else:
        rows = [["対象 / 目標", "最終位置 (m)", "判定"]]
        for object_id, value in data.evaluation["goal_results"].items():
            rows.append([f"{labels.get(object_id,object_id)} → {labels.get(value['target_id'],value['target_id'])}", ", ".join(f"{x:.4f}" for x in value["final_position"]), "達成" if value["passed"] else "未達成"])
        table(rows, [width*.34, width*.45, width*.21], header=True)
        add(f"赤いブロックの最大変位は {red['max_displacement_m']*1000:.3f} mm（許容 {red['limit_m']*1000:.1f} mm）。記録した全時点で確認した値であり、観測間の瞬間的な動きを保証するものではない。")
        tolerance = next(iter(data.evaluation["contract"]["targets"].values()))["center_tolerance"]
        add(f"目標中心からの許容差はX/Y/Z方向に各 {tolerance[0]:.3f}/{tolerance[1]:.3f}/{tolerance[2]:.3f} m。終端 {criteria['stable_duration_seconds']:.1f} 秒以上、領域内に留まり、位置変動の上限が {criteria['max_stable_position_spread_m']*1000:.1f} mm以下であることを判定条件とした。")
        table([["シミュレーション内の時間", f"{data.timing['simulation']:.2f} 秒（初期観測から最終観測）"], ["タスク全体の実時間", f"{data.timing['wall']:.2f} 秒（準備・記録終了を含む）"], ["VLM工程の待ち時間", f"{data.timing['model_wait']:.2f} 秒（資源確認・読込・推論等を含む）"], ["生成処理の実時間", f"{data.timing['generation']:.2f} 秒（推論開始から出力保存まで）"], ["記録された位置サンプル", f"{data.evaluation['sample_count']} 点 / 最大許容間隔 {criteria['max_sample_gap_seconds']:.2f} 秒"]])
    add("モデルが選んだ操作", heading)
    add("、".join(f"{index+1}: {action_description_ja(action, labels)}" for index, action in enumerate(data.actions)) + "。元のモデル回答、検査後の計画、実行記録の一致を確認した。", small if data.multistep else body)
    if data.summary:
        count = data.summary["formal"]
        add(f"この課題設定の正式試行は計 {count['total']} 回: 条件達成 {count['success']}、未達成・中止 {count['failure']}、未判定 {count['unscored']}。別設定の初期分類課題は合算せず、一般的な成功率の推定とも区別する。", small if data.multistep else body)
        development = data.summary["development"]
        add(f"別途、開発・調整時の失敗は {development['failed_runs']} 回。" + " ".join(development["notes_ja"]), small)
    else:
        add("本書は提示した1試行のケース記録であり、全試行に対する成功率は算出しない。開発時の描画・互換性・制御の調整は正式試行と分け、調整失敗を成功に数え直さない。", small)
    story.append(PageBreak())

    add("4. 実行条件と工夫した点", heading)
    packages = data.environment["packages"]
    model_settings = data.model["settings"]
    table([["モデル", "Qwen3-VL-8B-Instruct / " + str(model_settings.get("dtype", "記録なし"))], ["モデルrevision", data.model["model_revision"]], ["シミュレータ / Python", "Genesis " + packages["genesis-world"] + " / Python " + data.environment["python"]], ["推論環境", "PyTorch " + data.inference_environment["packages"]["torch"] + " / Transformers " + data.inference_environment["packages"]["transformers"]], ["物理演算 / GPU", "CPU物理演算 / " + data.environment["admitted_gpu_name"]], ["GPUメモリの計測最大値", f"描画工程 {data.timing['simulation_peak_mib']:,.0f} MiB / 推論工程 {data.timing['inference_peak_mib']:,.0f} MiB（各プロセス）"], ["カメラ / 生成上限", f"{data.timing['width']} × {data.timing['height']} px / max_pixels={model_settings.get('max_pixels')}, max_new_tokens={model_settings.get('max_new_tokens')}"], ["学習 / ソース識別", "追加学習なし / manifest SHA-256: " + data.manifest_digest[:20]]])
    if data.multistep:
        add(simulation_settings_ja(data.environment, data.evaluation["contract"]), small)
        add("工夫は、途中で言語が変わる指示を一つの順序付き行動列へ変換し、同じ物体への再操作を許しつつ赤の保護を維持する点にある。モデルの元回答、実行した各操作の開始・終了時刻、連続する位置履歴を対応づけ、初期分類から最終積層までを段階別に再計算した。")
        pair_method = "物理衝突を持つ延長指で2物体を同時に把持" if data.environment["simulation_settings"].get("finger_extensions") is True else "上下2個の境界付近を挟む把持"
        add("押し出しは閉じた指との接触、積み重ねは把持後の解放と重力支持、まとめての運搬は" + pair_method + "する方法で実装した。運搬中の上下関係は位置履歴で確認したが、接触力そのものや記録間の瞬間的な動作を、この評価だけで証明するものではない。")
        if data.environment["simulation_settings"].get("finger_extensions") is True:
            require(data.finger_geometry is not None, "Report lacks validated finger-extension geometry")
            dimensions = " × ".join(f"{value * 1000:g}" for value in data.finger_geometry["pad_size_m"])
            mass_g = data.finger_geometry["added_mass_kg_per_finger"] * 1000
            add(f"延長指は幅×厚さ×長さ {dimensions} mm、指当たり追加質量 {mass_g:g} g。Pandaに追加した独自の衝突形状であり、標準の指形状とは異なる。元の配布アセットは変更せず派生MJCFを生成する。実機での製作・動作検証は行っていない。", small)
        close_method = "把持は指を徐々に閉じてから力制御へ移し" if data.finger_geometry else "把持は指の力制御を用い"
        add(close_method + "、移動は逆運動学と関節制御を用いる。積み重ね時は物体位置の誤差を回数・移動量に上限のあるフィードバックで補正し、指を徐々に開いて解放する。把持・配置位置にはシミュレータの真値を利用しており、モデルが画素だけから連続動作を直接出力する学習済みVLAではない。", small)
    else:
        add("工夫は、自由文の指示を小さな行動語彙へ変換し、移動禁止対象を検査してから実行する点にある。操作の前後画像、モデルの元の回答、実行計画、物体の位置履歴を同じ試行へ対応づけ、評価をモデルの自己申告から切り離した。")
        add("把持は指の力制御、移動は逆運動学と関節の位置制御によって行う。位置情報にはシミュレータの真値を用いるため、画像から3次元把持位置を推定する実験や、連続動作を直接出力する学習済みVLAとは区別する。")
    add("5. 制限と再現方法", heading)
    if data.multistep:
        add("検証は固定配置の単純なブロックと、日・英・中が混在する一つの指示例に限られる。各言語や任意指示への一般的な性能を示す実験ではなく、初期分類だけの別課題とも成功率を合算しない。未知物体、視覚遮蔽、配置変更、実機での性能は未検証である。")
    else:
        add("対象は単純なブロックと固定配置に限られる。未知物体、任意の指示、視覚遮蔽、配置変更、実機での性能は検証していない。次の課題は、画像に基づく位置推定、失敗後の再観測、複数の初期配置での反復評価である。")
    add("提出コードには依存関係、タスク設定、実行処理、評価処理を含む。モデル重みと個別の機器設定は含めず、利用者自身が公式配布元からモデルを用意する。共有GPU環境では割り当て確認が必要であり、別ハードウェアでの再現性は未検証である。")
    if data.video_seconds is not None:
        add(f"提出動画の長さは {data.video_seconds:.2f} 秒（MP4メタデータで確認）。再生速度と省略区間は動画内の表示を参照する。", small)
    add("主な参照元・利用部分", heading)
    references = [
        ('Qwen3-VL-8B-Instruct: 事前学習済みモデルと推論API', 'https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct'),
        ('Genesis v1.4.4: シミュレーション・IK・公式把持例', 'https://github.com/Genesis-Embodied-AI/genesis-world/tree/v1.4.4'),
        ('Franka ROS: Pandaモデルの上流資料', 'https://github.com/frankarobotics/franka_ros/tree/develop/franka_description'),
    ]
    for text, url in references:
        story.append(Paragraph(f'<link href="{url}" color="#157979">{escape(text)}</link>', small))
    add("公式例の制御パラメータ等を参考に、タスク定義、モデル接続、制約検査、記録・評価を追加した。詳しい帰属は提出コードのTHIRD_PARTY_NOTICES.mdを参照。", small)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=48, bottomMargin=53, title="Physical AI応用1 最終課題", author=account, subject="VLMによる指示追従型卓上整理ロボット")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader
    pdf = PdfReader(output)
    require(2 <= len(pdf.pages) <= 3, f"PDF overflow: generated {len(pdf.pages)} pages; adjust layout before delivery")
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    safe_text(text[:4000])
    require(not re.search(PRIVATE_TEXT, text), "PDF contains a private path or email")
    return len(pdf.pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--account", default="shigin123")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--trial-summary", type=Path)
    parser.add_argument("--font", type=Path, help="Japanese TrueType font to embed; standard macOS/IPA locations are tried otherwise")
    args = parser.parse_args()
    expected = f"説明資料_PAI最終課題_{args.account}.pdf"
    require(args.output.name == expected, f"Output filename must be {expected}")
    data = load_evidence(args.run_dir, args.trial_summary, args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-", dir=args.output.parent) as directory:
        temporary = Path(directory) / expected
        pages = build_pdf(data, temporary, args.account, font_path=args.font)
        temporary.replace(args.output)
    print(json.dumps({"file": args.output.name, "pages": pages, "task_success": data.evaluation["success"], "source_manifest_sha256": data.manifest_digest, "visual_review_required": True}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Report refused: {exc}") from exc
