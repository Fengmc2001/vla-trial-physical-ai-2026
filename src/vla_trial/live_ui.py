"""CPU-only viewer and gated controls for a separate, bounded robot run.

The web process cannot import inference or simulation modules. An explicitly
enabled execution button may start a separate process with its own admission.
The producer writes
outputs/live.json as {"run_dir": "outputs/<run-id>"}, preferably atomically.
The selected directory supplies latest.jpg and status.json. Only finalized
simulation.mp4 files may appear in the video player.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


_TERMINAL_STATES = frozenset({"complete", "completed", "finished", "success", "succeeded", "failed", "aborted", "cancelled", "stopped", "action_limit_reached", "round_limit_reached", "model_reported_done_unverified"})
_VISIBLE_STATUS_KEYS = ("phase", "status", "stage", "model_used", "instruction", "action", "current_action", "action_index", "actions", "evaluation", "evaluation_scope", "instruction_matches_contract", "score", "whole_task_success", "message", "error", "elapsed_seconds", "elapsed_s", "run_id", "video_finalized")


def _json_object(path: Path, max_bytes: int) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise ValueError("JSON file is too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def resolve_run_directory(project_root: str | Path, marker: Mapping[str, Any]) -> Path:
    """Accept a relative descendant of outputs; reject traversal and symlinks out."""
    root = Path(project_root).resolve(strict=True)
    outputs = (root / "outputs").resolve(strict=True)
    if not outputs.is_relative_to(root) or not outputs.is_dir():
        raise ValueError("Outputs directory must remain inside the project")
    relative = marker.get("run_dir")
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("run_dir must be a project-relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Absolute and traversing run paths are not allowed")
    candidate = (root / path).resolve(strict=True)
    if candidate == outputs or not candidate.is_relative_to(outputs) or not candidate.is_dir():
        raise ValueError("run_dir must select a directory below this project's outputs")
    return candidate


def _contained_file(run_directory: Path, name: str) -> Path:
    path = (run_directory / name).resolve(strict=True)
    if not path.is_relative_to(run_directory) or not path.is_file():
        raise ValueError("Run artifact must stay inside its run directory")
    return path


def video_is_finalized(status: Mapping[str, Any]) -> bool:
    if "video_finalized" in status:
        return status["video_finalized"] is True
    return any(isinstance(status.get(key), str) and status[key].lower() in _TERMINAL_STATES for key in ("phase", "status"))


def summarize_run(status: Mapping[str, Any]) -> str:
    """Describe only the current run, keeping calibration separate from AI work."""
    phase = status.get("phase", status.get("status", "waiting_for_run"))
    stage = status.get("stage")
    title = {"preview": "相机验证", "control": "动作调试", "full": "指令任务", "infer": "AI 推理"}.get(stage, "运行状态")
    completed = phase in {"complete", "completed", "finished", "success", "succeeded"}
    if phase == "failed":
        detail = "本轮未能完成；正在检查原因。已有画面和录像会保留。"
    elif completed and stage == "preview":
        # Old camera-only runs may contain a false placement score. No objects
        # were meant to move, so that score cannot mean the task failed.
        detail = "相机验证完成，尚未执行搬运。"
    elif completed and stage == "control":
        detail = "动作调试已结束。本轮用于检查机械臂能否执行预设动作。"
    elif completed:
        evaluation = status.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        if evaluation.get("evaluation_scope") == "custom_task_unscored":
            detail = "本轮已结束。自定义任务需要结合录像确认，尚未给出任务成功判定。"
        elif evaluation.get("evaluation_scope") in {"standard_task", "ordered_multistep_task"} and evaluation.get("whole_task_success") is True:
            detail = "标准任务完成：独立检查通过。可在下方观看完整录像。"
        elif evaluation.get("evaluation_scope") in {"standard_task", "ordered_multistep_task"} and evaluation.get("whole_task_success") is False:
            detail = "本轮已结束，但尚未满足全部标准任务要求。可在下方查看录像。"
        else:
            detail = "本轮运行已结束，请结合录像确认结果。"
    elif phase == "moving":
        action = status.get("action")
        action = action if isinstance(action, Mapping) else {}
        objects = {"blue_block": "蓝色方块", "yellow_block": "黄色方块"}
        targets = {"left_tray": "托盘 A", "right_tray": "托盘 B"}
        kind = action.get("action")
        object_name = objects.get(action.get("object_id"))
        target_name = targets.get(action.get("target_id"))
        if kind == "push_out" and object_name and targets.get(action.get("source_target_id")):
            detail = f"正在将{object_name}推出{targets[action['source_target_id']]}。"
        elif kind == "stack" and object_name and objects.get(action.get("support_id")):
            detail = f"正在将{object_name}叠到{objects[action['support_id']]}上。"
        elif kind == "move_stack" and objects.get(action.get("base_id")) and objects.get(action.get("top_id")) and target_name:
            detail = f"正在将{objects[action['base_id']]}与上方的{objects[action['top_id']]}一起搬到{target_name}。"
        else:
            detail = f"正在将{object_name}搬到{target_name}。" if object_name and target_name else "机械臂正在执行动作。"
    else:
        detail = {
            "admission": "正在核对本轮可用资源，请稍候。",
            "building_scene": "正在搭建桌面场景，首张相机画面准备好后会自动显示。",
            "scene_ready": "场景和相机已就绪，正在准备本轮动作。" if stage != "preview" else "相机已就绪，正在记录验证画面。",
            "waiting_for_qwen": "等待 AI 读取画面与任务文字，并生成动作计划。",
            "loading_model": "正在准备 AI，画面可能暂时保持不动。",
            "inferencing": "AI 正在读取画面与任务文字。",
            "waiting_for_run": "等待运行。启用后可在上方输入任务并点击「执行任务」。",
            "waiting_for_valid_run": "等待本轮运行记录。",
            "waiting_for_status": "正在读取本轮状态。",
            "stopped": "本轮已停止。",
            "cancelled": "本轮已停止。",
            "aborted": "本轮已中止。",
        }.get(phase, "正在更新本轮状态。")
    ai = ""
    if stage in {"preview", "control"} or status.get("model_used") is False:
        ai = "\n\nAI：本轮未使用。"
    elif status.get("model_used") is True:
        ai = "\n\nAI：本轮已使用模型读取画面并生成动作。"
    return f"**{title}**\n\n{detail}{ai}"


def read_live_snapshot(project_root: str | Path) -> dict[str, Any]:
    """Read fresh producer output. Missing/partial files show waiting, never art."""
    snapshot: dict[str, Any] = {
        "frame": None, "video": None, "status": {"phase": "waiting_for_run"},
        "notice": "正在等待真实相机画面；当前没有可显示的运行帧。",
    }
    root = Path(project_root).resolve(strict=True)
    marker_path = root / "outputs/live.json"
    if not marker_path.exists():
        return snapshot
    try:
        # The pointer itself is also restricted; an escaping symlink is rejected.
        resolved_marker = marker_path.resolve(strict=True)
        outputs = (root / "outputs").resolve(strict=True)
        if not outputs.is_relative_to(root) or not resolved_marker.is_relative_to(outputs):
            raise ValueError("Run pointer escapes outputs")
        marker = _json_object(resolved_marker, 16384)
        run_directory = resolve_run_directory(root, marker)
    except (OSError, ValueError, RuntimeError):
        snapshot["status"] = {"phase": "waiting_for_valid_run"}
        snapshot["notice"] = "当前运行标记暂不可读或路径不合法，等待有效的真实运行输出。"
        return snapshot

    try:
        status = _json_object(_contained_file(run_directory, "status.json"), 262144)
        snapshot["status"] = {key: status[key] for key in _VISIBLE_STATUS_KEYS if key in status}
        if "error" in snapshot["status"]:
            # Raw exception text can contain private model or server paths.
            snapshot["status"]["error"] = "本轮未能完成；详细原因已保存在项目的运行日志中。"
    except (OSError, ValueError, RuntimeError):
        status = {}
        snapshot["status"] = {"phase": "waiting_for_status"}

    try:
        frame_path = _contained_file(run_directory, "latest.jpg")
        if frame_path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Camera frame is too large")
        from PIL import Image

        try:
            with Image.open(frame_path) as frame:
                if frame.format != "JPEG" or max(frame.size) > 4096 or min(frame.size) < 1:
                    raise ValueError("Expected a bounded JPEG camera frame")
                snapshot["frame"] = frame.convert("RGB").copy()
        except Image.DecompressionBombError as exc:
            raise ValueError("Camera frame exceeds decoder limits") from exc
        age = max(0.0, time.time() - frame_path.stat().st_mtime)
        if status.get("phase") in _TERMINAL_STATES:
            snapshot["notice"] = "本轮已结束，当前显示保存的最后一帧。"
        else:
            snapshot["notice"] = f"相机画面约 {age:.0f} 秒前更新；AI 处理任务时可能暂时保持不动。"
    except (OSError, ValueError, RuntimeError):
        snapshot["notice"] = "正在等待有效的真实相机帧；文件写入期间会在下一秒自动重试。"

    if video_is_finalized(status):
        try:
            video = _contained_file(run_directory, "simulation.mp4")
            if video.stat().st_size > 0:
                snapshot["video"] = str(video)
        except (OSError, ValueError, RuntimeError):
            pass
    return snapshot


def _submission_file(root: Path, relative: Any, suffix: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise ValueError("Submission artifact requires a relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != suffix:
        raise ValueError("Invalid submission artifact path")
    resolved = (root / path).resolve(strict=True)
    outputs = (root / "outputs").resolve(strict=True)
    if not outputs.is_relative_to(root) or not resolved.is_relative_to(outputs) or not resolved.is_file():
        raise ValueError("Submission artifact must be inside project outputs")
    return resolved


@lru_cache(maxsize=8)
def _verified_video_hash(path: str, size: int, modified_ns: int, expected: str) -> bool:
    """Hash a changed export once, not on every one-second UI refresh."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def read_submission_artifact(project_root: str | Path) -> dict[str, Any] | None:
    """Expose a finalized course export only for the currently selected task.

    The operator copies the final MP4 and timeline into outputs, then atomically
    writes outputs/submission.json with task, video and manifest relative paths.
    This does not create, render, publish or upload anything.
    """
    from .control_ui import active_task

    root = Path(project_root).resolve(strict=True)
    try:
        selected, task = active_task(root)
        marker_path = _submission_file(root, "outputs/submission.json", ".json")
        marker = _json_object(marker_path, 16384)
        if marker.get("task") != selected:
            return None
        video = _submission_file(root, marker.get("video"), ".mp4")
        manifest_path = _submission_file(root, marker.get("manifest"), ".json")
        manifest = _json_object(manifest_path, 262144)
        if (manifest.get("video") != video.name or manifest.get("task_id") != task.get("task_id")
                or manifest.get("task_schema_version") != task.get("schema_version")
                or not isinstance(task.get("task_id"), str) or type(task.get("schema_version")) is not int):
            return None
        duration = manifest["video_info"]["duration_s"]
        if type(duration) not in {int, float} or not 60 <= duration <= 180:
            return None
        expected_hash = manifest.get("video_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            return None
        info = video.stat()
        if not 0 < info.st_size <= 512 * 1024 * 1024:
            return None
        if not _verified_video_hash(str(video), info.st_size, info.st_mtime_ns, expected_hash):
            return None
        seconds = round(duration)
        label = f"提交用成片 · {seconds // 60} 分 {seconds % 60:02d} 秒"
        return {"video": str(video), "label": label,
                "download_label": f"下载提交用视频（{seconds // 60}:{seconds % 60:02d}）",
                "identity": (str(video), expected_hash)}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


def _prepare_cpu_process() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .ui import PreparationImportGuard

    if any(name in sys.modules for name in PreparationImportGuard.blocked):
        raise RuntimeError("Start the live viewer in a fresh CPU-only process")
    if not any(isinstance(finder, PreparationImportGuard) for finder in sys.meta_path):
        sys.meta_path.insert(0, PreparationImportGuard())


def build_live_ui(project_root: str | Path):
    _prepare_cpu_process()
    import gradio as gr
    from .control_ui import CourseControls, default_instruction

    root = Path(project_root).resolve(strict=True)
    initial = read_live_snapshot(root)
    final_export = read_submission_artifact(root)
    controls = CourseControls(root)
    availability = controls.availability()
    with gr.Blocks(title="尝试VLA · 实时观看机械臂", analytics_enabled=False, delete_cache=(60, 600)) as ui:
        gr.Markdown("# 尝试VLA · 给机械臂下达任务\n场景已由程序搭建：一台机械臂、蓝／黄／红三个方块，以及青色托盘 A 和紫色托盘 B。下方每秒刷新真实仿真画面。")
        instruction = gr.Textbox(value=default_instruction(root), lines=3, max_lines=6,
                                 label="输入给机械臂的要求（支持日文、中文、英文）",
                                 info=availability["info"])
        with gr.Row():
            start = gr.Button("执行任务", variant="primary", interactive=availability["can_start"])
            stop = gr.Button("停止本窗口启动的任务", interactive=availability["can_stop"])
            preset = gr.Button("填入当前课题指令")
        readiness = gr.Markdown(availability["notice"])
        request_status = gr.Markdown(availability["message"])
        task_note = gr.Markdown(availability["summary"])
        with gr.Row():
            with gr.Column(scale=3):
                camera = gr.Image(value=initial["frame"], label="机械臂画面", interactive=False, type="pil")
                notice = gr.Markdown(initial["notice"])
            with gr.Column(scale=2):
                summary = gr.Markdown(summarize_run(initial["status"]))
        submission_video = gr.Video(value=final_export["video"] if final_export else None,
                                    label=final_export["label"] if final_export else "提交用成片",
                                    visible=final_export is not None, interactive=False)
        download = gr.DownloadButton(label=final_export["download_label"] if final_export else "下载提交用视频",
                                     value=final_export["video"] if final_export else None,
                                     visible=final_export is not None)
        recording = gr.Video(value=initial["video"], label="本轮原始录像（不含课题说明）", visible=initial["video"] is not None, interactive=False)
        with gr.Accordion("运行记录（可选查看）", open=False):
            state = gr.JSON(value=initial["status"], label="当前运行的阶段、指令、动作与独立检查")
        timer = gr.Timer(value=1.0, active=True)
        last_export = [final_export["identity"] if final_export else None]
        last_recording = [initial["video"]]

        def refresh():
            current = read_live_snapshot(root)
            available = controls.availability()
            export = read_submission_artifact(root)
            export_id = export["identity"] if export else None
            if export_id != last_export[0]:
                last_export[0] = export_id
                export_update = gr.update(value=export["video"] if export else None,
                                          label=export["label"] if export else "提交用成片", visible=export is not None)
                download_update = gr.update(value=export["video"] if export else None,
                                            label=export["download_label"] if export else "下载提交用视频", visible=export is not None)
            else:
                export_update, download_update = gr.update(), gr.update()
            raw_update = gr.update()
            if current["video"] != last_recording[0]:
                last_recording[0] = current["video"]
                raw_update = gr.update(value=current["video"], visible=current["video"] is not None)
            return (current["frame"], current["status"], current["notice"],
                    raw_update,
                    gr.update(interactive=available["can_start"]),
                    gr.update(interactive=available["can_stop"]), available["notice"],
                    summarize_run(current["status"]),
                    gr.update(info=available["info"]), available["summary"],
                    export_update, download_update)

        start.click(controls.start, inputs=instruction, outputs=request_status, api_name=False)
        stop.click(controls.stop, outputs=request_status, api_name=False)
        preset.click(lambda: default_instruction(root), outputs=instruction, api_name=False)
        timer.tick(refresh, outputs=[camera, state, notice, recording, start, stop, readiness, summary, instruction, task_note, submission_video, download], api_name=False)
        gr.Markdown("模型读取相机图像并选择动作，运动控制器执行。关闭网页不会停止当前任务；运行受时间和资源检查限制。模型说“完成”并不等于任务成功，请结合录像与独立评分判断。")
    return ui


def main() -> None:
    parser = argparse.ArgumentParser(description="尝试VLA CPU-only live camera and authorized subprocess controls")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    build_live_ui(root).launch(server_name="127.0.0.1", server_port=7867, share=False,
                               inbrowser=False, show_error=False, ssr_mode=False,
                               allowed_paths=[str(root / "outputs")])


if __name__ == "__main__":
    main()
