"""CPU-only controls for an explicitly authorized, bounded child process.

The web process never imports the model or simulator. The child must acquire
the course execution lock and pass the normal GPU admission checks itself.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from typing import Any


MAX_INSTRUCTION_CHARS = 800
DEFAULT_INSTRUCTION = "青いブロックをトレーAへ、黄色いブロックをトレーBへ移してください。赤いブロックは動かさないでください。"
DEFAULT_TASK = "configs/tasks/minimal_sort_ja.json"
MULTISTEP_TASK = "configs/tasks/multistep_stack_mixed.json"
ALLOWED_TASKS = frozenset({DEFAULT_TASK, MULTISTEP_TASK})


def _read_site(project_root: str | Path) -> dict:
    path = Path(project_root) / "configs/site.json"
    if path.stat().st_size > 65536:
        raise ValueError("oversized configuration")
    site = json.loads(path.read_text())
    if not isinstance(site, dict):
        raise ValueError("invalid configuration")
    return site


def active_task(project_root: str | Path, site: dict | None = None) -> tuple[str, dict]:
    """Resolve only the two reviewed task contracts; never accept a free path."""
    root = Path(project_root).resolve(strict=True)
    if site is None:
        try:
            site = _read_site(root)
        except FileNotFoundError:
            site = {}
    selected = site.get("active_task", DEFAULT_TASK)
    if not isinstance(selected, str) or selected not in ALLOWED_TASKS:
        raise ValueError("当前任务配置不在已支持的任务列表中。")
    path = root / selected
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root / "configs/tasks") or not resolved.is_file():
        raise ValueError("当前任务文件路径无效。")
    if resolved.stat().st_size > 65536:
        raise ValueError("当前任务文件过大。")
    task = json.loads(resolved.read_text())
    if not isinstance(task, dict):
        raise ValueError("当前任务文件无效。")
    task["instruction"] = validate_instruction(task.get("instruction"))
    return selected, task


def default_instruction(project_root: str | Path) -> str:
    try:
        _, task = active_task(project_root)
        return task["instruction"]
    except (OSError, ValueError, KeyError, TypeError):
        # Do not silently substitute the simpler task if an active complex
        # contract is missing or invalid. The start gate also stays closed.
        return ""


def task_description(project_root: str | Path) -> dict[str, str]:
    try:
        selected, _ = active_task(project_root)
    except (OSError, ValueError, KeyError, TypeError):
        return {"info": "当前任务配置尚未就绪，暂时不能执行。", "summary": "当前任务配置尚未就绪。可继续查看已有运行画面。"}
    if selected == MULTISTEP_TASK:
        return {
            "info": "支持按顺序抓放、推出托盘、堆叠和整体搬运。红色方块始终受保护；每轮从初始场景重新开始。",
            "summary": "当前预设：蓝色 → A、黄色 → B；将蓝色推出 A，把黄色叠到蓝色上，最后将两块一起搬到 B。全过程保持红色方块不动。自定义指令需结合录像判断，不能直接套用预设任务的评分。",
        }
    return {
        "info": "目前支持将蓝色或黄色方块搬到托盘 A / B；红色方块始终受保护。每轮从初始场景重新开始。",
        "summary": "当前预设：蓝色 → A，黄色 → B，红色保持不动。自定义指令需结合录像判断，不能直接套用预设任务的评分。",
    }


def validate_instruction(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请先输入给机械臂的任务。")
    value = value.strip()
    if len(value) > MAX_INSTRUCTION_CHARS or len(value.encode("utf-8")) > 4096:
        raise ValueError(f"任务请控制在 {MAX_INSTRUCTION_CHARS} 字以内。")
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise ValueError("任务中含有不支持的控制字符。")
    return value


def authorization_status(project_root: str | Path) -> tuple[bool, str]:
    try:
        site = _read_site(project_root)
    except (OSError, ValueError):
        return False, "当前为观看模式：尚未配置交互运行权限。"
    if site.get("interactive_enabled") is not True:
        return False, "正在验证当前任务的机械臂动作。验证完成后将启用「执行任务」，现在可以先编辑任务文字。"
    today = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    if site.get("authorization_date_jst") != today:
        return False, "今天的交互运行授权尚未启用，当前可以观看已有结果。"
    try:
        active_task(project_root, site)
    except (OSError, ValueError, KeyError, TypeError):
        return False, "当前任务配置尚未就绪，执行按钮暂时停用。"
    return True, "可以执行。每次点击会重置场景，并重新核对 GPU 资源；一轮最多运行 25 分钟。"


def execution_busy(project_root: str | Path) -> bool:
    """Probe only; execute.py owns the authoritative lock across its full run."""
    path = Path(project_root) / "outputs/course-execution.lock"
    if not path.exists():
        return False
    try:
        with path.open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    except OSError:
        # A lock that cannot be checked must never be treated as free.
        return True


def process_start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class CourseControls:
    """Track only the child created by this instance; never adopt other jobs."""

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve(strict=True)
        self._mutex = threading.Lock()
        self._child: subprocess.Popen | None = None
        self._start_ticks: str | None = None
        self._message = "尚未从本窗口启动任务。"

    def _running(self) -> bool:
        return self._child is not None and self._child.poll() is None

    def availability(self) -> dict[str, Any]:
        with self._mutex:
            authorized, notice = authorization_status(self.root)
            running = self._running()
            busy = running or execution_busy(self.root)
            if busy:
                notice = "当前有一轮任务正在运行，请等待结束后再执行新的任务。"
            return {"can_start": authorized and not busy, "can_stop": running,
                    "notice": notice, "message": self._message,
                    **task_description(self.root)}

    def start(self, instruction: Any) -> str:
        with self._mutex:
            try:
                instruction = validate_instruction(instruction)
            except ValueError as error:
                self._message = str(error)
                return self._message
            authorized, notice = authorization_status(self.root)
            if not authorized:
                self._message = notice
                return self._message
            if self._running() or execution_busy(self.root):
                self._message = "已有任务正在运行；没有重复启动。"
                return self._message
            try:
                selected_task, task = active_task(self.root)
            except (OSError, ValueError, KeyError, TypeError):
                self._message = "当前任务配置尚未就绪，本轮未启动。"
                return self._message
            timeout = shutil.which("timeout")
            nice = shutil.which("nice")
            if not timeout or not nice:
                self._message = "缺少有时限的运行工具，本轮未启动。"
                return self._message
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_interactive")
            run_dir = self.root / "outputs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            command = [timeout, "--signal=TERM", "--kill-after=20s", "1500s", nice, "-n", "19",
                       sys.executable, "-u", "-m", "vla_trial.execute", "--stage", "full",
                       "--project-root", str(self.root), "--run-dir", str(run_dir),
                       "--task", selected_task, "--instruction", instruction, "--wall-seconds", "1440"]
            try:
                with (run_dir / "execution.log").open("wb") as log:
                    child = subprocess.Popen(command, cwd=self.root, stdout=log,
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             start_new_session=True, close_fds=True, shell=False)
            except OSError:
                self._message = "运行进程未能启动；没有自动重试。"
                return self._message
            self._child = child
            self._start_ticks = process_start_ticks(child.pid)
            standard = instruction == task["instruction"]
            note = "将按标准任务独立评分。" if standard else "这是自定义任务：标准任务的放置评分不能代表这条指令是否完成。"
            self._message = f"已启动资源核对，画面会自动更新。{note}"
            return self._message

    def stop(self) -> str:
        with self._mutex:
            if not self._running():
                self._message = "本窗口没有可停止的运行。"
                return self._message
            child = self._child
            # A live Popen handle plus Linux start time prevents targeting a
            # recycled PID. Signal only the session created by this instance.
            if self._start_ticks is None or process_start_ticks(child.pid) != self._start_ticks:
                self._message = "无法确认本窗口启动的进程身份，未发送停止信号。"
                return self._message
            try:
                if os.getpgid(child.pid) != child.pid:
                    self._message = "运行进程组身份不匹配，未发送停止信号。"
                    return self._message
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                self._message = "本轮运行已经结束。"
                return self._message
            self._message = "已请求停止本窗口启动的这一轮，正在保存已有录像和释放资源。"
            return self._message
