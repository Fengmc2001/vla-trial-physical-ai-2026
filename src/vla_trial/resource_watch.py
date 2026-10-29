"""Bounded, read-only observation of this process's CUDA and graphics memory.

This is a sampled abort guard, not a hard GPU allocation limit. Importing the
module performs no GPU query; use it only inside an admitted execution process.
It never signals another process and never persists other users' process data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET


class ResourceWatchError(RuntimeError):
    """The admitted process cannot remain within its measured resource bounds."""


def own_gpu_usage(xml_text: str, pid: int) -> dict[str, float]:
    """Extract total per-process memory from C, G, or C+G entries, in MiB.

    NVIDIA's process table reports per-process memory, not separate compute and
    graphics allocations. If the same PID appears twice on a GPU, taking the
    maximum avoids counting the same allocation twice. A missing PID means zero
    usage on that device, while missing/unavailable memory for a present PID is
    an error rather than an assumed zero.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ResourceWatchError("GPU process query returned malformed XML.") from error
    if root.tag != "nvidia_smi_log":
        raise ResourceWatchError("Unexpected GPU query document.")
    devices = root.findall("gpu")
    if not devices:
        raise ResourceWatchError("GPU process query returned no devices.")
    result: dict[str, float] = {}
    for device in devices:
        uuid = (device.findtext("uuid") or "").strip()
        if not uuid.startswith("GPU-"):
            raise ResourceWatchError("GPU process query omitted a device UUID.")
        processes = device.find("processes")
        if processes is None:
            raise ResourceWatchError("GPU process query omitted the process table.")
        for item in processes.findall("process_info"):
            raw_pid = (item.findtext("pid") or "").strip()
            if not raw_pid.isdecimal():
                raise ResourceWatchError("GPU process query contained an unreadable PID.")
            if int(raw_pid) != pid:
                continue
            raw_memory = (item.findtext("used_memory") or "").strip()
            match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s+MiB", raw_memory)
            if not match:
                raise ResourceWatchError("This process's GPU memory is unavailable or has unexpected units.")
            memory = float(match.group(1))
            if not math.isfinite(memory):
                raise ResourceWatchError("This process's GPU memory is not finite.")
            result[uuid] = max(memory, result.get(uuid, 0.0))
    return result


def query_own_gpu_usage(pid: int) -> dict[str, float]:
    """Read the complete NVIDIA process table without creating GPU contexts."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        # Do not copy arbitrary command stderr or other processes into evidence.
        raise ResourceWatchError(f"GPU process query failed ({type(error).__name__}).") from error
    return own_gpu_usage(result.stdout, pid)


class ProcessResourceWatch:
    """Watch only the calling PID until close or a bounded wall-time deadline.

    The initial check runs synchronously. Later failures save a receipt before
    sending SIGTERM to this same PID so the execution runner can close its own
    simulator/model. The caller must retain an outer hard timeout because GPU
    driver calls may delay Python signal handling.
    """

    def __init__(self, gpu_uuid: str, budget_mib: int, output_path: Path,
                 *, wall_seconds: int, interval_seconds: float = 5.0):
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
            raise ValueError("An exact admitted GPU UUID is required.")
        if not isinstance(budget_mib, int) or not 1 <= budget_mib <= 1_000_000:
            raise ValueError("GPU memory budget must be a positive MiB integer.")
        if not 1 <= wall_seconds <= 1800:
            raise ValueError("Resource watch must be bounded to at most 1800 seconds.")
        if not 5 <= interval_seconds <= 30:
            raise ValueError("Resource watch interval must be 5 to 30 seconds.")
        self.pid = os.getpid()
        self.gpu_uuid = gpu_uuid
        self.budget_mib = budget_mib
        self.output_path = Path(output_path)
        self.wall_seconds = wall_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = None
        self._started = None
        self._closed = False
        self.record = {
            "schema_version": 1, "pid": self.pid, "gpu_uuid": gpu_uuid,
            "budget_mib": budget_mib, "interval_seconds": interval_seconds,
            "wall_seconds": wall_seconds, "mechanism": "sampled_own_process_abort_guard",
            "hard_allocation_cap": False, "status": "not_started", "query_count": 0,
            "peak_measured_mib": 0.0, "peak_by_gpu_mib": {}, "samples": [],
        }

    def _save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.output_path)

    def _check(self):
        if os.getpid() != self.pid:
            raise ResourceWatchError("Resource watcher cannot be reused across processes.")
        if self._started is None:
            raise ResourceWatchError("Resource watcher was not started.")
        elapsed = time.monotonic() - self._started
        if elapsed >= self.wall_seconds:
            raise ResourceWatchError("Resource watch reached its bounded wall-time limit.")
        usage = query_own_gpu_usage(self.pid)
        sample = {"at": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": round(time.monotonic() - self._started, 3),
                  "memory_by_gpu_mib": usage}
        self.record["query_count"] += 1
        self.record["samples"].append(sample)
        self.record["samples"] = self.record["samples"][-120:]
        for uuid, memory in usage.items():
            self.record["peak_by_gpu_mib"][uuid] = max(memory, self.record["peak_by_gpu_mib"].get(uuid, 0.0))
        self.record["peak_measured_mib"] = max(self.record["peak_measured_mib"], usage.get(self.gpu_uuid, 0.0))
        self.record["last_checked_at"] = sample["at"]
        if any(uuid != self.gpu_uuid for uuid in usage):
            raise ResourceWatchError("This process appeared on a GPU outside its admitted UUID.")
        if usage.get(self.gpu_uuid, 0.0) > self.budget_mib:
            raise ResourceWatchError("This process exceeded its admitted measured GPU memory budget.")
        # Deadline can be reached while the bounded NVIDIA query is running.
        if time.monotonic() - self._started >= self.wall_seconds:
            raise ResourceWatchError("Resource watch reached its bounded wall-time limit.")
        self._save()

    def _fail(self, error):
        self.record.update(status="failed", error=f"{type(error).__name__}: {error}",
                           stopped_at=datetime.now(timezone.utc).isoformat())
        self._stop.set()
        self._save()

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self._check()
            except BaseException as error:
                try:
                    self._fail(error)
                finally:
                    # pid is captured internally, and a post-fork watcher refuses
                    # reuse. Never expose a PID argument that can target a job.
                    if os.getpid() == self.pid:
                        os.kill(self.pid, signal.SIGTERM)
                return

    def start(self):
        if self._started is not None or self._closed:
            raise ResourceWatchError("Resource watcher can only start once.")
        self._started = time.monotonic()
        self.record.update(status="watching", started_at=datetime.now(timezone.utc).isoformat())
        try:
            self._check()
        except BaseException as error:
            self._fail(error)
            raise
        self._thread = threading.Thread(target=self._run, name="vla-own-resource-watch", daemon=True)
        self._thread.start()
        return self

    def close(self):
        """Stop this watcher only; do not signal processes or start another query."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=32)
            if self._thread.is_alive():
                raise ResourceWatchError("Resource watcher did not finish its bounded query.")
        if self._started is not None and self.record["status"] != "failed":
            self.record.update(status="closed", stopped_at=datetime.now(timezone.utc).isoformat())
            self._save()
        self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
