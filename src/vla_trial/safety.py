"""Fail-closed, standard-library-only admission checks.

Receipts are operator records, NOT cryptographically verified permissions.
No query, model load, renderer creation, or GPU initialization occurs at import.
The prepared baseline deliberately cannot grant graphics execution until its
EGL/CUDA device binding has been implemented and verified on the chosen host.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


class AdmissionError(RuntimeError):
    pass


class RuntimeDisabledError(AdmissionError):
    pass


class GraphicsBindingUnverifiedError(AdmissionError):
    pass


@dataclass(frozen=True)
class AdmissionGrant:
    host: str
    gpu_uuid: str
    expires_at: datetime
    pid: int
    _proof: object


_ISSUED_PROOFS: set[object] = set()
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")


def _utc_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise AdmissionError("Receipt timestamps must be ISO 8601 strings with timezone.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdmissionError("Invalid receipt timestamp.") from exc
    if parsed.tzinfo is None:
        raise AdmissionError("Receipt timestamps require a timezone.")
    return parsed.astimezone(timezone.utc)


def validate_receipt(
    runtime: Mapping[str, Any],
    *,
    host: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read a small local receipt without contacting the GPU or importing ML."""
    if runtime.get("enabled") is not True:
        raise RuntimeDisabledError("GPU execution is disabled (runtime.enabled=false).")
    current_host = host or socket.gethostname()
    if "gpu200" in current_host.lower() or any(key.startswith("SLURM_") for key in os.environ):
        raise AdmissionError("GPU200/Slurm execution is unsupported by this baseline; use the site scheduler workflow.")
    gpu_uuid = runtime.get("gpu_uuid", "")
    if not isinstance(gpu_uuid, str) or not _GPU_UUID.fullmatch(gpu_uuid):
        raise AdmissionError("An exact full GPU UUID is required; ordinal IDs and MIG selections are not accepted.")
    receipt_value = runtime.get("approval_receipt")
    if not isinstance(receipt_value, str) or not receipt_value:
        raise AdmissionError("A current operator approval receipt is required.")
    receipt_path = Path(receipt_value)
    if not receipt_path.is_absolute():
        raise AdmissionError("approval_receipt must be an absolute path.")
    try:
        if receipt_path.stat().st_size > 16384:
            raise AdmissionError("Receipt is too large.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdmissionError("Cannot read a valid operator receipt.") from exc
    if not isinstance(receipt, dict):
        raise AdmissionError("Receipt must be a JSON object.")
    if receipt.get("authorized") is not True or receipt.get("purpose") != "vla-trial":
        raise AdmissionError("Receipt must explicitly authorize this project.")
    if receipt.get("host") != current_host or receipt.get("gpu_uuid") != gpu_uuid:
        raise AdmissionError("Receipt host or GPU UUID does not match this process.")
    for field in ("operator", "allocation_reference"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise AdmissionError(f"Receipt requires {field}.")
    issued_at, expires_at = _utc_time(receipt.get("issued_at")), _utc_time(receipt.get("expires_at"))
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise AdmissionError("Current admission time must be timezone-aware.")
    if not issued_at <= current_time < expires_at:
        raise AdmissionError("Receipt is not yet valid or has expired.")
    if current_time - issued_at > timedelta(minutes=5):
        raise AdmissionError("Admission receipt must have been issued within five minutes.")
    if expires_at - issued_at > timedelta(hours=1):
        raise AdmissionError("Receipt validity may not exceed one hour.")
    return {**receipt, "expires_at_parsed": expires_at}


def _assert_gpu_idle(gpu_uuid: str) -> None:
    """Future runtime check, called only after explicit authorization.

Uses nvidia-smi XML to include both compute and graphics processes. A check
failure is a denial, never a reason to retry or kill a process.
"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x", "-i", gpu_uuid],
            check=True, capture_output=True, text=True, timeout=10,
        )
        root = ET.fromstring(result.stdout)
    except (OSError, subprocess.SubprocessError, ET.ParseError) as exc:
        raise AdmissionError("Cannot verify selected GPU state; refusing execution.") from exc
    gpus = root.findall("gpu")
    if len(gpus) != 1 or gpus[0].findtext("uuid") != gpu_uuid:
        raise AdmissionError("nvidia-smi did not return exactly the selected GPU UUID.")
    gpu = gpus[0]
    process_group = gpu.find("processes")
    if process_group is None or process_group.findall("process_info"):
        raise AdmissionError("Selected GPU has processes or process state is unavailable.")
    if (process_group.text or "").strip() not in {"", "None"}:
        raise AdmissionError("Selected GPU process state is unavailable.")
    try:
        memory_text = gpu.findtext("fb_memory_usage/used", "").split()
        utilization_text = gpu.findtext("utilization/gpu_util", "").split()
        if memory_text[1] != "MiB" or utilization_text[1] != "%":
            raise ValueError("Unknown measurement units")
        used = float(memory_text[0])
        utilization = float(utilization_text[0])
    except (ValueError, IndexError) as exc:
        raise AdmissionError("GPU memory/utilization could not be read.") from exc
    if not math.isfinite(used) or not math.isfinite(utilization) or used < 0 or utilization < 0:
        raise AdmissionError("GPU usage measurements are invalid.")
    if used > 512 or utilization > 5:
        raise AdmissionError("Selected GPU is not idle; do not displace another workload.")


def authorize_runtime(runtime: Mapping[str, Any]) -> AdmissionGrant:
    receipt = validate_receipt(runtime)
    if runtime.get("execution_profile") == "bounded-course-run":
        from .admission import admit_course_run
        admit_course_run(runtime, receipt)
        proof = object()
        _ISSUED_PROOFS.add(proof)
        return AdmissionGrant(socket.gethostname(), runtime["gpu_uuid"], receipt["expires_at_parsed"], os.getpid(), proof)
    # Deliberate stop in this prepared baseline. CUDA_VISIBLE_DEVICES alone
    # cannot prove an OpenGL/EGL renderer chooses that same physical device.
    # Replace this stop only after implementing a host-tested device check.
    raise GraphicsBindingUnverifiedError(
        "Real execution remains locked: EGL/CUDA GPU binding has not been verified "
        "on this host. Complete that validation before enabling the prepared runtime."
    )


def _admit_after_verified_graphics_binding(runtime: Mapping[str, Any]) -> AdmissionGrant:
    """Reserved integration step; not called by the baseline public runtime.

    A future verified graphics-binding implementation may call this only after
    checking physical device identity. It is intentionally not a bypass API.
    """
    raise GraphicsBindingUnverifiedError("No verified graphics-binding implementation is installed.")


def _three_idle_checks(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Admission building block for future reviewed graphics integration."""
    receipt = validate_receipt(runtime)
    _assert_gpu_idle(receipt["gpu_uuid"])
    time.sleep(30)
    validate_receipt(runtime)
    _assert_gpu_idle(receipt["gpu_uuid"])
    validate_receipt(runtime)
    _assert_gpu_idle(receipt["gpu_uuid"])
    return receipt


def require_grant(grant: AdmissionGrant) -> None:
    """Guard direct use of Qwen/Genesis classes as well as Runtime.run."""
    if not isinstance(grant, AdmissionGrant) or grant._proof not in _ISSUED_PROOFS:
        raise AdmissionError("No valid admission grant; direct model/simulation loading is disabled.")
    if grant.pid != os.getpid() or grant.host != socket.gethostname():
        raise AdmissionError("Admission belongs to a different process or host.")
    if datetime.now(timezone.utc) >= grant.expires_at:
        raise AdmissionError("Admission expired; stop without automatically restarting.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != grant.gpu_uuid:
        raise AdmissionError("CUDA device binding differs from the admitted GPU UUID.")
