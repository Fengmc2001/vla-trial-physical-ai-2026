"""Admission for an explicitly allocated, exclusive Linux NVIDIA GPU.

This portable mode is intentionally unsuitable for co-running with existing
research jobs. A local lock cannot grant allocation or coordinate independent
users/checkouts; the operator must already have an exclusive allocation. No GPU
query, library import, context creation or allocation happens at module import.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import time
import xml.etree.ElementTree as ET

from .safety import AdmissionError, validate_receipt


_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_TRUSTED_XORG = frozenset({"/usr/lib/xorg/Xorg", "/usr/libexec/Xorg", "/usr/bin/Xorg"})


def _write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _mib(text: str | None) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s+MiB", (text or "").strip())
    if match is None:
        raise AdmissionError("GPU memory measurement is unavailable or has unexpected units.")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise AdmissionError("GPU memory measurement is not finite.")
    return value


def process_identity(pid: int) -> dict:
    """Verify Linux process identity twice; never signal or attach to it."""
    if not isinstance(pid, int) or pid <= 0:
        raise AdmissionError("Invalid process PID.")
    base = Path("/proc") / str(pid)
    try:
        first_stat = (base / "stat").read_text()
        first_uid = (base / "status").read_text().split("\nUid:", 1)[1].splitlines()[0].split()
        if len(first_uid) != 4:
            raise ValueError("incomplete UID fields")
        uid, euid, suid, fsuid = map(int, first_uid)
        start_ticks = first_stat.rsplit(")", 1)[1].split()[19]
        if not start_ticks.isdecimal():
            raise ValueError("invalid start ticks")
        argv = [part.decode("utf-8", errors="strict") for part in (base / "cmdline").read_bytes().split(b"\0") if part]
        executable = os.readlink(base / "exe")
        executable_stat = Path(executable).stat()
        second_stat = (base / "stat").read_text()
        second_uid = (base / "status").read_text().split("\nUid:", 1)[1].splitlines()[0].split()
        if second_stat.rsplit(")", 1)[1].split()[19] != start_ticks or second_uid != first_uid:
            raise ValueError("process identity changed while reading")
        user, effective_user = pwd.getpwuid(uid).pw_name, pwd.getpwuid(euid).pw_name
    except (OSError, ValueError, IndexError, KeyError, UnicodeError) as error:
        raise AdmissionError("Cannot verify a GPU process identity from /proc; refusing launch.") from error
    return {
        "pid": pid, "uid": uid, "euid": euid, "suid": suid, "fsuid": fsuid,
        "user": user, "effective_user": effective_user,
        "start_ticks": start_ticks, "argv": argv, "executable": executable,
        "executable_uid": executable_stat.st_uid,
        "executable_mode": stat.S_IMODE(executable_stat.st_mode),
        "executable_device": executable_stat.st_dev, "executable_inode": executable_stat.st_ino,
    }


def _is_system_xorg(identity: dict) -> bool:
    executable = identity.get("executable")
    argv = identity.get("argv")
    return (
        identity.get("user") in {"root", "gdm"}
        and identity.get("effective_user") in {"root", "gdm"}
        and executable in _TRUSTED_XORG
        and isinstance(argv, list) and bool(argv) and argv[0] in _TRUSTED_XORG
        and identity.get("executable_uid") == 0
        and isinstance(identity.get("executable_mode"), int)
        and identity["executable_mode"] & 0o022 == 0
        and str(identity.get("start_ticks", "")).isdecimal()
    )


def parse_gpu_snapshot(xml_text: str, gpu_uuid: str, expected_pci: str) -> dict:
    """Inspect one exact GPU, rejecting all compute and unknown graphics use."""
    from .graphics_binding import normalize_pci_bus_id
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise AdmissionError("GPU query returned malformed XML.") from error
    devices = root.findall("gpu")
    if root.tag != "nvidia_smi_log" or len(devices) != 1:
        raise AdmissionError("Expected exactly one queried GPU.")
    device = devices[0]
    if device.findtext("uuid") != gpu_uuid:
        raise AdmissionError("Queried physical GPU UUID differs from the allocation.")
    try:
        pci = normalize_pci_bus_id(device.findtext("pci/pci_bus_id"))
        if pci != normalize_pci_bus_id(expected_pci):
            raise AdmissionError("GPU UUID and configured PCI identity do not match.")
    except (TypeError, ValueError, RuntimeError) as error:
        if isinstance(error, AdmissionError):
            raise
        raise AdmissionError("Cannot verify the selected GPU PCI identity.") from error
    mig = (device.findtext("mig_mode/current_mig") or "").strip()
    if mig not in {"Disabled", "N/A"}:
        raise AdmissionError("Dedicated admission requires a verified non-MIG physical GPU.")
    total = _mib(device.findtext("fb_memory_usage/total"))
    used = _mib(device.findtext("fb_memory_usage/used"))
    free = _mib(device.findtext("fb_memory_usage/free"))
    if total <= 0 or used > total or free > total or free + used > total + 1:
        raise AdmissionError("GPU memory measurements are inconsistent.")
    group = device.find("processes")
    if group is None or (group.text or "").strip() not in {"", "None"}:
        raise AdmissionError("GPU process table is unavailable.")
    graphics = []
    seen = set()
    for item in group.findall("process_info"):
        kind = (item.findtext("type") or "").strip()
        if kind != "G":
            # Includes C, C+G, MPS and unrecognized process kinds. Ownership
            # does not permit co-running in this exclusive mode.
            raise AdmissionError("Dedicated GPU has a compute or unrecognized process; no co-running is allowed.")
        raw_pid = (item.findtext("pid") or "").strip()
        if not raw_pid.isdecimal() or int(raw_pid) <= 0 or int(raw_pid) in seen:
            raise AdmissionError("Graphics process PID is missing, invalid or duplicated.")
        pid = int(raw_pid)
        seen.add(pid)
        identity = process_identity(pid)
        if not _is_system_xorg(identity):
            raise AdmissionError("Only a verified system Xorg graphics process may remain on the dedicated GPU.")
        graphics.append({**identity, "used_mib": _mib(item.findtext("used_memory"))})
    return {"gpu_uuid": gpu_uuid, "pci_bus_id": pci, "total_mib": total,
            "used_mib": used, "free_mib": free, "compute_processes": [],
            "graphics_identities": graphics, "mig_mode": mig}


def dedicated_snapshot(gpu_uuid: str, expected_pci: str, project_root: Path) -> dict:
    """One bounded read of GPU, process, host-memory and project-disk state."""
    try:
        result = subprocess.run(["nvidia-smi", "-q", "-x", "-i", gpu_uuid],
                                check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise AdmissionError(f"Cannot query allocated GPU ({type(error).__name__}); no automatic retry.") from error
    snapshot = parse_gpu_snapshot(result.stdout, gpu_uuid, expected_pci)
    try:
        available_line = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        _, available, unit = available_line.split()
        if unit != "kB" or not available.isdecimal():
            raise ValueError("invalid host memory units")
        available_gib = int(available) / 2**20
        disk_free_gib = shutil.disk_usage(project_root).free / 2**30
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", boot_id):
            raise ValueError("invalid boot identity")
    except (OSError, ValueError, StopIteration) as error:
        raise AdmissionError("Cannot verify host or project storage headroom.") from error
    return {**snapshot, "boot_id": boot_id, "host_available_gib": available_gib,
            "disk_free_gib": disk_free_gib, "captured_at": datetime.now(timezone.utc).isoformat()}


def _identity(snapshot):
    return (snapshot["boot_id"], snapshot["gpu_uuid"], snapshot["pci_bus_id"],
            sorted((p["pid"], p["uid"], p["euid"], p["start_ticks"], p["executable"],
                    p["executable_device"], p["executable_inode"], tuple(p["argv"]))
                   for p in snapshot["graphics_identities"]))


def admit_dedicated_run(runtime, receipt):
    """Validate exclusive allocation, initialize the selected device, return lock.

    On success the caller must retain the returned locked file until its model
    or scene has finished startup, then release it using the normal admission
    cleanup. On every failure this function closes its own lock. It never
    modifies allocation files or terminates an existing process.
    """
    validated = validate_receipt(runtime)
    if any(receipt.get(key) != validated.get(key) for key in ("host", "gpu_uuid", "issued_at", "operator", "allocation_reference")):
        raise AdmissionError("Caller receipt differs from the current validated receipt.")
    root = Path(runtime["project_root"]).resolve(strict=True)
    output = Path(runtime["admission_output"]).resolve()
    if not output.is_relative_to(root / "outputs"):
        raise AdmissionError("Admission evidence must stay inside project outputs.")
    site = json.loads(Path(runtime["site_config"]).read_text())
    if site.get("admission_mode") != "dedicated_exclusive" or site.get("dedicated_exclusive_allocation_confirmed") is not True:
        raise AdmissionError("Dedicated execution requires a separately confirmed exclusive GPU allocation.")
    today = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    if site.get("authorization_date_jst") != today or site.get("host") != validated["host"]:
        raise AdmissionError("Dedicated allocation date or host does not match.")
    if site.get("operator") != validated["operator"] or site.get("authorization_reference") != validated["allocation_reference"]:
        raise AdmissionError("Dedicated operator allocation reference differs from the validated receipt.")
    gpu_uuid = runtime["gpu_uuid"]
    if not isinstance(gpu_uuid, str) or not _UUID.fullmatch(gpu_uuid) or gpu_uuid not in site.get("allowed_gpu_uuids", []):
        raise AdmissionError("Exact GPU UUID is not in the exclusive allocation.")
    if site.get("simulation_gpu_uuid") == site.get("inference_gpu_uuid"):
        raise AdmissionError("This runner needs separately allocated simulation and inference GPUs; it does not co-run them.")
    expected_pci = site.get("gpu_pci_bus_ids", {}).get(gpu_uuid)
    if not expected_pci:
        raise AdmissionError("Allocated GPU has no configured PCI identity.")
    budget = runtime.get("peak_budget_mib")
    allocator_cap = runtime.get("allocator_cap_mib")
    maximum = site.get("max_task_budget_mib")
    if any(type(value) is not int for value in (budget, allocator_cap, maximum)) or not 512 <= allocator_cap <= budget <= maximum:
        raise AdmissionError("Invalid process memory budget or allocator cap.")
    cache = root / ".cache"
    cache.mkdir(exist_ok=True)
    if not cache.resolve().is_relative_to(root):
        raise AdmissionError("Project lock directory escapes the project.")
    lock_path = cache / f"dedicated-admission-{gpu_uuid}.lock"
    if lock_path.is_symlink():
        raise AdmissionError("Admission lock must not be a symlink.")
    output.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise AdmissionError("This allocated GPU is already being admitted in this project; no automatic retry.")
    except BaseException:
        handle.close()
        raise
    record = {"mode": "dedicated_exclusive", "pid": os.getpid(), "gpu_uuid": gpu_uuid,
              "budget_mib": budget, "allocator_cap_mib": allocator_cap,
              "operator_reference": validated["allocation_reference"], "snapshots": [],
              "allocation_source": "operator-confirmed external exclusive allocation",
              "lock_scope": "this project only; not a scheduler allocation"}
    try:
        def check():
            validate_receipt(runtime)
            if site["authorization_date_jst"] != datetime.now(timezone(timedelta(hours=9))).date().isoformat():
                raise AdmissionError("Dedicated allocation date changed during admission.")
            snapshot = dedicated_snapshot(gpu_uuid, expected_pci, root)
            record["snapshots"].append(snapshot)
            _write(output / "admission.json", record)
            if snapshot["host_available_gib"] < 64 or snapshot["disk_free_gib"] < 32:
                raise AdmissionError("Dedicated run requires at least 64 GiB available host RAM and 32 GiB free project storage.")
            if snapshot["free_mib"] - budget < snapshot["total_mib"] * 0.10:
                raise AdmissionError("Dedicated run must leave at least ten percent of GPU memory free after its full budget.")
            return snapshot
        first = check()
        time.sleep(30)
        second = check()
        if _identity(first) != _identity(second):
            raise AdmissionError("GPU/host/process identities changed during the 30-second check.")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        import torch
        if torch.cuda.is_initialized():
            raise AdmissionError("Dedicated admission requires a fresh process with no initialized CUDA context.")
        torch.set_num_threads(2)
        final = check()
        if _identity(second) != _identity(final):
            raise AdmissionError("GPU/host/process identities changed before CUDA initialization.")
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        if allocator_cap >= total_mib:
            raise AdmissionError("Allocator cap exceeds the selected CUDA device memory.")
        from .graphics_binding import _cuda_identity_for_pci, select_egl_device
        cuda_binding = _cuda_identity_for_pci(expected_pci)
        if cuda_binding["device_uuid"].lower() != gpu_uuid.removeprefix("GPU-").lower():
            raise AdmissionError("CUDA physical device UUID differs from the allocated NVIDIA UUID.")
        record["cuda_binding"] = cuda_binding
        torch.cuda.set_per_process_memory_fraction(allocator_cap / total_mib, 0)
        if runtime.get("uses_renderer"):
            binding = select_egl_device(expected_pci)
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["EGL_DEVICE_ID"] = str(binding["egl_index"])
            record["graphics_binding"] = binding
        record["decision"] = "ADMITTED_STARTUP_LOCK_HELD"
        _write(output / "admission.json", record)
        return handle
    except BaseException as error:
        record.update(decision="REFUSED", error=f"{type(error).__name__}: {error}")
        try:
            _write(output / "admission.json", record)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        raise
