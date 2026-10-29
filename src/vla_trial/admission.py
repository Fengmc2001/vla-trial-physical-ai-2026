"""Explicit, bounded execution using the host's existing shared admission lock.

This module is only reached from a fresh operator receipt. Importing it does not
query a GPU. Site paths live in private configuration, not in public source.
"""
from __future__ import annotations
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

_LOCKS = []


def release_startup_lock():
    while _LOCKS:
        f = _LOCKS.pop()
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def admit_course_run(runtime, receipt):
    from .safety import AdmissionError, validate_receipt
    output = Path(runtime["admission_output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    site = json.loads(Path(runtime["site_config"]).read_text())
    mode = site.get("admission_mode", "shared_adapter")
    if mode == "dedicated_exclusive":
        from .dedicated_admission import admit_dedicated_run
        _LOCKS.append(admit_dedicated_run(runtime, receipt))
        return
    if mode != "shared_adapter":
        raise AdmissionError("Unrecognized admission mode; no execution was started.")
    uuid = runtime["gpu_uuid"]
    if receipt["host"] != site["host"] or uuid not in site["allowed_gpu_uuids"]:
        raise AdmissionError("This host/GPU was not assigned to the course run.")
    budget = int(runtime["peak_budget_mib"])
    if not 512 <= budget <= int(site["max_task_budget_mib"]):
        raise AdmissionError("Requested task budget is outside the site allowance.")
    lib_path = Path(site["admission_library"])
    if hashlib.sha256(lib_path.read_bytes()).hexdigest() != site["admission_library_sha256"]:
        raise AdmissionError("Site admission implementation changed; review before use.")
    spec = importlib.util.spec_from_file_location("vla_site_policy", lib_path)
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    shared_lock = lib.HERE / ("runs/queues/admission_%s.lock" % uuid[-8:])
    handle = shared_lock.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise AdmissionError("Another job is being admitted; no automatic retry.")
    record = {"pid": os.getpid(), "gpu_uuid": uuid, "budget_mib": budget,
              "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")",1)[1].split()[19],
              "operator_reference": receipt["allocation_reference"], "snapshots": []}
    try:
        def check():
            validate_receipt(runtime)
            s = lib.full_snapshot(uuid)
            growth = 0
            for proc in s["compute_processes"]:
                ident = proc["identity"]
                if not ident["ours"]:
                    raise AdmissionError("A foreign compute process is present.")
                cmd = ident["cmd"]
                if ("scripts/wp1_readout.py" in cmd or "scripts/s3_vlm_eval.py" in cmd) and ident["cwd"] == str(lib.HERE):
                    peak = re.search(r"--est-peak-mib\s+(\d+)", cmd)
                    if peak is None:
                        raise AdmissionError("Existing task has no verified peak estimate.")
                    growth += max(0, int(peak.group(1)) - proc["used_mib"])
                    # Match the live process to its current progress record.
                    matches = list((lib.HERE / "runs/wp1").glob(f"*_{proc['pid']}/progress.json"))
                    matches += list((lib.HERE / "runs/wp1_002v2").glob(f"*_{proc['pid']}/progress.json"))
                    if "scripts/s3_vlm_eval.py" in cmd:
                        matches += list((lib.HERE / "runs/wp2").glob(f"*_{proc['pid']}/progress.json"))
                    if not matches:
                        raise AdmissionError("Existing readout process has no current task record.")
                    progress = json.loads(matches[-1].read_text())
                    if progress.get("pid") != proc["pid"]:
                        raise AdmissionError("Existing task PID/record mismatch.")
                    proc["progress_path"] = str(matches[-1])
                    proc["progress"] = progress
                elif "scripts/gpu_bench_monitor.py" in cmd and ident["cwd"] == str(lib.HERE):
                    registered = json.loads((lib.HERE / f"runs/claimq/gpu_bench_{uuid[-8:]}.json").read_text())
                    if registered.get("pid") != proc["pid"]:
                        raise AdmissionError("Monitor registration does not match.")
                elif str(Path(runtime["project_root"]).resolve()) in cmd:
                    registrations = list((Path(runtime["project_root"]) / "outputs").glob("*/admission.json"))
                    registrations += list((Path(runtime["project_root"]) / "outputs").glob("*/inference/admission.json"))
                    matches = []
                    for registration in registrations:
                        value = json.loads(registration.read_text())
                        if value.get("pid") == proc["pid"] and value.get("start_ticks") == ident["startticks"] and value.get("gpu_uuid") == uuid:
                            matches.append(value)
                    if len(matches) != 1 or "vla_trial.execute" not in cmd:
                        raise AdmissionError("Course process lacks a unique matching identity and budget record.")
                    growth += max(0, matches[0]["budget_mib"] - proc["used_mib"])
                else:
                    raise AdmissionError("Own process has not been reviewed for this launch.")
            s["graphics_identities"] = []
            for pid in s["graphics_pids"]:
                ident = lib.proc_identity(pid)
                s["graphics_identities"].append(ident)
                desktop = ident["user"] == "gdm" and ident["cmd"].startswith("/usr/lib/xorg/Xorg ")
                course = ident["ours"] and str(Path(runtime["project_root"]).resolve()) in ident["cmd"]
                if not (desktop or course):
                    raise AdmissionError("Unknown graphics process; refusing launch.")
            s["reviewed_growth_mib"] = growth
            record["snapshots"].append(s)
            _write(output / "admission.json", record)
            if s["host_available_gib"] < 64 or s["nas_free_gib"] < 32:
                raise AdmissionError("Insufficient host/NAS headroom.")
            if s["used_mib"] + growth + budget > s["total_mib"] * 0.90:
                raise AdmissionError("Existing peaks plus this task exceed the 90% memory boundary.")
            return s
        first = check()
        time.sleep(30)
        second = check()
        identity = lambda s: (sorted((p["pid"], p["identity"]["startticks"]) for p in s["compute_processes"]), sorted((p["pid"],p["startticks"]) for p in s["graphics_identities"]))
        if first["boot_id"] != second["boot_id"] or identity(first) != identity(second):
            raise AdmissionError("Process identities changed during the 30-second admission check.")
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        # Torch import alone is before the final CUDA check; device calls follow.
        import torch
        torch.set_num_threads(2)
        final = check()
        if final["boot_id"] != first["boot_id"] or identity(final) != identity(second):
            raise AdmissionError("Host or process identities changed before CUDA initialization.")
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(int(runtime["allocator_cap_mib"]) / total, 0)
        if runtime.get("uses_renderer"):
            from .graphics_binding import select_egl_device
            binding = select_egl_device(site["gpu_pci_bus_ids"][uuid])
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["EGL_DEVICE_ID"] = str(binding["egl_index"])
            record["graphics_binding"] = binding
        record["decision"] = "ADMITTED_STARTUP_LOCK_HELD"
        _write(output / "admission.json", record)
        _LOCKS.append(handle)
    except BaseException as error:
        record.update(decision="REFUSED", error=f"{type(error).__name__}: {error}")
        _write(output / "admission.json", record)
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        raise
