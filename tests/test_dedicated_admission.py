"""Portable exclusive-GPU admission tests; no GPU query or context is created."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from vla_trial.dedicated_admission import admit_dedicated_run, dedicated_snapshot, parse_gpu_snapshot
from vla_trial.safety import AdmissionError


UUID = "GPU-11111111-2222-3333-4444-555555555555"
OTHER_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
PCI = "00000000:47:00.0"


def process(pid=100, kind="G", memory="4 MiB"):
    return f"<process_info><pid>{pid}</pid><type>{kind}</type><used_memory>{memory}</used_memory></process_info>"


def xml(*entries, uuid=UUID, pci=PCI, mig="Disabled", process_text="", free="90000 MiB"):
    return (f"<nvidia_smi_log><gpu><uuid>{uuid}</uuid><pci><pci_bus_id>{pci}</pci_bus_id></pci>"
            f"<mig_mode><current_mig>{mig}</current_mig></mig_mode>"
            "<fb_memory_usage><total>97887 MiB</total><used>100 MiB</used>"
            f"<free>{free}</free></fb_memory_usage><processes>{process_text}{''.join(entries)}</processes>"
            "</gpu></nvidia_smi_log>")


def xorg_identity():
    return {"pid": 100, "uid": 126, "euid": 126, "suid": 126, "fsuid": 126,
            "user": "gdm", "effective_user": "gdm", "start_ticks": "1234",
            "argv": ["/usr/lib/xorg/Xorg", ":0"], "executable": "/usr/lib/xorg/Xorg",
            "executable_uid": 0, "executable_mode": 0o755,
            "executable_device": 1, "executable_inode": 22}


def snapshot():
    return {"gpu_uuid": UUID, "pci_bus_id": "0000:47:00.0", "total_mib": 97887.0,
            "used_mib": 100.0, "free_mib": 90000.0, "compute_processes": [],
            "graphics_identities": [], "mig_mode": "Disabled",
            "boot_id": "11111111-1111-1111-1111-111111111111", "host_available_gib": 128,
            "disk_free_gib": 100, "captured_at": "fixture"}


class XMLAdmissionTests(unittest.TestCase):
    def test_empty_selected_device_has_verified_uuid_pci_and_memory(self):
        result = parse_gpu_snapshot(xml(), UUID, "0000:47:00.0")
        self.assertEqual(result["gpu_uuid"], UUID)
        self.assertEqual(result["pci_bus_id"], "0000:47:00.0")
        self.assertEqual(result["free_mib"], 90000)

    def test_any_compute_combined_or_unknown_process_is_denied_before_identity_lookup(self):
        for kind in ("C", "C+G", "M", "", "unknown"):
            with self.subTest(kind=kind), patch("vla_trial.dedicated_admission.process_identity") as identify:
                with self.assertRaisesRegex(AdmissionError, "no co-running"):
                    parse_gpu_snapshot(xml(process(pid=os.getpid(), kind=kind)), UUID, PCI)
                identify.assert_not_called()

    def test_system_xorg_is_allowed_with_complete_identity(self):
        with patch("vla_trial.dedicated_admission.process_identity", return_value=xorg_identity()):
            result = parse_gpu_snapshot(xml(process()), UUID, PCI)
        self.assertEqual(result["graphics_identities"][0]["start_ticks"], "1234")
        self.assertEqual(result["graphics_identities"][0]["used_mib"], 4)

    def test_user_graphics_spoofed_executable_and_writable_system_binary_are_denied(self):
        for changes in ({"user": "operator"}, {"effective_user": "operator"},
                        {"executable": "/tmp/Xorg"}, {"argv": ["/tmp/Xorg"]},
                        {"executable_uid": 1000}, {"executable_mode": 0o777}, {"start_ticks": None}):
            identity = dict(xorg_identity(), **changes)
            with self.subTest(changes=changes), patch("vla_trial.dedicated_admission.process_identity", return_value=identity):
                with self.assertRaisesRegex(AdmissionError, "verified system Xorg"):
                    parse_gpu_snapshot(xml(process()), UUID, PCI)

    def test_pid_unreadable_or_recycled_is_denied(self):
        with patch("vla_trial.dedicated_admission.process_identity", side_effect=AdmissionError("cannot verify")):
            with self.assertRaisesRegex(AdmissionError, "cannot verify"):
                parse_gpu_snapshot(xml(process()), UUID, PCI)

    def test_uuid_pci_mig_and_unknown_process_state_fail_closed(self):
        for data in (xml(uuid=OTHER_UUID), xml(pci="0000:20:00.0"), xml(mig="Enabled"),
                     xml(mig=""), xml(process_text="N/A"), xml(free="N/A"), "<broken"):
            with self.subTest(data=data), self.assertRaises(AdmissionError):
                parse_gpu_snapshot(data, UUID, PCI)

    def test_query_timeout_is_thirty_seconds_with_no_shell_or_retry(self):
        with patch("vla_trial.dedicated_admission.subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 30)) as run:
            with self.assertRaisesRegex(AdmissionError, "no automatic retry"):
                dedicated_snapshot(UUID, PCI, Path("."))
        run.assert_called_once_with(["nvidia-smi", "-q", "-x", "-i", UUID], check=True, capture_output=True, text=True, timeout=30)


class DedicatedFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "configs").mkdir()
        self.output = self.root / "outputs/test"
        self.site_path = self.root / "configs/site.json"
        self.site = {"admission_mode": "dedicated_exclusive", "dedicated_exclusive_allocation_confirmed": True,
                     "authorization_date_jst": datetime.now(timezone(timedelta(hours=9))).date().isoformat(),
                     "host": "fixture-host", "operator": "fixture-operator", "authorization_reference": "exclusive allocation fixture",
                     "allowed_gpu_uuids": [UUID, OTHER_UUID], "simulation_gpu_uuid": UUID,
                     "inference_gpu_uuid": OTHER_UUID, "gpu_pci_bus_ids": {UUID: PCI}, "max_task_budget_mib": 20000}
        self.write_site()
        self.receipt = {"host": "fixture-host", "gpu_uuid": UUID, "issued_at": "fixture",
                        "operator": "fixture-operator", "allocation_reference": "exclusive allocation fixture"}
        self.runtime = {"project_root": str(self.root), "admission_output": str(self.output),
                        "site_config": str(self.site_path), "gpu_uuid": UUID,
                        "peak_budget_mib": 6144, "allocator_cap_mib": 4096, "uses_renderer": True}
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=MagicMock(return_value=False),
                                   get_device_properties=MagicMock(return_value=SimpleNamespace(total_memory=97887 * 2**20)),
                                   set_per_process_memory_fraction=MagicMock()), set_num_threads=MagicMock())
        self.validate = patch("vla_trial.dedicated_admission.validate_receipt", return_value=self.receipt).start()
        self.snapshots = patch("vla_trial.dedicated_admission.dedicated_snapshot", side_effect=lambda *args: deepcopy(snapshot())).start()
        self.sleep = patch("vla_trial.dedicated_admission.time.sleep").start()
        self.cuda_identity = patch("vla_trial.graphics_binding._cuda_identity_for_pci", return_value={"device_uuid": UUID[4:], "pci_bus_id": "0000:47:00.0"}).start()
        self.egl = patch("vla_trial.graphics_binding.select_egl_device", return_value={"egl_index": 7, "pci_bus_id": "0000:47:00.0"}).start()
        self.modules = patch.dict("sys.modules", {"torch": self.torch}).start()
        self.environment = patch.dict(os.environ, {}, clear=False).start()
        self.addCleanup(patch.stopall)

    def write_site(self):
        self.site_path.write_text(json.dumps(self.site))

    def assert_lock_released(self):
        path = self.root / ".cache" / f"dedicated-admission-{UUID}.lock"
        if path.exists():
            with path.open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)

    def test_three_checks_precede_cuda_and_return_held_startup_lock(self):
        order = []
        self.snapshots.side_effect = lambda *args: (order.append("snapshot") or deepcopy(snapshot()))
        self.torch.cuda.get_device_properties.side_effect = lambda *args: (order.append("cuda") or SimpleNamespace(total_memory=97887 * 2**20))
        handle = admit_dedicated_run(self.runtime, self.receipt)
        self.addCleanup(handle.close)
        self.assertEqual(order, ["snapshot", "snapshot", "snapshot", "cuda"])
        self.sleep.assert_called_once_with(30)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], UUID)
        self.assertEqual(os.environ["EGL_DEVICE_ID"], "7")
        with Path(handle.name).open("a+") as other:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = json.loads((self.output / "admission.json").read_text())
        self.assertEqual(len(evidence["snapshots"]), 3)
        self.assertEqual(evidence["decision"], "ADMITTED_STARTUP_LOCK_HELD")
        handle.close()
        self.assert_lock_released()

    def test_false_allocation_flag_refuses_before_gpu_query(self):
        self.site["dedicated_exclusive_allocation_confirmed"] = False
        self.write_site()
        with self.assertRaisesRegex(AdmissionError, "confirmed exclusive"):
            admit_dedicated_run(self.runtime, self.receipt)
        self.snapshots.assert_not_called()
        self.torch.cuda.get_device_properties.assert_not_called()

    def test_wrong_date_host_gpu_and_same_gpu_roles_refuse(self):
        for changes in ({"authorization_date_jst": "2000-01-01"}, {"host": "different"},
                        {"allowed_gpu_uuids": [OTHER_UUID]}, {"inference_gpu_uuid": UUID}):
            original = dict(self.site)
            self.site.update(changes)
            self.write_site()
            with self.subTest(changes=changes), self.assertRaises(AdmissionError):
                admit_dedicated_run(self.runtime, self.receipt)
            self.site = original
        self.snapshots.assert_not_called()

    def test_insufficient_memory_host_or_storage_releases_lock_without_cuda(self):
        for field, value in (("free_mib", 10000), ("host_available_gib", 63.9), ("disk_free_gib", 31.9)):
            result = dict(snapshot(), **{field: value})
            self.snapshots.side_effect = lambda *args: deepcopy(result)
            with self.subTest(field=field), self.assertRaises(AdmissionError):
                admit_dedicated_run(self.runtime, self.receipt)
            self.assert_lock_released()
        self.torch.cuda.get_device_properties.assert_not_called()

    def test_changed_boot_between_checks_refuses_before_cuda(self):
        changed = dict(snapshot(), boot_id="different-boot")
        self.snapshots.side_effect = [snapshot(), changed]
        with self.assertRaisesRegex(AdmissionError, "identities changed"):
            admit_dedicated_run(self.runtime, self.receipt)
        self.torch.cuda.get_device_properties.assert_not_called()
        self.assert_lock_released()

    def test_final_changed_graphics_identity_refuses_before_cuda(self):
        changed = dict(snapshot(), graphics_identities=[xorg_identity()])
        self.snapshots.side_effect = [snapshot(), snapshot(), changed]
        with self.assertRaisesRegex(AdmissionError, "before CUDA"):
            admit_dedicated_run(self.runtime, self.receipt)
        self.torch.cuda.get_device_properties.assert_not_called()
        self.assert_lock_released()

    def test_preinitialized_cuda_is_not_accepted(self):
        self.torch.cuda.is_initialized.return_value = True
        with self.assertRaisesRegex(AdmissionError, "fresh process"):
            admit_dedicated_run(self.runtime, self.receipt)
        self.torch.cuda.get_device_properties.assert_not_called()
        self.assert_lock_released()

    def test_cuda_uuid_mismatch_refuses_and_saves_failure(self):
        self.cuda_identity.return_value = {"device_uuid": OTHER_UUID[4:], "pci_bus_id": "0000:47:00.0"}
        with self.assertRaisesRegex(AdmissionError, "CUDA physical device UUID"):
            admit_dedicated_run(self.runtime, self.receipt)
        self.egl.assert_not_called()
        self.assertEqual(json.loads((self.output / "admission.json").read_text())["decision"], "REFUSED")
        self.assert_lock_released()

    def test_gpu_lock_collision_does_not_start_a_check(self):
        cache = self.root / ".cache"
        cache.mkdir()
        with (cache / f"dedicated-admission-{UUID}.lock").open("a+") as existing:
            fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(AdmissionError, "already being admitted"):
                admit_dedicated_run(self.runtime, self.receipt)
        self.snapshots.assert_not_called()


if __name__ == "__main__":
    unittest.main()
