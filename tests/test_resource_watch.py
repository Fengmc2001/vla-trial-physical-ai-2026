"""Resource guard tests use XML fixtures and never query a GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from vla_trial.resource_watch import ProcessResourceWatch, ResourceWatchError, own_gpu_usage, query_own_gpu_usage


def document(*devices):
    return "<nvidia_smi_log>" + "".join(devices) + "</nvidia_smi_log>"


def device(uuid, *processes):
    return f"<gpu><uuid>{uuid}</uuid><processes>{''.join(processes)}</processes></gpu>"


def process(pid, memory, kind="C"):
    return f"<process_info><pid>{pid}</pid><type>{kind}</type><used_memory>{memory}</used_memory></process_info>"


class ProcessXMLTests(unittest.TestCase):
    def test_includes_compute_graphics_and_combined_but_excludes_foreign_pids(self):
        xml = document(device("GPU-A", process(42, "512 MiB", "C+G"), process(99, "80000 MiB")),
                       device("GPU-B", process(42, "16 MiB", "G")),
                       device("GPU-C", process(42, "128 MiB", "C")))
        self.assertEqual(own_gpu_usage(xml, 42), {"GPU-A": 512.0, "GPU-B": 16.0, "GPU-C": 128.0})

    def test_duplicate_process_entries_are_not_double_counted(self):
        xml = document(device("GPU-A", process(42, "512 MiB", "C"), process(42, "500 MiB", "G")))
        self.assertEqual(own_gpu_usage(xml, 42), {"GPU-A": 512.0})

    def test_absent_pid_is_zero_not_foreign_memory(self):
        xml = document(device("GPU-A", process(99, "N/A", "G")))
        self.assertEqual(own_gpu_usage(xml, 42), {})

    def test_unavailable_own_memory_and_unexpected_units_fail_closed(self):
        for value in ("N/A", "", "1 GiB", "-1 MiB", "nan MiB"):
            with self.subTest(value=value), self.assertRaises(ResourceWatchError):
                own_gpu_usage(document(device("GPU-A", process(42, value))), 42)

    def test_incomplete_query_is_not_an_empty_success(self):
        for xml in ("invalid", "<other />", document(), "<nvidia_smi_log><gpu><uuid>GPU-A</uuid></gpu></nvidia_smi_log>", document(device("", process(42, "1 MiB")))):
            with self.subTest(xml=xml), self.assertRaises(ResourceWatchError):
                own_gpu_usage(xml, 42)

    def test_query_is_timeout_bounded_and_shell_free(self):
        response = subprocess.CompletedProcess([], 0, document(device("GPU-A", process(42, "4 MiB"))), "")
        with patch("vla_trial.resource_watch.subprocess.run", return_value=response) as run:
            self.assertEqual(query_own_gpu_usage(42), {"GPU-A": 4.0})
        run.assert_called_once_with(["nvidia-smi", "-q", "-x"], check=True, capture_output=True, text=True, timeout=30)

    def test_query_failure_does_not_echo_driver_stderr(self):
        with patch("vla_trial.resource_watch.subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi", stderr="private other-process data")):
            with self.assertRaises(ResourceWatchError) as caught:
                query_own_gpu_usage(42)
        self.assertNotIn("private", str(caught.exception))


class WatchLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "watch.json"

    def watch(self, **kwargs):
        return ProcessResourceWatch("GPU-A", 1000, self.output, wall_seconds=60, **kwargs)

    def test_constructor_is_inert_and_rejects_unbounded_or_fast_polling(self):
        with patch("vla_trial.resource_watch.query_own_gpu_usage") as query:
            self.watch()
        query.assert_not_called()
        for wall, interval in ((1801, 5), (0, 5), (60, 1), (60, 31)):
            with self.subTest(wall=wall, interval=interval), self.assertRaises(ValueError):
                ProcessResourceWatch("GPU-A", 1000, self.output, wall_seconds=wall, interval_seconds=interval)

    def test_start_close_records_only_own_usage_and_no_signals(self):
        with patch("vla_trial.resource_watch.query_own_gpu_usage", return_value={"GPU-A": 600.0}) as query, patch("vla_trial.resource_watch.os.kill") as kill:
            with self.watch() as watcher:
                self.assertEqual(watcher.record["peak_measured_mib"], 600.0)
        query.assert_called_once_with(os.getpid())
        kill.assert_not_called()
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved["status"], "closed")
        self.assertFalse(saved["hard_allocation_cap"])
        self.assertEqual(saved["samples"][0]["memory_by_gpu_mib"], {"GPU-A": 600.0})

    def test_initial_wrong_gpu_raises_without_signalling(self):
        with patch("vla_trial.resource_watch.query_own_gpu_usage", return_value={"GPU-A": 4.0, "GPU-B": 1.0}), patch("vla_trial.resource_watch.os.kill") as kill:
            with self.assertRaisesRegex(ResourceWatchError, "outside"):
                self.watch().start()
        kill.assert_not_called()
        self.assertEqual(json.loads(self.output.read_text())["status"], "failed")

    def test_background_budget_failure_saves_evidence_then_signals_only_self(self):
        watcher = self.watch()
        watcher._started = 1.0
        watcher.record["status"] = "watching"
        evidence_at_signal = []
        def inspect_signal(pid, sig):
            evidence_at_signal.append(json.loads(self.output.read_text()))
            self.assertEqual((pid, sig), (os.getpid(), signal.SIGTERM))
        with patch("vla_trial.resource_watch.time.monotonic", return_value=2.0), patch("vla_trial.resource_watch.query_own_gpu_usage", return_value={"GPU-A": 1001.0}), patch.object(watcher._stop, "wait", return_value=False), patch("vla_trial.resource_watch.os.kill", side_effect=inspect_signal) as kill:
            watcher._run()
        kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
        self.assertEqual(evidence_at_signal[0]["peak_measured_mib"], 1001.0)
        self.assertEqual(evidence_at_signal[0]["status"], "failed")
        watcher.close()
        self.assertEqual(json.loads(self.output.read_text())["status"], "failed")

    def test_query_failure_fails_closed(self):
        with patch("vla_trial.resource_watch.query_own_gpu_usage", side_effect=ResourceWatchError("unavailable")):
            with self.assertRaisesRegex(ResourceWatchError, "unavailable"):
                self.watch().start()
        self.assertEqual(json.loads(self.output.read_text())["status"], "failed")

    def test_monotonic_deadline_and_cross_process_reuse_are_rejected(self):
        watcher = self.watch()
        watcher._started = 1.0
        with patch("vla_trial.resource_watch.time.monotonic", return_value=61.0), patch("vla_trial.resource_watch.query_own_gpu_usage") as query:
            with self.assertRaisesRegex(ResourceWatchError, "wall-time"):
                watcher._check()
        query.assert_not_called()
        with patch("vla_trial.resource_watch.os.getpid", return_value=watcher.pid + 1):
            with self.assertRaisesRegex(ResourceWatchError, "across processes"):
                watcher._check()

    def test_peak_is_preserved_when_current_usage_falls(self):
        watcher = self.watch()
        watcher._started = 1.0
        with patch("vla_trial.resource_watch.time.monotonic", return_value=2.0), patch("vla_trial.resource_watch.query_own_gpu_usage", side_effect=[{"GPU-A": 900.0}, {"GPU-A": 100.0}]):
            watcher._check()
            watcher._check()
        self.assertEqual(watcher.record["peak_measured_mib"], 900.0)
        self.assertEqual(watcher.record["query_count"], 2)


if __name__ == "__main__":
    unittest.main()
