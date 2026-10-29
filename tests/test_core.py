"""CPU-only tests. Never load model weights, Genesis, torch or CUDA."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vla_trial.runtime import Runtime, RuntimeConfig
from vla_trial.safety import (
    AdmissionError, GraphicsBindingUnverifiedError, RuntimeDisabledError,
    _assert_gpu_idle, authorize_runtime, validate_receipt,
)
from vla_trial.schema import PlanValidationError, parse_plan, validate_instruction
from vla_trial.simulation import FrankaSimulation


UUID = "GPU-12345678-1234-1234-1234-123456789abc"
VALID_PLAN = '{"actions":[{"action":"pick_place","object_id":"blue_block","target_id":"left_tray"},{"action":"done"}]}'


class PlanTests(unittest.TestCase):
    def test_parse_valid_plan(self):
        plan = parse_plan(VALID_PLAN)
        self.assertEqual(plan.actions[0].object_id, "blue_block")
        self.assertEqual(plan.to_dict(), json.loads(VALID_PLAN))

    def test_reject_protected_object(self):
        with self.assertRaises(PlanValidationError):
            parse_plan(VALID_PLAN, protected_ids=["blue_block"])

    def test_reject_unknown_protected_object(self):
        with self.assertRaises(PlanValidationError):
            parse_plan(VALID_PLAN, protected_ids=["unrecognized"])

    def test_reject_executable_and_extra_fields(self):
        for value in (
            '{"actions":[{"action":"exec","command":"anything"},{"action":"done"}]}',
            VALID_PLAN.replace('"target_id":"left_tray"', '"target_id":"left_tray","code":"anything"'),
            '{"actions":[{"action":"done"}],"code":"anything"}',
            '__import__("os")',
            "```json\n" + VALID_PLAN + "\n```",
        ):
            with self.subTest(value=value), self.assertRaises(PlanValidationError):
                parse_plan(value)

    def test_reject_unknown_objects_targets_and_duplicate_keys(self):
        for value in (
            VALID_PLAN.replace("blue_block", "remote_path"),
            VALID_PLAN.replace("left_tray", "other_location"),
            '{"actions":[],"actions":[{"action":"done"}]}',
            '{"actions":[{"action":"done","action":"done"}]}',
        ):
            with self.subTest(value=value), self.assertRaises(PlanValidationError):
                parse_plan(value)

    def test_reject_missing_early_done_duplicate_object_and_action_overflow(self):
        pick = json.loads(VALID_PLAN)["actions"][0]
        for actions in ([pick], [{"action": "done"}, pick], [pick, pick, {"action": "done"}], []):
            with self.subTest(actions=actions), self.assertRaises(PlanValidationError):
                parse_plan(json.dumps({"actions": actions}))
        with self.assertRaises(PlanValidationError):
            parse_plan(VALID_PLAN, max_actions=1)

    def test_instruction_limits(self):
        self.assertEqual(validate_instruction("  青いブロックを移動  "), "青いブロックを移動")
        for value in ("", " ", "x" * 4001, "bad\x00text", None):
            with self.subTest(value=str(value)[:15]), self.assertRaises(PlanValidationError):
                validate_instruction(value)


class RuntimeTests(unittest.TestCase):
    def test_config_roundtrip_preserves_additional_ui_section(self):
        config = RuntimeConfig.from_mapping({"ui": {"port": 7865}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config.save(path)
            result = RuntimeConfig.load(path)
        self.assertEqual(config.to_dict(), result.to_dict())
        self.assertFalse(result.enabled)

    def test_config_rejects_strings_for_enabled_and_unbounded_limits(self):
        for value in ({"runtime": {"enabled": "false"}}, {"simulation": {"max_video_frames": 9999999}}, {"runtime": {"max_actions": True}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RuntimeConfig.from_mapping(value)

    def test_safe_mode_run_does_not_query_gpu_or_write_outputs(self):
        with tempfile.TemporaryDirectory() as directory, patch("vla_trial.safety.subprocess.run") as query:
            runtime = Runtime({}, directory)
            self.assertFalse(runtime.status()["simulation_started"])
            with self.assertRaises(RuntimeDisabledError):
                runtime.run("Move the blue block.")
            query.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_empty_model_path_is_not_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(Runtime({}, directory).preflight()["model_directory_exists"])

    def test_safe_imports_in_fresh_interpreter_have_no_ml_modules(self):
        code = (
            "import sys; import vla_trial.runtime, vla_trial.safety, vla_trial.schema, vla_trial.qwen, vla_trial.simulation; "
            "assert not any(m in sys.modules for m in ('torch','transformers','genesis','numpy','PIL')); "
            "print('no ML imports')"
        )
        result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
        self.assertIn("no ML imports", result.stdout)


class FutureRuntimeControlFlowTests(unittest.TestCase):
    """Model, simulator and admission are mocks, never actual hardware paths."""

    def fixtures(self, simulator, planner):
        simulator.frame.return_value = "CPU test sentinel, not an image"
        simulator.object_positions.return_value = {"red_block": [0, 0, 0.02], "blue_block": [0.4, 0, 0.02]}
        planner.plan.return_value = parse_plan('{"actions":[{"action":"done"}]}')

    def read_audit(self, directory):
        logs = list(Path(directory).glob("outputs/*/run.json"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(list(Path(directory).glob("outputs/*/*.tmp")), [])
        return json.loads(logs[0].read_text(encoding="utf-8"))

    def test_inherits_scene_protection_when_argument_omitted(self):
        with tempfile.TemporaryDirectory() as directory, patch("vla_trial.runtime.authorize_runtime", return_value=object()), patch("vla_trial.qwen.QwenPlanner") as planner_type, patch("vla_trial.simulation.FrankaSimulation") as simulation_type:
            self.fixtures(simulation_type.return_value, planner_type.return_value)
            result = Runtime({"scene": {"protected_ids": ["blue_block"]}}, directory).run("do nothing")
            self.assertEqual(planner_type.return_value.plan.call_args.kwargs["protected_ids"], ("blue_block",))
            audit = self.read_audit(directory)
            self.assertEqual(audit["protected_ids"], ["blue_block"])
            self.assertIsNone(audit["whole_task_success"])
            self.assertFalse(audit["execution_failed"])
            self.assertEqual(result.status, "model_reported_done_unverified")

    def test_explicit_empty_protection_overrides_scene(self):
        with tempfile.TemporaryDirectory() as directory, patch("vla_trial.runtime.authorize_runtime", return_value=object()), patch("vla_trial.qwen.QwenPlanner") as planner_type, patch("vla_trial.simulation.FrankaSimulation") as simulation_type:
            self.fixtures(simulation_type.return_value, planner_type.return_value)
            Runtime({}, directory).run("do nothing", protected_ids=[])
            self.assertEqual(planner_type.return_value.plan.call_args.kwargs["protected_ids"], ())

    def test_failure_audit_and_independent_cleanup_preserve_primary_error(self):
        with tempfile.TemporaryDirectory() as directory, patch("vla_trial.runtime.authorize_runtime", return_value=object()), patch("vla_trial.qwen.QwenPlanner") as planner_type, patch("vla_trial.simulation.FrankaSimulation") as simulation_type:
            simulator, planner = simulation_type.return_value, planner_type.return_value
            self.fixtures(simulator, planner)
            planner.plan.side_effect = RuntimeError("inference failed")
            simulator.close.side_effect = OSError("video close failed")
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                Runtime({}, directory).run("do nothing")
            simulator.close.assert_called_once()
            planner.close.assert_called_once()
            audit = self.read_audit(directory)
            self.assertEqual(audit["status"], "failed")
            self.assertIsNone(audit["whole_task_success"])
            self.assertTrue(audit["execution_failed"])
            self.assertEqual(audit["error"]["message"], "inference failed")
            self.assertEqual(audit["cleanup_errors"][0]["message"], "video close failed")

    def test_build_failure_also_leaves_failed_audit(self):
        with tempfile.TemporaryDirectory() as directory, patch("vla_trial.runtime.authorize_runtime", return_value=object()), patch("vla_trial.qwen.QwenPlanner") as planner_type, patch("vla_trial.simulation.FrankaSimulation") as simulation_type:
            simulation_type.return_value.build.side_effect = RuntimeError("build failed")
            with self.assertRaisesRegex(RuntimeError, "build failed"):
                Runtime({}, directory).run("do nothing")
            audit = self.read_audit(directory)
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(audit["plans"], [])
            planner_type.return_value.close.assert_called_once()

    def test_cleanup_failure_after_normal_run_is_reported_failed(self):
        with tempfile.TemporaryDirectory() as directory, patch("vla_trial.runtime.authorize_runtime", return_value=object()), patch("vla_trial.qwen.QwenPlanner") as planner_type, patch("vla_trial.simulation.FrankaSimulation") as simulation_type:
            self.fixtures(simulation_type.return_value, planner_type.return_value)
            planner_type.return_value.close.side_effect = RuntimeError("planner cleanup failed")
            with self.assertRaisesRegex(RuntimeError, "planner cleanup failed"):
                Runtime({}, directory).run("do nothing")
            self.assertEqual(self.read_audit(directory)["status"], "failed")


class FutureMotionPlanningTests(unittest.TestCase):
    """Only fixed mock data; no Genesis, NumPy, torch or robot is created."""

    def simulation(self, valid):
        simulation = FrankaSimulation({}, object())
        simulation._np = SimpleNamespace(array=lambda value: value)
        simulation.robot = Mock()
        simulation.robot.inverse_kinematics.return_value = [0] * 9
        simulation.robot.plan_path.return_value = ([[0] * 9], valid)
        simulation.hand = object()
        simulation.arm_dofs = list(range(7))
        simulation._check_limits = Mock()
        simulation._step = Mock()
        simulation._assert_hand_near = Mock()
        return simulation

    def test_failed_validity_mask_rejects_nonempty_path_without_motion(self):
        simulation = self.simulation(False)
        with self.assertRaisesRegex(RuntimeError, "No valid collision-checked path"):
            simulation._move_hand((0.5, 0.0, 0.3))
        simulation.robot.control_dofs_position.assert_not_called()
        simulation._step.assert_not_called()
        options = simulation.robot.plan_path.call_args.kwargs
        self.assertTrue(options["return_valid_mask"])
        self.assertEqual(options["max_retry"], 0)
        self.assertGreater(options["timeout"], 0)
        self.assertLessEqual(options["timeout"], 60)
        self.assertFalse(options["ignore_collision"])

    def test_true_scalar_tensor_mask_allows_prepared_waypoint_control(self):
        simulation = self.simulation(SimpleNamespace(item=lambda: True))
        simulation._move_hand((0.5, 0.0, 0.3))
        simulation.robot.control_dofs_position.assert_called_once()
        self.assertEqual(simulation._step.call_count, 2)

    def test_ambiguous_validity_value_is_rejected(self):
        for valid in ([False], 1, None):
            simulation = self.simulation(valid)
            with self.subTest(valid=valid), self.assertRaises(RuntimeError):
                simulation._move_hand((0.5, 0.0, 0.3))
            simulation.robot.control_dofs_position.assert_not_called()

    def test_scene_teardown_attempted_when_video_close_fails(self):
        simulation = FrankaSimulation({}, object())
        writer, scene = Mock(), Mock()
        writer.close.side_effect = RuntimeError("encoder close failed")
        simulation.writer, simulation.scene = writer, scene
        with self.assertRaisesRegex(RuntimeError, "encoder close failed"):
            simulation.close()
        scene.destroy.assert_called_once()
        self.assertIsNone(simulation.writer)
        self.assertIsNone(simulation.scene)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = datetime.now(timezone.utc)
        self.path = Path(self.temp.name) / "receipt.json"
        self.receipt = {
            "authorized": True, "purpose": "vla-trial", "host": "gpu-test",
            "gpu_uuid": UUID, "operator": "test-operator", "allocation_reference": "CPU-test-only",
            "issued_at": (self.now - timedelta(minutes=1)).isoformat(),
            "expires_at": (self.now + timedelta(minutes=30)).isoformat(),
        }
        self.config = {"enabled": True, "gpu_uuid": UUID, "approval_receipt": str(self.path)}
        self.clean_environment = patch.dict(os.environ, {}, clear=True)
        self.clean_environment.start()
        self.addCleanup(self.clean_environment.stop)

    def write(self):
        self.path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def test_valid_receipt_only_validates_metadata(self):
        self.write()
        with patch("vla_trial.safety.subprocess.run") as query:
            result = validate_receipt(self.config, host="gpu-test", now=self.now)
            self.assertEqual(result["gpu_uuid"], UUID)
            query.assert_not_called()

    def test_reject_missing_receipt_uuid_and_wrong_host(self):
        self.write()
        for change in ({"gpu_uuid": "0"}, {"approval_receipt": ""}, {"approval_receipt": "relative.json"}):
            with self.subTest(change=change), self.assertRaises(AdmissionError):
                validate_receipt({**self.config, **change}, host="gpu-test", now=self.now)
        with self.assertRaises(AdmissionError):
            validate_receipt(self.config, host="other-host", now=self.now)

    def test_reject_expired_stale_and_naive_timestamps(self):
        for changes in (
            {"expires_at": (self.now - timedelta(seconds=1)).isoformat()},
            {"issued_at": (self.now - timedelta(minutes=6)).isoformat()},
            {"issued_at": (self.now + timedelta(seconds=1)).isoformat()},
            {"expires_at": (self.now + timedelta(hours=2)).isoformat()},
            {"issued_at": "2026-10-15T12:00:00"},
            {"authorized": "true"},
        ):
            original = dict(self.receipt)
            self.receipt.update(changes)
            self.write()
            with self.subTest(changes=changes), self.assertRaises(AdmissionError):
                validate_receipt(self.config, host="gpu-test", now=self.now)
            self.receipt = original

    def test_reject_gpu200_and_slurm(self):
        self.write()
        with self.assertRaises(AdmissionError):
            validate_receipt(self.config, host="GPU200", now=self.now)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}), self.assertRaises(AdmissionError):
            validate_receipt(self.config, host="gpu-test", now=self.now)

    def test_graphics_lock_precedes_gpu_queries_even_with_valid_receipt(self):
        self.write()
        with patch("vla_trial.safety.socket.gethostname", return_value="gpu-test"), patch("vla_trial.safety.subprocess.run") as query:
            with self.assertRaises(GraphicsBindingUnverifiedError):
                authorize_runtime(self.config)
            query.assert_not_called()


class GPUQueryParsingTests(unittest.TestCase):
    """These test fixed XML strings; subprocess is always mocked."""
    def xml(self, processes="", used="0 MiB", utilization="0 %", uuid=UUID):
        return f"<nvidia_smi_log><gpu><uuid>{uuid}</uuid><processes>{processes}</processes><fb_memory_usage><used>{used}</used></fb_memory_usage><utilization><gpu_util>{utilization}</gpu_util></utilization></gpu></nvidia_smi_log>"

    def test_idle_fixture(self):
        with patch("vla_trial.safety.subprocess.run", return_value=subprocess.CompletedProcess([], 0, self.xml())):
            _assert_gpu_idle(UUID)

    def test_graphics_process_memory_utilization_unknown_and_wrong_uuid_deny(self):
        fixtures = [
            self.xml('<process_info><pid>1234</pid><type>G</type></process_info>'),
            self.xml(used="513 MiB"), self.xml(utilization="6 %"),
            self.xml(utilization="N/A"), self.xml(uuid="GPU-other"), "not XML",
            self.xml(processes="N/A"), self.xml(used="1 GiB"), self.xml(used="nan MiB"),
        ]
        for fixture in fixtures:
            with self.subTest(fixture=fixture), patch("vla_trial.safety.subprocess.run", return_value=subprocess.CompletedProcess([], 0, fixture)), self.assertRaises(AdmissionError):
                _assert_gpu_idle(UUID)


if __name__ == "__main__":
    unittest.main()
