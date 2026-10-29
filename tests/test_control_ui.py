"""Exercise admission gating and ownership without launching any process/GPU."""
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from vla_trial.control_ui import (
    CourseControls, DEFAULT_INSTRUCTION, DEFAULT_TASK, MULTISTEP_TASK, authorization_status,
    active_task, default_instruction, execution_busy, task_description, validate_instruction,
)
from vla_trial.live_ui import read_live_snapshot, read_submission_artifact, summarize_run


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs/tasks").mkdir(parents=True)
        (self.root / "outputs").mkdir()
        (self.root / "configs/tasks/minimal_sort_ja.json").write_text(json.dumps({"instruction": DEFAULT_INSTRUCTION}))
        self.controls = CourseControls(self.root)
        self.child = Mock(spec=subprocess.Popen)
        self.child.pid = 12345
        self.child.poll.return_value = None

    def authorize(self, **updates):
        site = {"interactive_enabled": True,
                "authorization_date_jst": datetime.now(timezone(timedelta(hours=9))).date().isoformat()}
        site.update(updates)
        (self.root / "configs/site.json").write_text(json.dumps(site))

    def launch(self, instruction=DEFAULT_INSTRUCTION):
        with patch("vla_trial.control_ui.shutil.which", side_effect=lambda x: "/usr/bin/" + x), \
             patch("vla_trial.control_ui.subprocess.Popen", return_value=self.child) as popen, \
             patch("vla_trial.control_ui.process_start_ticks", return_value="98765"):
            message = self.controls.start(instruction)
        return message, popen

    def test_missing_or_disabled_authorization_never_launches(self):
        for settings in (None, {"interactive_enabled": False}, {"interactive_enabled": "true"},
                         {"authorization_date_jst": "2000-01-01"}):
            if settings is not None:
                self.authorize(**settings)
            with self.subTest(settings=settings):
                self.assertFalse(authorization_status(self.root)[0])
                self.assertFalse(self.controls.availability()["can_start"])
                _, popen = self.launch()
                popen.assert_not_called()

    def test_malformed_private_config_fails_closed(self):
        for content in ("not json", "[]", "null", " " * 65537):
            (self.root / "configs/site.json").write_text(content)
            self.assertFalse(authorization_status(self.root)[0])

    def test_exact_standard_task_gets_standard_scoring_notice(self):
        self.authorize()
        message, popen = self.launch()
        self.assertIn("按标准任务独立评分", message)
        self.assertTrue(self.controls.availability()["can_stop"])
        self.assertFalse(self.controls.availability()["can_start"])
        self.assertEqual(popen.call_count, 1)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--task") + 1], DEFAULT_TASK)

    def test_complex_active_task_drives_preset_and_child_contract(self):
        instruction = "Blue → A、yellow → B。青を A から押し出す。黄色を青の上に積み、両方を B へ。赤は動かさない。"
        (self.root / MULTISTEP_TASK).write_text(json.dumps({"instruction": instruction}))
        self.authorize(active_task=MULTISTEP_TASK)
        self.assertEqual(active_task(self.root)[0], MULTISTEP_TASK)
        self.assertEqual(default_instruction(self.root), instruction)
        self.assertIn("整体搬运", task_description(self.root)["info"])
        message, popen = self.launch(instruction)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--task") + 1], MULTISTEP_TASK)
        self.assertEqual(command[command.index("--instruction") + 1], instruction)
        self.assertIn("按标准任务独立评分", message)

    def test_custom_text_uses_active_contract_but_is_not_called_standard(self):
        (self.root / MULTISTEP_TASK).write_text(json.dumps({"instruction": "完整多步任务"}))
        self.authorize(active_task=MULTISTEP_TASK)
        message, popen = self.launch("只把蓝色放到 B")
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--task") + 1], MULTISTEP_TASK)
        self.assertIn("自定义任务", message)

    def test_unknown_or_traversing_active_task_never_falls_back_or_launches(self):
        for path in ("other.json", "../tasks/minimal_sort_ja.json", "/tmp/task.json", None,
                     "configs/tasks/minimal_sort_ja.json; touch unwanted"):
            self.authorize(active_task=path)
            self.assertFalse(authorization_status(self.root)[0])
            self.assertEqual(default_instruction(self.root), "")
            _, popen = self.launch()
            popen.assert_not_called()

    def test_missing_invalid_or_symlinked_complex_contract_never_launches(self):
        self.authorize(active_task=MULTISTEP_TASK)
        path = self.root / MULTISTEP_TASK
        for content in (None, "[]", "not json", '{"instruction": ""}'):
            if content is not None:
                path.write_text(content)
            self.assertFalse(authorization_status(self.root)[0])
            _, popen = self.launch()
            popen.assert_not_called()
        path.unlink()
        path.symlink_to(self.root / DEFAULT_TASK)
        self.assertFalse(authorization_status(self.root)[0])
        _, popen = self.launch()
        popen.assert_not_called()

    def test_action_summary_uses_each_complex_action_semantics(self):
        cases = (
            ({"action": "push_out", "object_id": "blue_block", "source_target_id": "left_tray"}, "蓝色方块推出托盘 A"),
            ({"action": "stack", "object_id": "yellow_block", "support_id": "blue_block"}, "黄色方块叠到蓝色方块上"),
            ({"action": "move_stack", "base_id": "blue_block", "top_id": "yellow_block", "target_id": "right_tray"}, "一起搬到托盘 B"),
        )
        for action, expected in cases:
            text = summarize_run({"phase": "moving", "stage": "full", "action": action})
            self.assertIn(expected, text)
            self.assertNotIn("独立检查通过", text)

    def test_ordered_multistep_result_is_not_confused_with_custom_unscored(self):
        status = {"phase": "completed", "stage": "full", "model_used": True,
                  "evaluation": {"evaluation_scope": "ordered_multistep_task", "whole_task_success": False}}
        self.assertIn("尚未满足全部标准任务要求", summarize_run(status))
        status["evaluation"]["whole_task_success"] = True
        self.assertIn("独立检查通过", summarize_run(status))

    def prepare_submission_pointer(self):
        self.authorize(active_task=MULTISTEP_TASK)
        task = {"instruction": "完整多步任务", "schema_version": 2, "task_id": "multistep_test"}
        (self.root / MULTISTEP_TASK).write_text(json.dumps(task))
        folder = self.root / "outputs/submission"
        folder.mkdir()
        video = folder / "unit-test-payload.mp4"
        # A file-path/hash unit fixture, never a rendered scene or video result.
        video.write_bytes(b"temporary unit test payload")
        manifest = {"video": video.name, "task_id": task["task_id"], "task_schema_version": 2,
                    "video_info": {"duration_s": 65}, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
        (folder / "timeline.json").write_text(json.dumps(manifest))
        marker = {"task": MULTISTEP_TASK, "video": "outputs/submission/" + video.name,
                  "manifest": "outputs/submission/timeline.json"}
        (self.root / "outputs/submission.json").write_text(json.dumps(marker))
        return video, manifest, marker

    def test_submission_download_requires_active_task_and_matching_file_hash(self):
        self.assertIsNone(read_submission_artifact(self.root))
        video, manifest, marker = self.prepare_submission_pointer()
        result = read_submission_artifact(self.root)
        self.assertEqual(result["video"], str(video.resolve()))
        self.assertIn("1 分 05 秒", result["label"])
        self.assertIn("1:05", result["download_label"])
        self.authorize(active_task=DEFAULT_TASK)
        self.assertIsNone(read_submission_artifact(self.root))
        self.authorize(active_task=MULTISTEP_TASK)
        video.write_bytes(b"changed payload is not the approved export")
        self.assertIsNone(read_submission_artifact(self.root))

    def test_submission_marker_cannot_expose_paths_outside_outputs(self):
        video, manifest, marker = self.prepare_submission_pointer()
        for invalid in ("/tmp/private.mp4", "../private.mp4", "configs/private.mp4"):
            marker["video"] = invalid
            (self.root / "outputs/submission.json").write_text(json.dumps(marker))
            self.assertIsNone(read_submission_artifact(self.root))

    def test_submission_manifest_must_match_task_schema_and_course_duration(self):
        video, manifest, marker = self.prepare_submission_pointer()
        path = video.parent / "timeline.json"
        for updates in ({"task_id": "old_task"}, {"task_schema_version": 1},
                        {"video_info": {"duration_s": 41.4}}, {"video_info": {"duration_s": 181}}):
            path.write_text(json.dumps({**manifest, **updates}))
            self.assertIsNone(read_submission_artifact(self.root))

    def test_instruction_is_one_literal_argument_not_shell_code(self):
        self.authorize()
        attack = "蓝色放进 B; $(touch /tmp/unwanted) `echo test`\n红色不要动。"
        message, popen = self.launch(attack)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--instruction") + 1], attack)
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(command[:6], ["/usr/bin/timeout", "--signal=TERM", "--kill-after=20s", "1500s", "/usr/bin/nice", "-n"])
        self.assertEqual(command[command.index("--stage") + 1], "full")
        self.assertIn("自定义任务", message)

    def test_empty_long_and_control_character_instructions_rejected(self):
        self.authorize()
        for value in ("", " ", "a" * 801, "task\x00x", None, 3):
            with self.subTest(value=repr(value)[:30]):
                with self.assertRaises(ValueError):
                    validate_instruction(value)
                _, popen = self.launch(value)
                popen.assert_not_called()

    def test_shared_manual_run_lock_blocks_ui_execution(self):
        self.authorize()
        path = self.root / "outputs/course-execution.lock"
        with path.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(execution_busy(self.root))
            self.assertFalse(self.controls.availability()["can_start"])
            _, popen = self.launch()
            popen.assert_not_called()
        self.assertFalse(execution_busy(self.root))

    def test_second_click_does_not_launch_another_child(self):
        self.authorize()
        self.launch()
        _, popen = self.launch()
        popen.assert_not_called()

    def test_missing_timeout_or_spawn_failure_is_not_retried(self):
        self.authorize()
        with patch("vla_trial.control_ui.shutil.which", return_value=None), \
             patch("vla_trial.control_ui.subprocess.Popen") as popen:
            self.controls.start(DEFAULT_INSTRUCTION)
            popen.assert_not_called()
        with patch("vla_trial.control_ui.shutil.which", return_value="/usr/bin/tool"), \
             patch("vla_trial.control_ui.subprocess.Popen", side_effect=OSError) as popen:
            message = self.controls.start(DEFAULT_INSTRUCTION)
            self.assertEqual(popen.call_count, 1)
            self.assertIn("没有自动重试", message)

    def test_stop_does_not_adopt_an_external_manual_run(self):
        with patch("vla_trial.control_ui.os.killpg") as kill:
            self.controls.stop()
            kill.assert_not_called()

    def test_stop_rejects_reused_pid_and_wrong_process_group(self):
        self.authorize()
        self.launch()
        with patch("vla_trial.control_ui.process_start_ticks", return_value="different"), \
             patch("vla_trial.control_ui.os.killpg") as kill:
            self.controls.stop()
            kill.assert_not_called()
        with patch("vla_trial.control_ui.process_start_ticks", return_value="98765"), \
             patch("vla_trial.control_ui.os.getpgid", return_value=999), \
             patch("vla_trial.control_ui.os.killpg") as kill:
            self.controls.stop()
            kill.assert_not_called()

    def test_stop_signals_only_verified_owned_session(self):
        self.authorize()
        self.launch()
        with patch("vla_trial.control_ui.process_start_ticks", return_value="98765"), \
             patch("vla_trial.control_ui.os.getpgid", return_value=12345), \
             patch("vla_trial.control_ui.os.killpg") as kill:
            self.controls.stop()
            kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_private_exception_paths_are_not_rendered(self):
        run = self.root / "outputs/test"
        run.mkdir()
        (self.root / "outputs/live.json").write_text(json.dumps({"run_dir": "outputs/test"}))
        (run / "status.json").write_text(json.dumps({"phase": "failed", "error": "Failed /private/model/path"}))
        snapshot = read_live_snapshot(self.root)
        self.assertNotIn("/private", json.dumps(snapshot["status"]))
        self.assertEqual(snapshot["status"]["phase"], "failed")


if __name__ == "__main__":
    unittest.main()
