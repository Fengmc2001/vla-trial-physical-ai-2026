"""CPU-only action-language and independent multistep-scoring tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from vla_trial.evaluation import ContractError
from vla_trial.evaluation_multistep import evaluate_multistep_task, load_multistep_contract, validate_multistep_contract
from vla_trial.qwen import build_planner_prompt
from vla_trial.schema import PlanValidationError, parse_plan


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = [
    {"action": "pick_place", "object_id": "blue_block", "target_id": "left_tray"},
    {"action": "pick_place", "object_id": "yellow_block", "target_id": "right_tray"},
    {"action": "push_out", "object_id": "blue_block", "source_target_id": "left_tray"},
    {"action": "stack", "object_id": "yellow_block", "support_id": "blue_block"},
    {"action": "move_stack", "base_id": "blue_block", "top_id": "yellow_block", "target_id": "right_tray"},
    {"action": "done"},
]


class MultiActionSchemaTests(unittest.TestCase):
    def parse(self, actions, **options):
        return parse_plan(json.dumps({"actions": actions}), protected_ids=["red_block"], max_actions=12,
                          allow_repeated_objects=True, **options)

    def test_full_mixed_task_roundtrip_has_exact_primitive_arguments(self):
        self.assertEqual(self.parse(ACTIONS).to_dict(), {"actions": ACTIONS})
        self.assertEqual(self.parse(ACTIONS).actions[4].base_id, "blue_block")

    def test_old_default_keeps_single_move_and_original_vocabulary(self):
        for actions in ([ACTIONS[0], ACTIONS[0], ACTIONS[-1]], [ACTIONS[2], ACTIONS[-1]]):
            with self.subTest(actions=actions), self.assertRaises(PlanValidationError):
                parse_plan(json.dumps({"actions": actions}))

    def test_red_is_rejected_in_every_manipulated_or_support_argument(self):
        for index, field in ((0, "object_id"), (2, "object_id"), (3, "object_id"),
                             (3, "support_id"), (4, "base_id"), (4, "top_id")):
            actions = copy.deepcopy(ACTIONS)
            actions[index][field] = "red_block"
            with self.subTest(index=index, field=field), self.assertRaisesRegex(PlanValidationError, "Protected object"):
                self.parse(actions)

    def test_self_stack_unknown_ids_extra_arguments_and_wrong_source_are_rejected(self):
        variants = [
            {"action": "stack", "object_id": "blue_block", "support_id": "blue_block"},
            {"action": "move_stack", "base_id": "blue_block", "top_id": "blue_block", "target_id": "right_tray"},
            {"action": "push_out", "object_id": "unknown", "source_target_id": "left_tray"},
            {"action": "push_out", "object_id": "blue_block", "source_target_id": "blue_block"},
            {"action": "stack", "object_id": "yellow_block", "support_id": "blue_block", "position": [0, 0, 0]},
        ]
        for action in variants:
            with self.subTest(action=action), self.assertRaises(PlanValidationError):
                self.parse([action, {"action": "done"}])

    def test_multi_json_mode_flag_and_twelve_action_limit_fail_closed(self):
        with self.assertRaises(PlanValidationError):
            parse_plan('{"actions":[{"action":"done"}]}{"actions":[{"action":"done"}]}', allow_repeated_objects=True)
        with self.assertRaises(PlanValidationError):
            parse_plan('{"actions":[{"action":"done"}]}', allow_repeated_objects="true")
        with self.assertRaises(PlanValidationError):
            parse_plan('{"actions":[{"action":"done"}]}', max_actions=13)

    def test_prompt_describes_tools_without_inserting_an_answer_plan(self):
        manifest = {"objects": [{"object_id": "blue_block"}], "targets": [], "protected_ids": ["red_block"]}
        standard = build_planner_prompt(manifest, 6)
        extended = build_planner_prompt(manifest, 12, multistep=True)
        self.assertNotIn('"action":"push_out"', standard)
        for kind in ("pick_place", "push_out", "stack", "move_stack"):
            self.assertIn(f'"action":"{kind}"', extended)
        self.assertIn("one actions list", extended)
        self.assertNotIn('"actions": [', extended)
        self.assertNotIn('"object_id":"blue_block","target_id":"left_tray"', extended)


def fixture(reverse_first_two=False):
    task = load_multistep_contract(ROOT / "configs/tasks/multistep_stack_mixed.json")
    ground = task["success_criteria"]["ground_center_z_m"]
    height = task["success_criteria"]["block_height_m"]
    initial = {"red_block": [0.42, -0.10, ground], "blue_block": [0.42, 0.10, ground], "yellow_block": [0.56, 0.10, ground]}
    blue_a, yellow_b = list(task["targets"]["left_tray"]["center"]), list(task["targets"]["right_tray"]["center"])
    blue_out, yellow_stack = [0.50, 0.15, ground], [0.50, 0.15, ground + height]
    base_b, top_b = list(yellow_b), [yellow_b[0], yellow_b[1], yellow_b[2] + height]
    def interpolate(first, last, phase):
        fraction = min(1.0, max(0.0, phase / 1.4))
        return [a + (b - a) * fraction for a, b in zip(first, last)]
    samples = []
    for index in range(101):
        t = round(index / 10, 1)
        b_start, y_start = (2.0, 0.0) if reverse_first_two else (0.0, 2.0)
        blue = interpolate(initial["blue_block"], blue_a, t - b_start)
        yellow = interpolate(initial["yellow_block"], yellow_b, t - y_start)
        if t >= 4:
            blue = interpolate(blue_a, blue_out, t - 4)
        if t >= 6:
            yellow = interpolate(yellow_b, yellow_stack, t - 6)
        if t >= 8:
            blue = interpolate(blue_out, base_b, t - 8)
            yellow = interpolate(yellow_stack, top_b, t - 8)
        samples.append({"elapsed_s": t, "positions": {"red_block": list(initial["red_block"]), "blue_block": blue, "yellow_block": yellow}})
    actions = copy.deepcopy(ACTIONS[:-1])
    if reverse_first_two:
        actions[:2] = reversed(actions[:2])
    milestones = [{"action": action, "started_s": index * 2.0, "ended_s": (index + 1) * 2.0, "completed": True}
                  for index, action in enumerate(actions)]
    return task, initial, samples, milestones


class MultiStepEvaluationTests(unittest.TestCase):
    def test_all_observed_milestones_pass_in_either_initial_placement_order(self):
        for reverse in (False, True):
            result = evaluate_multistep_task(*fixture(reverse))
            self.assertTrue(result["success"], result["violations"])
            self.assertTrue(result["whole_task_success"])
            self.assertEqual(result["action_count"], 5)
            json.dumps(result, allow_nan=False)

    def test_final_stack_alone_cannot_replace_initial_sort_and_push(self):
        task, initial, samples, milestones = fixture()
        final = copy.deepcopy(samples[-1]["positions"])
        for sample in samples:
            sample["positions"] = copy.deepcopy(final)
        result = evaluate_multistep_task(task, final, samples, milestones)
        self.assertFalse(result["success"])
        self.assertFalse(result["milestone_results"]["initial_sort"]["passed"])

    def test_grasping_then_putting_outside_does_not_count_as_push(self):
        task, initial, samples, milestones = fixture()
        samples[45]["positions"]["blue_block"][2] = 0.12
        result = evaluate_multistep_task(task, initial, samples, milestones)
        self.assertFalse(result["success"])
        self.assertFalse(result["milestone_results"]["push_blue_outside_a"]["passed"])
        self.assertTrue(result["milestone_results"]["intact_stack_in_b"]["passed"])

    def test_unstacking_during_transport_fails_even_if_restacked_at_end(self):
        task, initial, samples, milestones = fixture()
        samples[85]["positions"]["yellow_block"][0] += 0.05
        result = evaluate_multistep_task(task, initial, samples, milestones)
        self.assertFalse(result["success"])
        self.assertFalse(result["milestone_results"]["stack_preserved_during_transport"]["passed"])
        self.assertTrue(result["milestone_results"]["intact_stack_in_b"]["passed"])

    def test_red_move_and_return_is_still_a_violation(self):
        task, initial, samples, milestones = fixture()
        samples[52]["positions"]["red_block"][1] += 0.02
        result = evaluate_multistep_task(task, initial, samples, milestones)
        self.assertFalse(result["success"])
        self.assertFalse(result["protected_results"]["red_block"]["passed"])

    def test_declared_success_or_model_like_plan_is_not_execution_evidence(self):
        task, initial, samples, milestones = fixture()
        result = evaluate_multistep_task(task, initial, samples, ACTIONS[:-1])
        self.assertFalse(result["success"])
        self.assertFalse(result["evaluation_complete"])
        milestones[2]["completed"] = False
        self.assertFalse(evaluate_multistep_task(task, initial, samples, milestones)["success"])

    def test_wrong_executed_order_or_target_fails_despite_good_geometry(self):
        task, initial, samples, milestones = fixture()
        milestones[2]["action"]["source_target_id"] = "right_tray"
        result = evaluate_multistep_task(task, initial, samples, milestones)
        self.assertFalse(result["success"])
        self.assertIn("wrong_executed_action_order_or_arguments", [x["code"] for x in result["violations"]])

    def test_missing_samples_time_gap_and_nonfinite_position_fail_closed(self):
        for change in ("gap", "nan", "missing"):
            task, initial, samples, milestones = fixture()
            if change == "gap":
                del samples[41:43]
            elif change == "nan":
                samples[20]["positions"]["blue_block"][0] = float("nan")
            else:
                del samples[20]["positions"]["yellow_block"]
            with self.subTest(change=change):
                result = evaluate_multistep_task(task, initial, samples, milestones)
                self.assertFalse(result["success"])
                self.assertFalse(result["evaluation_complete"])

    def test_final_hover_and_unstable_stack_fail(self):
        for change in ("hover", "unstable"):
            task, initial, samples, milestones = fixture()
            if change == "hover":
                for sample in samples[95:]:
                    sample["positions"]["blue_block"][2] += 0.05
                    sample["positions"]["yellow_block"][2] += 0.05
            else:
                samples[97]["positions"]["yellow_block"][0] += 0.008
            result = evaluate_multistep_task(task, initial, samples, milestones)
            self.assertFalse(result["success"])
            self.assertFalse(result["milestone_results"]["intact_stack_in_b"]["passed"])

    def test_contract_rejects_protected_support_and_invalid_geometry(self):
        task, *_ = fixture()
        task["protected_ids"].append("blue_block")
        with self.assertRaises(ContractError):
            validate_multistep_contract(task)
        task, *_ = fixture()
        task["success_criteria"]["push_max_center_rise_m"] = float("inf")
        with self.assertRaises(ContractError):
            validate_multistep_contract(task)


if __name__ == "__main__":
    unittest.main()
