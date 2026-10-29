"""Independent scoring tests use only declared goals and synthetic telemetry."""
import copy
import json
import math
from pathlib import Path
import unittest

from vla_trial.evaluation import ContractError, TaskContract, evaluate_task, load_contract


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract(Path(__file__).resolve().parents[1] / "configs/tasks/minimal_sort_ja.json")
        self.initial = {"blue_block": [0.42, 0.10, 0.02], "yellow_block": [0.56, 0.10, 0.02], "red_block": [0.42, -0.10, 0.02], "green_block": [0.56, -0.10, 0.02]}
        settled = {**copy.deepcopy(self.initial), "blue_block": [0.5, 0.3, 0.03], "yellow_block": [0.5, -0.3, 0.03]}
        self.samples = [{"elapsed_s": 0.0, "positions": copy.deepcopy(self.initial)}]
        self.samples += [{"elapsed_s": round(i / 10, 2), "positions": copy.deepcopy(settled)} for i in range(1, 9)]

    def score(self):
        return evaluate_task(self.contract, self.initial, self.samples)

    def codes(self, result):
        return {v["code"] for v in result["violations"]}

    def test_complete_contract_and_stable_placements_pass(self):
        result = self.score()
        self.assertTrue(result["success"])
        self.assertTrue(result["whole_task_success"])
        self.assertEqual(result["scoring_source"], "predeclared_contract_not_model_output")
        json.dumps(result, allow_nan=False)

    def test_wrong_direction_fails_even_if_both_objects_are_in_a_target(self):
        for sample in self.samples[1:]:
            sample["positions"]["blue_block"][1] = -0.3
            sample["positions"]["yellow_block"][1] = 0.3
        self.assertFalse(self.score()["success"])
        self.assertIn("goal_not_reached", self.codes(self.score()))

    def test_hovering_above_right_xy_fails(self):
        for sample in self.samples[1:]:
            sample["positions"]["blue_block"][2] = 0.10
        self.assertIn("goal_not_reached", self.codes(self.score()))

    def test_entering_only_in_last_frame_fails(self):
        for sample in self.samples[1:-1]:
            sample["positions"]["blue_block"] = self.initial["blue_block"][:]
        self.assertIn("goal_not_held_for_stability_window", self.codes(self.score()))

    def test_moving_inside_target_is_not_stable(self):
        for i, sample in enumerate(self.samples[1:]):
            sample["positions"]["blue_block"][0] = 0.48 if i % 2 else 0.52
        self.assertIn("object_not_stable", self.codes(self.score()))

    def test_protected_excursion_then_return_fails(self):
        self.samples[1]["positions"]["red_block"][0] += 0.03
        result = self.score()
        self.assertIn("protected_object_moved", self.codes(result))
        self.assertAlmostEqual(result["protected_results"]["red_block"]["max_displacement_m"], 0.03)

    def test_protected_motion_in_z_also_fails(self):
        self.samples[2]["positions"]["red_block"][2] += 0.03
        self.assertIn("protected_object_moved", self.codes(self.score()))

    def test_green_can_be_omitted_or_moved(self):
        self.initial.pop("green_block")
        for sample in self.samples:
            sample["positions"].pop("green_block")
        self.assertTrue(self.score()["success"])

    def test_missing_required_object_fails(self):
        for object_id in ("blue_block", "yellow_block", "red_block"):
            samples = copy.deepcopy(self.samples)
            samples[3]["positions"].pop(object_id)
            result = evaluate_task(self.contract, self.initial, samples)
            self.assertFalse(result["success"])
            self.assertIn("invalid_telemetry", self.codes(result))

    def test_nonfinite_and_wrong_coordinate_types_fail(self):
        for value in (math.nan, math.inf, -math.inf, "0.5", True):
            samples = copy.deepcopy(self.samples)
            samples[2]["positions"]["blue_block"][0] = value
            with self.subTest(value=value):
                result = evaluate_task(self.contract, self.initial, samples)
                self.assertFalse(result["success"])
                self.assertIn("invalid_telemetry", self.codes(result))
                json.dumps(result, allow_nan=False)

    def test_nonfinite_initial_position_fails(self):
        self.initial["red_block"][0] = math.nan
        self.assertIn("invalid_telemetry", self.codes(self.score()))

    def test_insufficient_duration_fails(self):
        self.samples = self.samples[:4]
        self.assertIn("insufficient_stability_evidence", self.codes(self.score()))

    def test_empty_or_wrong_sample_container_fails(self):
        for value in ([], None, {}, "fake"):
            result = evaluate_task(self.contract, self.initial, value)
            self.assertFalse(result["success"])

    def test_timestamp_duplicate_reverse_nonfinite_negative_fail(self):
        for time_value in (self.samples[1]["elapsed_s"], -0.1, math.nan, math.inf, True):
            samples = copy.deepcopy(self.samples)
            samples[2]["elapsed_s"] = time_value
            with self.subTest(time_value=time_value):
                result = evaluate_task(self.contract, self.initial, samples)
                self.assertIn("invalid_telemetry", self.codes(result))

    def test_missing_initial_sample_fails(self):
        self.samples.pop(0)
        self.assertIn("missing_initial_sample", self.codes(self.score()))

    def test_large_telemetry_gap_fails(self):
        self.samples = self.samples[:2] + self.samples[5:]
        self.assertIn("telemetry_gap", self.codes(self.score()))

    def test_does_not_mutate_supplied_data(self):
        contract = self.contract.to_dict()
        before = copy.deepcopy((contract, self.initial, self.samples))
        evaluate_task(contract, self.initial, self.samples)
        self.assertEqual((contract, self.initial, self.samples), before)

    def test_model_plan_cannot_serve_as_goal_contract(self):
        with self.assertRaises(ContractError):
            evaluate_task({"actions": [{"action": "done"}]}, self.initial, self.samples)

    def test_invalid_contract_rejected(self):
        cases = []
        c = self.contract.to_dict(); c["targets"]["left_tray"]["center"][0] = math.nan; cases.append(c)
        c = self.contract.to_dict(); c["targets"]["left_tray"]["center_tolerance"][2] = -1; cases.append(c)
        c = self.contract.to_dict(); c["expected_placements"]["red_block"] = "left_tray"; cases.append(c)
        c = self.contract.to_dict(); c["expected_placements"]["blue_block"] = "missing"; cases.append(c)
        c = self.contract.to_dict(); c["success_criteria"]["min_stable_samples"] = True; cases.append(c)
        for contract in cases:
            with self.subTest(contract=contract), self.assertRaises(ContractError):
                TaskContract.from_mapping(contract)


if __name__ == "__main__":
    unittest.main()
