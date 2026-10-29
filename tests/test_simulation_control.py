"""Control-contract tests with mocks only: no physics, renderer or ML imports."""
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vla_trial.safety import AdmissionError
from vla_trial.simulation import FrankaSimulation, HOME_QPOS
from vla_trial.finger_extensions import build_extended_panda_model, PAD_MASS_KG


class FakeVector(list):
    def __setitem__(self, key, value):
        if isinstance(key, slice) and isinstance(value, (int, float)):
            value = [value] * len(self[key])
        super().__setitem__(key, value)


class SimulationControlTests(unittest.TestCase):
    def simulation(self):
        simulation = FrankaSimulation({}, object())
        simulation._np = SimpleNamespace(array=lambda value: FakeVector(value))
        simulation.robot = Mock()
        simulation.robot.inverse_kinematics.side_effect = lambda **kwargs: FakeVector([0] * 9)
        simulation.robot.get_qpos.return_value = FakeVector([0] * 9)
        simulation.robot.plan_path.return_value = ([FakeVector([0] * 9)], True)
        simulation.arm_dofs = list(range(7))
        simulation.finger_dofs = [7, 8]
        simulation.hand = Mock()
        simulation._check_limits = Mock()
        simulation._step = Mock()
        simulation._assert_hand_near = Mock()
        return simulation

    def test_cpu_backend_still_requires_admission_before_imports(self):
        simulation = FrankaSimulation({"backend": "cpu"}, object())
        with patch("vla_trial.simulation.require_grant", side_effect=AdmissionError("denied")):
            with self.assertRaisesRegex(AdmissionError, "denied"):
                simulation.build()

    def test_only_optional_green_can_be_removed(self):
        FrankaSimulation({"active_object_ids": ["red_block", "blue_block", "yellow_block"]}, object())._validate_configuration()
        for ids in ([], ["blue_block"], ["red_block", "blue_block", "yellow_block", "unknown"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                FrankaSimulation({"active_object_ids": ids}, object())._validate_configuration()

    def test_unknown_backend_or_renderer_refused(self):
        for config in ({"backend": "auto"}, {"renderer": "unverified-batch"}, {"grasp_force": 1000},
                       {"block_size_m": 0.02}, {"block_size_m": float("nan")}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                FrankaSimulation(config, object())._validate_configuration()

    def test_pair_grasp_force_is_bounded_and_separate_from_normal_grasp(self):
        FrankaSimulation({"pair_grasp_force": 4.0, "grasp_force": 2.0, "block_size_m": 0.025}, object())._validate_configuration()
        for force in (0.5, 11, float("nan")):
            with self.subTest(force=force), self.assertRaisesRegex(ValueError, "pair_grasp_force"):
                FrankaSimulation({"pair_grasp_force": force}, object())._validate_configuration()

    def test_physics_solver_options_are_bounded_integers_and_default_unchanged(self):
        default = FrankaSimulation({}, object())
        self.assertEqual((default.physics_substeps, default.noslip_iterations), (1, 0))
        FrankaSimulation({"physics_substeps": 4, "noslip_iterations": 5}, object())._validate_configuration()
        for config in ({"physics_substeps": 0}, {"physics_substeps": 9}, {"physics_substeps": True},
                       {"physics_substeps": 2.5}, {"noslip_iterations": -1}, {"noslip_iterations": 11},
                       {"noslip_iterations": "5"}, {"noslip_iterations": False}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                FrankaSimulation(config, object())._validate_configuration()

    def test_build_passes_substeps_and_noslip_to_installed_genesis_api(self):
        simulation = FrankaSimulation({"physics_substeps": 4, "noslip_iterations": 5, "dt": 0.01}, object())
        simulation._initialize_home = Mock()
        simulation._step = Mock()
        np = SimpleNamespace(array=lambda value: FakeVector(value), arange=lambda *args: list(range(*args)))
        gs = SimpleNamespace(init=Mock(), cpu=object(), gpu=object(), Scene=Mock(),
                             options=SimpleNamespace(SimOptions=Mock(), RigidOptions=Mock()),
                             renderers=SimpleNamespace(Rasterizer=Mock()),
                             morphs=SimpleNamespace(Plane=Mock(), Box=Mock(), MJCF=Mock()),
                             materials=SimpleNamespace(Rigid=Mock()), surfaces=SimpleNamespace(Default=Mock()))
        with patch("vla_trial.simulation.require_grant"), patch.dict("sys.modules", {"genesis": gs, "numpy": np}):
            simulation.build()
        gs.options.SimOptions.assert_called_once_with(dt=0.01, substeps=4)
        gs.options.RigidOptions.assert_called_once_with(noslip_iterations=5)
        self.assertIs(gs.Scene.call_args.kwargs["rigid_options"], gs.options.RigidOptions.return_value)
        self.assertEqual(simulation.dt, 0.01)

    def test_home_initialization_sets_and_holds_all_nine_joints(self):
        simulation = self.simulation()
        simulation._initialize_home()
        self.assertEqual(simulation.robot.set_qpos.call_args.args[0], list(HOME_QPOS))
        self.assertEqual(simulation.robot.control_dofs_position.call_args.args[0], list(HOME_QPOS))
        self.assertEqual(len(simulation.robot.control_dofs_position.call_args.args), 1)
        simulation._step.assert_not_called()

    def test_carry_planner_gets_object_geometry_without_welding(self):
        simulation = self.simulation()
        entity = object()
        simulation.objects = {"blue_block": entity}
        simulation._holding_object_id = "blue_block"
        simulation._move_hand((0.46, 0.26, 0.32))
        options = simulation.robot.plan_path.call_args.kwargs
        self.assertIs(options["with_entity"], entity)
        self.assertEqual(options["ee_link_name"], "hand")
        self.assertEqual(options["max_retry"], 0)
        self.assertFalse(options["ignore_collision"])
        simulation.robot.set_qpos.assert_not_called()
        simulation.robot.attach.assert_not_called()

    def test_failed_planner_path_never_reaches_controller(self):
        simulation = self.simulation()
        simulation.robot.plan_path.return_value = ([FakeVector([0] * 9)], False)
        with self.assertRaises(RuntimeError):
            simulation._move_hand((0.46, 0.26, 0.32))
        simulation.robot.control_dofs_position.assert_not_called()
        simulation._step.assert_not_called()

    def test_contact_servo_is_bounded_vertical_arm_only_and_preserves_physics(self):
        simulation = self.simulation()
        simulation.hand_position = Mock(return_value=[0.42, 0.10, 0.32])
        simulation._contact_move((0.42, 0.10, 0.13))
        self.assertGreater(simulation.robot.inverse_kinematics.call_count, 1)
        self.assertLessEqual(simulation.robot.inverse_kinematics.call_count, 30)
        for call in simulation.robot.control_dofs_position.call_args_list:
            self.assertEqual(len(call.args[0]), 7)
            self.assertEqual(call.args[1], list(range(7)))
        simulation.robot.plan_path.assert_not_called()
        simulation.robot.set_qpos.assert_not_called()
        simulation.robot.set_pos.assert_not_called()
        simulation._step.assert_called()

    def test_contact_servo_refuses_sideways_or_out_of_workspace_move(self):
        for destination in ((0.7, 0.1, 0.13), (0.42, 0.1, 0.05), (0.42, 0.1, 0.8)):
            simulation = self.simulation()
            simulation.hand_position = Mock(return_value=[0.42, 0.10, 0.32])
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                simulation._contact_move(destination)
            simulation.robot.control_dofs_position.assert_not_called()

    def test_pick_place_has_contact_phases_and_measured_lift(self):
        simulation = self.simulation()
        simulation.objects = {"blue_block": Mock(), "red_block": Mock()}
        initial = {"blue_block": [0.42, 0.10, 0.02], "red_block": [0.42, -0.10, 0.02]}
        lifted = {**initial, "blue_block": [0.42, 0.10, 0.21]}
        final = {**initial, "blue_block": [0.46, 0.26, 0.03]}
        simulation.object_positions = Mock(side_effect=[initial, lifted, final])
        simulation._move_hand = Mock()
        simulation._contact_move = Mock()
        result = simulation.pick_place("blue_block", "left_tray")
        self.assertTrue(result["lift_verified"])
        self.assertTrue(result["placement_verified"])
        self.assertEqual(simulation._move_hand.call_count, 2)
        self.assertEqual(simulation._contact_move.call_count, 4)
        self.assertEqual(simulation.robot.control_dofs_force.call_args.args[0], [-2.0, -2.0])
        self.assertIsNone(simulation._holding_object_id)
        for entity in simulation.objects.values():
            entity.set_pos.assert_not_called()

    def test_grasp_failure_stops_before_transport(self):
        simulation = self.simulation()
        simulation.objects = {"blue_block": Mock()}
        positions = {"blue_block": [0.42, 0.10, 0.02]}
        simulation.object_positions = Mock(return_value=positions)
        simulation._move_hand = Mock()
        simulation._contact_move = Mock()
        with self.assertRaisesRegex(RuntimeError, "did not lift"):
            simulation.pick_place("blue_block", "left_tray")
        self.assertEqual(simulation._move_hand.call_count, 1)
        self.assertIsNone(simulation._holding_object_id)

    def contact_fixture(self):
        simulation = self.simulation()
        simulation.objects = {key: Mock() for key in ("red_block", "blue_block", "yellow_block")}
        simulation._move_hand = Mock()
        simulation._contact_move = Mock()
        simulation._cartesian_contact_line = Mock()
        simulation._pair_diagnostic = Mock()
        simulation._align_stack_release = Mock(return_value=(0.46, 0.08, 0.17))
        simulation._gentle_open = Mock()
        simulation._gentle_close = Mock()
        initial = {"red_block": [0.42, -0.10, 0.02], "blue_block": [0.46, 0.26, 0.03],
                   "yellow_block": [0.46, -0.34, 0.03]}
        return simulation, initial

    def test_push_requires_measured_displacement_outside_tray(self):
        simulation, initial = self.contact_fixture()
        final = {**initial, "blue_block": [0.46, 0.08, 0.02]}
        simulation.object_positions = Mock(side_effect=[initial, final])
        result = simulation.push_out("blue_block", "left_tray")
        self.assertTrue(result["push_verified"])
        self.assertTrue(result["outside_source"])
        self.assertGreater(result["displacement_m"], 0.1)
        self.assertEqual(simulation._cartesian_contact_line.call_count, 2)
        simulation.robot.control_dofs_force.assert_not_called()
        for entity in simulation.objects.values():
            entity.set_pos.assert_not_called()

    def test_failed_push_is_not_an_action_success(self):
        simulation, initial = self.contact_fixture()
        simulation.object_positions = Mock(return_value=initial)
        with self.assertRaisesRegex(RuntimeError, "Physical push did not"):
            simulation.push_out("blue_block", "left_tray")
        self.assertFalse(simulation.control_events[-1]["push_verified"])

    def test_push_rejects_occupied_corridor_before_moving(self):
        simulation, initial = self.contact_fixture()
        initial["yellow_block"] = [0.46, 0.15, 0.02]
        simulation.object_positions = Mock(return_value=initial)
        with self.assertRaisesRegex(RuntimeError, "push corridor"):
            simulation.push_out("blue_block", "left_tray")
        simulation._move_hand.assert_not_called()

    def test_red_is_never_an_allowed_contact_primitive_target(self):
        simulation, initial = self.contact_fixture()
        simulation.object_positions = Mock(return_value=initial)
        for operation in (lambda: simulation.push_out("red_block", "left_tray"),
                          lambda: simulation.stack("yellow_block", "red_block"),
                          lambda: simulation.move_stack("red_block", "yellow_block", "right_tray")):
            with self.assertRaisesRegex(ValueError, "non-red"):
                operation()
        simulation._move_hand.assert_not_called()

    def test_guard_stops_when_protected_block_moves(self):
        simulation, initial = self.contact_fixture()
        simulation.object_positions = Mock(return_value=initial)
        simulation._begin_contact_primitive("blue_block")
        displaced = {**initial, "red_block": [0.43, -0.10, 0.02]}
        with self.assertRaisesRegex(RuntimeError, "Protected red"):
            simulation._guard_contact_state(displaced)

    def test_guard_stops_pair_separation(self):
        simulation, initial = self.contact_fixture()
        simulation.object_positions = Mock(return_value=initial)
        simulation._begin_contact_primitive("blue_block", "yellow_block")
        simulation._carried_stack = ("blue_block", "yellow_block")
        with self.assertRaisesRegex(RuntimeError, "Stack geometry changed"):
            simulation._guard_contact_state(initial)

    def test_stack_verifies_release_and_repeated_settle(self):
        simulation, initial = self.contact_fixture()
        initial["blue_block"] = [0.46, 0.08, 0.02]
        lifted = {**initial, "yellow_block": [0.46, -0.34, 0.21]}
        stacked = {**initial, "yellow_block": [0.46, 0.08, 0.06]}
        simulation.object_positions = Mock(side_effect=[initial, lifted, initial] + [stacked] * 5)
        result = simulation.stack("yellow_block", "blue_block")
        self.assertTrue(result["stack_verified"])
        self.assertTrue(result["lift_verified"])
        self.assertAlmostEqual(result["height_gap_m"], 0.04)
        self.assertIsNone(simulation._holding_object_id)
        for entity in simulation.objects.values():
            entity.set_pos.assert_not_called()

    def test_stack_fails_if_blocks_settle_side_by_side(self):
        simulation, initial = self.contact_fixture()
        initial["blue_block"] = [0.46, 0.08, 0.02]
        lifted = {**initial, "yellow_block": [0.46, -0.34, 0.21]}
        fallen = {**initial, "yellow_block": [0.51, 0.08, 0.02]}
        simulation.object_positions = Mock(side_effect=[initial, lifted, initial] + [fallen] * 5)
        with self.assertRaisesRegex(RuntimeError, "stable physical stack"):
            simulation.stack("yellow_block", "blue_block")

    def test_pair_transport_measures_both_lifts_and_does_not_attach(self):
        simulation, initial = self.contact_fixture()
        initial.update({"blue_block": [0.46, 0.08, 0.02], "yellow_block": [0.46, 0.08, 0.06]})
        lifted = {**initial, "blue_block": [0.46, 0.08, 0.21], "yellow_block": [0.46, 0.08, 0.25]}
        final = {**initial, "blue_block": [0.50, -0.30, 0.03], "yellow_block": [0.50, -0.30, 0.07]}
        simulation.object_positions = Mock(side_effect=[initial, initial, initial, lifted] + [final] * 5)
        result = simulation.move_stack("blue_block", "yellow_block", "right_tray")
        self.assertTrue(result["stack_lift_verified"])
        self.assertTrue(result["both_in_target"])
        self.assertTrue(result["stack_verified"])
        self.assertEqual(simulation._move_hand.call_count, 1)
        self.assertIsNone(simulation._carried_stack)
        simulation.robot.attach.assert_not_called()
        for entity in simulation.objects.values():
            entity.set_pos.assert_not_called()

    def test_pair_transport_refuses_when_only_top_lifted(self):
        simulation, initial = self.contact_fixture()
        initial.update({"blue_block": [0.46, 0.08, 0.02], "yellow_block": [0.46, 0.08, 0.06]})
        lifted_top = {**initial, "yellow_block": [0.46, 0.08, 0.25]}
        simulation.object_positions = Mock(side_effect=[initial, initial, initial, lifted_top])
        with self.assertRaisesRegex(RuntimeError, "did not lift both"):
            simulation.move_stack("blue_block", "yellow_block", "right_tray")
        self.assertEqual(simulation._cartesian_contact_line.call_count, 2)

    def test_cartesian_contact_servo_preserves_finger_force_and_has_no_attachment(self):
        simulation = self.simulation()
        simulation.hand_position = Mock(return_value=[0.46, 0.26, 0.13])
        simulation._cartesian_contact_line((0.46, 0.11, 0.13))
        self.assertLessEqual(simulation.robot.inverse_kinematics.call_count, 40)
        for call in simulation.robot.control_dofs_position.call_args_list:
            self.assertEqual(len(call.args[0]), 7)
            self.assertEqual(call.args[1], list(range(7)))
        simulation.robot.plan_path.assert_not_called()
        simulation.robot.set_qpos.assert_not_called()
        simulation.robot.attach.assert_not_called()

    def test_cartesian_servo_rejects_nonfinite_and_unbounded_commands(self):
        for point in ((0.46, float("nan"), 0.13), (0.8, 0.1, 0.13), (0.46, -0.6, 0.13), (0.46, 0.1, 0.05)):
            simulation = self.simulation()
            simulation.hand_position = Mock(return_value=[0.46, 0.26, 0.13])
            with self.subTest(point=point), self.assertRaises(ValueError):
                simulation._cartesian_contact_line(point)
            simulation.robot.control_dofs_position.assert_not_called()

    def test_three_centimeter_scene_uses_its_actual_stack_gap(self):
        simulation = FrankaSimulation({"block_size_m": 0.03}, object())
        simulation._validate_configuration()
        aligned = {"blue_block": [0.46, 0.08, 0.015], "yellow_block": [0.46, 0.08, 0.045]}
        self.assertTrue(simulation._stack_measurements(aligned, "blue_block", "yellow_block")["stack_verified"])
        old_gap = {**aligned, "yellow_block": [0.46, 0.08, 0.055]}
        self.assertFalse(simulation._stack_measurements(old_gap, "blue_block", "yellow_block")["stack_verified"])

    def test_three_centimeter_pair_grasp_and_final_height_are_size_derived(self):
        simulation, initial = self.contact_fixture()
        simulation.block_size_m = 0.03
        initial.update({"blue_block": [0.46, 0.08, 0.015], "yellow_block": [0.462, 0.08, 0.045]})
        lifted = {**initial, "blue_block": [0.46, 0.08, 0.24], "yellow_block": [0.462, 0.08, 0.27]}
        final = {**initial, "blue_block": [0.50, -0.30, 0.025], "yellow_block": [0.50, -0.30, 0.055]}
        simulation.object_positions = Mock(side_effect=[initial, initial, initial, lifted] + [final] * 5)
        result = simulation.move_stack("blue_block", "yellow_block", "right_tray")
        self.assertTrue(result["placement_verified"])
        first = simulation._cartesian_contact_line.call_args_list[0].args[0]
        self.assertAlmostEqual(first[0], 0.461)
        self.assertAlmostEqual(first[2], 0.130)
        release = simulation._cartesian_contact_line.call_args_list[3].args[0]
        self.assertAlmostEqual(release[2], 0.142)
        self.assertEqual(simulation._pair_diagnostic.call_count, 4)

    def test_twentyfive_mm_pair_uses_pocket_fit_and_separate_four_newton_force(self):
        simulation, initial = self.contact_fixture()
        simulation.block_size_m = 0.025
        initial.update({"blue_block": [0.46, 0.08, 0.0125], "yellow_block": [0.46, 0.08, 0.0375]})
        lifted = {**initial, "blue_block": [0.46, 0.08, 0.23], "yellow_block": [0.46, 0.08, 0.255]}
        final = {**initial, "blue_block": [0.50, -0.30, 0.0225], "yellow_block": [0.50, -0.30, 0.0475]}
        simulation.object_positions = Mock(side_effect=[initial, initial, initial, lifted] + [final] * 5)
        result = simulation.move_stack("blue_block", "yellow_block", "right_tray")
        self.assertTrue(result["placement_verified"])
        first = simulation._cartesian_contact_line.call_args_list[0].args[0]
        self.assertAlmostEqual(first[2], 0.1225)
        release = simulation._cartesian_contact_line.call_args_list[3].args[0]
        self.assertAlmostEqual(release[2], 0.1345)
        self.assertEqual(simulation.robot.control_dofs_force.call_args.args[0], [-4.0, -4.0])
        self.assertEqual(simulation.control_events[0]["pair_grasp_force_n"], 4.0)

    def test_release_alignment_accumulates_commands_using_measured_object_error(self):
        simulation = self.simulation()
        simulation.block_size_m = 0.025
        simulation._contact_move = Mock()
        simulation._cartesian_contact_line = Mock()
        simulation.hand_position = Mock(return_value=[0.451, 0.08, 0.15])
        support = [0.46, 0.08, 0.0125]
        initial = {"blue_block": support, "yellow_block": [0.468, 0.08, 0.0505]}
        corrected = {**initial, "yellow_block": [0.4605, 0.0802, 0.0407]}
        simulation.object_positions = Mock(side_effect=[initial, corrected])
        result = simulation._align_stack_release("yellow_block", "blue_block", support)
        self.assertAlmostEqual(result[0], 0.452)
        self.assertAlmostEqual(result[2], 0.1555)
        self.assertEqual(simulation._cartesian_contact_line.call_count, 1)
        self.assertEqual(simulation.control_events[-1]["stage"], "stack_before_release")
        self.assertLessEqual(simulation.control_events[-1]["xy_error_m"], 0.002)
        simulation.robot.control_dofs_position.assert_not_called()
        simulation.robot.set_pos.assert_not_called()

    def test_release_alignment_failure_never_opens_gripper_or_runs_unbounded_corrections(self):
        simulation = self.simulation()
        simulation.block_size_m = 0.025
        simulation._contact_move = Mock()
        simulation._cartesian_contact_line = Mock()
        support = [0.46, 0.08, 0.0125]
        simulation.object_positions = Mock(return_value={"blue_block": support, "yellow_block": [0.465, 0.08, 0.0405]})
        with self.assertRaisesRegex(RuntimeError, "bounded corrections"):
            simulation._align_stack_release("yellow_block", "blue_block", support)
        self.assertEqual(simulation._cartesian_contact_line.call_count, 4)
        simulation.robot.control_dofs_position.assert_not_called()

    def test_pair_descent_rejects_stack_drift_after_approach(self):
        simulation, initial = self.contact_fixture()
        initial.update({"blue_block": [0.46, 0.08, 0.02], "yellow_block": [0.46, 0.08, 0.06]})
        drifted = {**initial, "yellow_block": [0.466, 0.08, 0.06]}
        simulation.object_positions = Mock(side_effect=[initial, drifted])
        with self.assertRaisesRegex(RuntimeError, "4mm alignment"):
            simulation.move_stack("blue_block", "yellow_block", "right_tray")
        simulation._cartesian_contact_line.assert_not_called()
        simulation.robot.control_dofs_force.assert_not_called()

    def test_gentle_release_ramps_measured_fingers_without_moving_arm(self):
        simulation = self.simulation()
        simulation.robot.get_dofs_position.return_value = [0.011, 0.012]
        simulation._gentle_open()
        calls = simulation.robot.control_dofs_position.call_args_list
        self.assertEqual(len(calls), 25)
        self.assertLess(calls[0].args[0][0], 0.013)
        for value in calls[-1].args[0]:
            self.assertAlmostEqual(value, 0.04)
        for previous, current in zip(calls, calls[1:]):
            self.assertGreater(current.args[0][0], previous.args[0][0])
            self.assertLess(current.args[0][0] - previous.args[0][0], 0.0013)
            self.assertEqual(current.args[1], [7, 8])
        self.assertEqual(sum(call.args[0] for call in simulation._step.call_args_list), 125)
        simulation.robot.inverse_kinematics.assert_not_called()
        simulation.robot.set_qpos.assert_not_called()

    def test_gentle_release_refuses_unknown_finger_state(self):
        simulation = self.simulation()
        simulation.robot.get_dofs_position.return_value = [float("nan"), 0.01]
        with self.assertRaisesRegex(RuntimeError, "measured finger"):
            simulation._gentle_open()
        simulation.robot.control_dofs_position.assert_not_called()

    def test_gentle_close_is_monotonic_bounded_and_finger_only_before_force(self):
        simulation = self.simulation()
        simulation.block_size_m = 0.025
        simulation.robot.get_dofs_position.return_value = [0.04, 0.039]
        simulation._gentle_close()
        calls = simulation.robot.control_dofs_position.call_args_list
        self.assertEqual(len(calls), 20)
        for value in calls[-1].args[0]:
            self.assertAlmostEqual(value, 0.0115)
        for previous, current in zip(calls, calls[1:]):
            self.assertLess(current.args[0][0], previous.args[0][0])
            self.assertLess(previous.args[0][0] - current.args[0][0], 0.0015)
            self.assertEqual(current.args[1], [7, 8])
        self.assertEqual(sum(call.args[0] for call in simulation._step.call_args_list), 150)
        simulation.robot.control_dofs_force.assert_not_called()
        simulation.robot.inverse_kinematics.assert_not_called()
        simulation.robot.set_qpos.assert_not_called()

    def test_gentle_close_refuses_unknown_finger_state(self):
        simulation = self.simulation()
        simulation.robot.get_dofs_position.return_value = [float("nan"), 0.04]
        with self.assertRaisesRegex(RuntimeError, "measured finger"):
            simulation._gentle_close()
        simulation.robot.control_dofs_position.assert_not_called()

    def test_extension_model_is_separate_symmetric_physical_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            (root / "assets" / "mesh.stl").write_bytes(b"test fixture mesh path only")
            source = root / "panda.xml"
            source.write_text('''<mujoco><compiler meshdir="assets"/><asset><mesh file="mesh.stl"/></asset><worldbody>
              <body name="left_finger" pos="0 0 0.0584"><inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/><joint name="finger_joint1"/></body>
              <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1"><inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/><joint name="finger_joint2"/></body>
              </worldbody></mujoco>''')
            original = source.read_bytes()
            generated = build_extended_panda_model(source, root / "cache")
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(build_extended_panda_model(source, root / "cache"), generated)
            model = ET.parse(generated).getroot()
            self.assertTrue(Path(model.find("./asset/mesh").get("file")).is_absolute())
            self.assertEqual(len(model.findall(".//joint")), 2)
            for name in ("left_finger", "right_finger"):
                body = model.find(f".//body[@name='{name}']")
                pad = body.find("geom[@class='collision']")
                self.assertEqual(pad.get("type"), "box")
                self.assertEqual([float(v) for v in pad.get("size").split()], [0.025, 0.003, 0.03])
                center = [float(v) for v in pad.get("pos").split()]
                self.assertAlmostEqual(center[2] + 0.0584 - 0.03, 0.110)
                self.assertAlmostEqual(center[2] + 0.0584 + 0.03, 0.170)
                inertial = body.find("inertial")
                self.assertAlmostEqual(float(inertial.get("mass")), 0.015 + PAD_MASS_KG)
                values = [float(v) for v in inertial.get("fullinertia").split()]
                self.assertTrue(all(v > 0 for v in values[:3]))
                self.assertGreater(values[1] * values[2] - values[5] ** 2, 0)
            self.assertEqual(model.find(".//body[@name='right_finger']").get("quat"), "0 0 0 1")

    def test_extension_model_refuses_unknown_mounting_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.xml"
            source.write_text('<mujoco><compiler/><worldbody><body name="left_finger" pos="0 0 0.1"/></worldbody></mujoco>')
            with self.assertRaisesRegex(ValueError, "mounting height"):
                build_extended_panda_model(source, root / "cache")
            self.assertFalse((root / "cache").exists())

    def test_extension_mode_preserves_tray_clearance_during_push(self):
        simulation, initial = self.contact_fixture()
        simulation.block_size_m = 0.025
        simulation.finger_extensions_enabled = True
        simulation.grasp_hand_offset_m = 0.165
        initial["blue_block"] = [0.46, 0.26, 0.0225]
        final = {**initial, "blue_block": [0.46, 0.08, 0.0125]}
        simulation.object_positions = Mock(side_effect=[initial, final])
        simulation.push_out("blue_block", "left_tray")
        descend = simulation._contact_move.call_args_list[0].args[0]
        self.assertAlmostEqual(descend[2], 0.186)
        self.assertGreater(descend[2] - 0.170 - 0.003, 0.01)

    def test_extension_mode_uses_longer_pair_grasp_offset(self):
        simulation, initial = self.contact_fixture()
        simulation.config.update({"pair_grasp_offset_m": 0.165})
        simulation.block_size_m = 0.025
        simulation.finger_extensions_enabled = True
        initial.update({"blue_block": [0.46, 0.08, 0.0125], "yellow_block": [0.46, 0.08, 0.0375]})
        lifted = {**initial, "blue_block": [0.46, 0.08, 0.17], "yellow_block": [0.46, 0.08, 0.195]}
        final = {**initial, "blue_block": [0.50, -0.30, 0.0225], "yellow_block": [0.50, -0.30, 0.0475]}
        simulation.object_positions = Mock(side_effect=[initial, initial, initial, lifted] + [final] * 5)
        result = simulation.move_stack("blue_block", "yellow_block", "right_tray")
        self.assertTrue(result["placement_verified"])
        self.assertAlmostEqual(simulation._cartesian_contact_line.call_args_list[0].args[0][2], 0.1775)
        self.assertAlmostEqual(simulation._cartesian_contact_line.call_args_list[3].args[0][2], 0.1895)
        simulation._gentle_close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
