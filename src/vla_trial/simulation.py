"""Genesis Franka scene and bounded physical manipulation primitives.

The controller uses simulator object poses (privileged state), not visual pose
estimation. RGB observations are real camera renders only. Recording writes
frames incrementally; no mock frame can count as task execution evidence.
"""
from __future__ import annotations

import json
import math
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from .safety import AdmissionGrant, require_grant
from .schema import DEFAULT_OBJECTS, DEFAULT_TARGETS


class SimulationLimitError(RuntimeError):
    pass


HOME_QPOS = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04, 0.04)


class FrankaSimulation:
    def __init__(
        self, config: Mapping[str, Any], grant: AdmissionGrant,
        *, video_path: Path | None = None, on_frame: Callable[[Any], None] | None = None,
        max_runtime_seconds: int = 180,
    ):
        self.config = dict(config)
        self.grant = grant
        self.video_path = video_path
        self.on_frame = on_frame
        self.max_runtime_seconds = max_runtime_seconds
        self.scene = None
        self.camera = None
        self.robot = None
        self.objects: dict[str, Any] = {}
        self.writer = None
        self.step_count = 0
        self.frame_count = 0
        self.started_at = None
        self.last_frame = None
        self.control_events: list[dict[str, Any]] = []
        self.backend = self.config.get("backend", "cpu")
        self.renderer = self.config.get("renderer", "rasterizer")
        self.active_object_ids = tuple(self.config.get("active_object_ids", [spec.object_id for spec in DEFAULT_OBJECTS]))
        self._holding_object_id: str | None = None
        self._protected_reference: list[float] | None = None
        self._carried_stack: tuple[str, str] | None = None
        self.block_size_m = float(self.config.get("block_size_m", 0.04))
        self.physics_substeps = self.config.get("physics_substeps", 1)
        self.noslip_iterations = self.config.get("noslip_iterations", 0)
        self.finger_extensions_enabled = self.config.get("finger_extensions", False)
        self.grasp_hand_offset_m = float(self.config.get("grasp_offset_m", 0.165 if self.finger_extensions_enabled else 0.11))

    def _validate_configuration(self) -> None:
        if self.backend not in {"cpu", "gpu"}:
            raise ValueError("simulation.backend must be cpu or gpu.")
        if self.renderer != "rasterizer":
            raise ValueError("Only the tested Rasterizer interface is implemented; CPU/GPU GL binding belongs to admission.")
        known_ids = {spec.object_id for spec in DEFAULT_OBJECTS}
        if not self.active_object_ids or len(set(self.active_object_ids)) != len(self.active_object_ids) or not set(self.active_object_ids) <= known_ids:
            raise ValueError("active_object_ids must be a nonempty unique subset of the fixed scene.")
        if not {"red_block", "blue_block", "yellow_block"} <= set(self.active_object_ids):
            raise ValueError("Keep red, blue and yellow blocks; only green may be omitted for the minimal task.")
        if not 0.1 <= float(self.config.get("grasp_force", 2.0)) <= 10.0:
            raise ValueError("grasp_force must be between 0.1 and 10 N per finger.")
        if not math.isfinite(self.block_size_m) or self.block_size_m not in (0.025, 0.03, 0.04):
            raise ValueError("block_size_m must be an explicit supported cube size: 0.025, 0.03 or 0.04 m.")
        if not 1.0 <= float(self.config.get("pair_grasp_force", 4.0)) <= 10.0:
            raise ValueError("pair_grasp_force must be between 1 and 10 N per finger.")
        if type(self.physics_substeps) is not int or not 1 <= self.physics_substeps <= 8:
            raise ValueError("physics_substeps must be an integer between 1 and 8.")
        if type(self.noslip_iterations) is not int or not 0 <= self.noslip_iterations <= 10:
            raise ValueError("noslip_iterations must be an integer between 0 and 10.")
        if type(self.finger_extensions_enabled) is not bool:
            raise ValueError("finger_extensions must be an explicit boolean.")
        for name, value in (("grasp_offset_m", self.grasp_hand_offset_m),
                            ("pair_grasp_offset_m", float(self.config.get("pair_grasp_offset_m", 0.165)))):
            if not math.isfinite(value) or not 0.10 <= value <= 0.18:
                raise ValueError(f"{name} must be between 0.10 and 0.18 m.")

    def build(self) -> None:
        require_grant(self.grant)
        self._validate_configuration()
        if self.scene is not None:
            raise RuntimeError("This scene is already built; create a fresh runtime for another run.")
        import numpy as np
        import genesis as gs

        self._np = np
        self._gs = gs
        self.dt = float(self.config.get("dt", 0.01))
        self.fps = int(self.config.get("fps", 10))
        self.frame_stride = max(1, round(1 / (self.dt * self.fps)))
        self.max_video_frames = int(self.config.get("max_video_frames", 1800))
        # A CPU physics choice does NOT authorize a renderer or bypass the
        # grant. The parent launcher owns verified EGL/OSMesa/device selection.
        gs.init(backend=gs.cpu if self.backend == "cpu" else gs.gpu)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.physics_substeps),
            rigid_options=gs.options.RigidOptions(noslip_iterations=self.noslip_iterations),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        tray_colors = {"left_tray": (0.20, 0.70, 0.75, 1.0), "right_tray": (0.70, 0.40, 0.70, 1.0)}
        for target in DEFAULT_TARGETS:
            self.scene.add_entity(
                gs.morphs.Box(size=target.size, pos=target.position, fixed=True),
                surface=gs.surfaces.Default(color=tray_colors[target.target_id]),
            )
        for spec in DEFAULT_OBJECTS:
            if spec.object_id not in self.active_object_ids:
                continue
            self.objects[spec.object_id] = self.scene.add_entity(
                gs.morphs.Box(size=(self.block_size_m,) * 3,
                              pos=(spec.position[0], spec.position[1], self.block_size_m / 2)),
                material=gs.materials.Rigid(rho=500.0, friction=1.0),
                surface=gs.surfaces.Default(color=spec.color),
            )
        robot_model = "xml/franka_emika_panda/panda.xml"
        if self.finger_extensions_enabled:
            from .finger_extensions import build_extended_panda_model, PAD_HAND_Z_M, PAD_LENGTH_M, PAD_MASS_KG, PAD_THICKNESS_M, PAD_WIDTH_M
            source_xml = Path(gs.__file__).resolve().parent / "assets" / robot_model
            default_cache = (self.video_path.parent if self.video_path is not None else Path(tempfile.gettempdir()) / "vla-trial") / "generated-assets"
            robot_model = str(build_extended_panda_model(source_xml, self.config.get("generated_model_cache_dir", default_cache)))
            self._event("physical_finger_extensions", pad_size_m=[PAD_WIDTH_M, PAD_THICKNESS_M, PAD_LENGTH_M],
                        hand_local_z_m=list(PAD_HAND_Z_M), added_mass_kg_per_finger=PAD_MASS_KG,
                        grasp_hand_offset_m=self.grasp_hand_offset_m, manipulated_objects_attached=False)
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=robot_model))
        self.camera = self.scene.add_camera(
            res=(int(self.config.get("width", 640)), int(self.config.get("height", 480))),
            pos=(1.15, -1.25, 1.0), lookat=(0.35, 0.0, 0.30), fov=50, GUI=False,
        )
        self.scene.build()
        self.robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.robot.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        self.hand = self.robot.get_link("hand")
        self.arm_dofs = np.arange(7)
        self.finger_dofs = np.arange(7, 9)
        self._initialize_home()
        if self.video_path is not None:
            import imageio.v2 as imageio
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(
                str(self.video_path), fps=self.fps, codec="libx264", quality=7,
                macro_block_size=2, ffmpeg_params=["-threads", "2"],
            )
        self.started_at = time.monotonic()
        self._step(100)

    def _initialize_home(self) -> None:
        """Set the initial robot pose once, then hold every joint before stepping.

        This initialization happens before task recording. Objects are never
        teleported and this method is never called during pick/place execution.
        """
        home = self._np.array(HOME_QPOS)
        self.robot.set_qpos(home)
        self.robot.control_dofs_position(home)

    def _event(self, stage: str, **details: Any) -> None:
        self.control_events.append({"stage": stage, "step": self.step_count, **details})

    def _check_limits(self) -> None:
        require_grant(self.grant)
        if self.started_at is None or self.scene is None:
            raise RuntimeError("Scene has not been built.")
        if time.monotonic() - self.started_at > self.max_runtime_seconds:
            raise SimulationLimitError("Wall-time limit reached; stopping without restart.")
        if self.frame_count >= self.max_video_frames:
            raise SimulationLimitError("Recording frame limit reached; stopping without buffering more frames.")

    def frame(self) -> Any:
        self._check_limits()
        result = self.camera.render(rgb=True, depth=False, segmentation=False)
        rgb = result[0] if isinstance(result, tuple) else result
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        rgb = self._np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            raise RuntimeError("Renderer did not return an RGB image; no fallback image will be substituted.")
        self.last_frame = rgb[:, :, :3].astype(self._np.uint8, copy=False)
        return self.last_frame

    def _step(self, steps: int) -> None:
        for _ in range(steps):
            self._check_limits()
            self.scene.step()
            self.step_count += 1
            if self._protected_reference is not None and self.step_count % 10 == 0:
                self._guard_contact_state()
            if self.step_count % self.frame_stride == 0 and (self.writer is not None or self.on_frame is not None):
                rgb = self.frame()
                if self.writer is not None:
                    self.writer.append_data(rgb)
                self.frame_count += 1
                if self.on_frame is not None:
                    self.on_frame(rgb)

    def object_positions(self) -> dict[str, list[float]]:
        self._check_limits()
        positions = {}
        for object_id, entity in self.objects.items():
            value = entity.get_pos()
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            positions[object_id] = self._np.asarray(value).reshape(-1).tolist()
        return positions

    def hand_position(self) -> list[float]:
        self._check_limits()
        value = self.hand.get_pos()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return self._np.asarray(value).reshape(-1).tolist()

    def save_observation(self, image_path: str | Path) -> dict[str, Any]:
        """Save an actual renderer frame and measured positions, never a mock."""
        self._check_limits()
        destination = Path(image_path)
        if destination.suffix.lower() != ".png":
            raise ValueError("Observation images must use .png.")
        rgb = self.frame()
        from PIL import Image
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(destination)
        metadata = {
            "image": destination.name, "step": self.step_count,
            "simulation_seconds": self.step_count * self.dt,
            "object_positions": self.object_positions(), "hand_position": self.hand_position(),
            "physics_backend": self.backend, "renderer": self.renderer,
            "evidence": "actual simulator camera; object coordinates are privileged simulator measurements",
        }
        destination.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return metadata

    def _assert_hand_near(self, position: tuple[float, float, float], tolerance: float = 0.025) -> None:
        actual = self.hand_position()
        distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(actual, position)))
        if not math.isfinite(distance) or distance > tolerance:
            raise RuntimeError(f"Hand did not reach commanded pose (error {distance:.4f}m); stopping.")

    def _move_hand(self, position: tuple[float, float, float], *, open_gripper: bool = False) -> None:
        """Collision-checked free-space move, with carried-object geometry."""
        self._check_limits()
        qpos = self.robot.inverse_kinematics(
            link=self.hand, pos=self._np.array(position), quat=self._np.array([0, 1, 0, 0]),
        )
        if open_gripper:
            qpos[-2:] = 0.04
        attachment = {}
        if self._holding_object_id is not None:
            attachment = {"ee_link_name": "hand", "with_entity": self.objects[self._holding_object_id]}
        path, is_valid = self.robot.plan_path(
            qpos_goal=qpos, num_waypoints=200, return_valid_mask=True,
            timeout=45.0, max_retry=0, ignore_collision=False, **attachment,
        )
        # Genesis can return a nonempty path even when planning failed.
        # The implementation returns ~is_invalid for return_valid_mask=True;
        # the source docstring's is_invalid return label is misleading.
        valid_value = is_valid.item() if hasattr(is_valid, "item") else is_valid
        if type(valid_value) is not bool or not valid_value or path is None or len(path) == 0:
            raise RuntimeError("No valid collision-checked path; refusing all returned waypoints.")
        self._check_limits()
        for waypoint in path:
            if open_gripper:
                self.robot.control_dofs_position(waypoint)
            else:
                self.robot.control_dofs_position(waypoint[:7], self.arm_dofs)
            self._step(1)
        self._step(100)
        self._assert_hand_near(position)

    def _contact_move(self, position: tuple[float, float, float], *, open_gripper: bool = False) -> None:
        """Slow vertical IK servo for intentional grasp/release contact.

        Global collision planning rejects intentional finger/object contact.
        These bounded vertical segments keep the physics collision solver on,
        command arm joints only, and never weld, attach or teleport objects.
        """
        self._check_limits()
        start = self.hand_position()
        if math.hypot(start[0] - position[0], start[1] - position[1]) > 0.025:
            raise ValueError("Contact motion must be vertical within 2.5cm; use a planned free-space move first.")
        if not 0.11 <= position[2] <= 0.40 or abs(position[2] - start[2]) > 0.30:
            raise ValueError("Contact motion leaves the bounded tabletop workspace.")
        if open_gripper:
            self.robot.control_dofs_position(self._np.array([0.04, 0.04]), self.finger_dofs)
        count = max(4, min(30, math.ceil(abs(position[2] - start[2]) / 0.01)))
        for index in range(1, count + 1):
            fraction = index / count
            point = [a + fraction * (b - a) for a, b in zip(start, position)]
            qpos = self.robot.inverse_kinematics(
                link=self.hand, pos=self._np.array(point), quat=self._np.array([0, 1, 0, 0]),
                init_qpos=self.robot.get_qpos(), max_samples=5, max_solver_iters=30,
            )
            self.robot.control_dofs_position(qpos[:7], self.arm_dofs)
            self._step(10)
        self._step(80)
        self._assert_hand_near(position)

    def pick_place(self, object_id: str, target_id: str) -> dict[str, Any]:
        self._check_limits()
        targets = {target.target_id: target for target in DEFAULT_TARGETS}
        if object_id not in self.objects or target_id not in targets:
            raise ValueError("Object/target is not in the fixed scene allowlist.")
        positions_before = self.object_positions()
        start = positions_before[object_id]
        target = targets[target_id]
        slots = [(target.position[0] + dx, target.position[1] + dy) for dx in (-0.04, 0.04) for dy in (-0.04, 0.04)]
        free = [slot for slot in slots if all(
            key == object_id or (value[0] - slot[0]) ** 2 + (value[1] - slot[1]) ** 2 > 0.055 ** 2
            for key, value in positions_before.items()
        )]
        if not free:
            raise RuntimeError("No free target placement slot.")
        x, y = free[0]
        # Franka hand link is approximately 0.11m above cube center in the
        # official grasp example. These motions require later physics tuning.
        self._event("approach", object_id=object_id, target_id=target_id)
        self._move_hand((start[0], start[1], 0.32), open_gripper=True)
        self._event("descend_to_grasp")
        self._contact_move((start[0], start[1], start[2] + self.grasp_hand_offset_m), open_gripper=True)
        force = float(self.config.get("grasp_force", 2.0))
        if self.finger_extensions_enabled:
            self._gentle_close()
        self.robot.control_dofs_force(self._np.array([-force, -force]), self.finger_dofs)
        self._step(100)
        self._event("physical_lift")
        self._contact_move((start[0], start[1], 0.32))
        lifted = self.object_positions()[object_id][2] - start[2] > 0.05
        if not lifted:
            raise RuntimeError("Grasp did not lift the object; stopping instead of claiming success.")
        self._holding_object_id = object_id
        self._event("transport")
        self._move_hand((x, y, 0.32))
        self._event("descend_to_place")
        tray_top = target.position[2] + target.size[2] / 2
        self._contact_move((x, y, tray_top + self.block_size_m / 2 + self.grasp_hand_offset_m + 0.005))
        self.robot.control_dofs_position(self._np.array([0.04, 0.04]), self.finger_dofs)
        self._step(100)
        self._holding_object_id = None
        self._event("release_and_retreat")
        self._contact_move((x, y, 0.32), open_gripper=True)
        self._step(100)
        final = self.object_positions()[object_id]
        inside = abs(final[0] - target.position[0]) <= 0.07 and abs(final[1] - target.position[1]) <= 0.07
        settled_height = abs(final[2] - (tray_top + self.block_size_m / 2)) <= 0.015
        return {
            "object_id": object_id, "target_id": target_id,
            "position_before": start, "position_after": final,
            "placement_verified": bool(inside and settled_height),
            "lift_verified": bool(lifted), "controller": "planned transport + vertical contact IK; physical finger force",
            "verification": "simulator position at final settle; not a perception-only result",
        }

    def _begin_contact_primitive(self, *object_ids: str) -> dict[str, list[float]]:
        self._check_limits()
        if not object_ids or len(set(object_ids)) != len(object_ids) or any(
            object_id not in self.objects or object_id == "red_block" for object_id in object_ids
        ):
            raise ValueError("Contact primitives require distinct non-red objects from the fixed scene.")
        positions = self.object_positions()
        if "red_block" not in positions:
            raise ValueError("The protected red block is required.")
        if self._protected_reference is None:
            self._protected_reference = list(positions["red_block"])
        self._guard_contact_state(positions)
        return positions

    def _stack_measurements(self, positions: Mapping[str, list[float]], base_id: str, top_id: str) -> dict[str, Any]:
        base, top = positions[base_id], positions[top_id]
        xy_error = math.hypot(top[0] - base[0], top[1] - base[1])
        height_gap = top[2] - base[2]
        return {
            "relative_xy_m": xy_error, "height_gap_m": height_gap,
            "stack_verified": bool(math.isfinite(xy_error) and math.isfinite(height_gap)
                                   and xy_error <= 0.012 and abs(height_gap - self.block_size_m) <= 0.008),
        }

    def _guard_contact_state(self, positions: Mapping[str, list[float]] | None = None) -> None:
        if self._protected_reference is None:
            return
        positions = self.object_positions() if positions is None else positions
        displacement = math.dist(positions["red_block"], self._protected_reference)
        if not math.isfinite(displacement) or displacement > 0.005:
            self._event("protected_object_moved", displacement_m=displacement)
            raise RuntimeError("Protected red block moved over 5mm; stopping contact motion.")
        if self._carried_stack is not None:
            geometry = self._stack_measurements(positions, *self._carried_stack)
            if (not math.isfinite(geometry["relative_xy_m"]) or not math.isfinite(geometry["height_gap_m"])
                    or geometry["relative_xy_m"] > 0.018 or abs(geometry["height_gap_m"] - self.block_size_m) > 0.014):
                self._pair_diagnostic("stack_pair_guard_diagnostic", *self._carried_stack)
                self._event("stack_separation_detected", **geometry)
                raise RuntimeError("Stack geometry changed during transport; stopping without reconstructing it.")

    def _cartesian_contact_line(
        self, position: tuple[float, float, float], *, step_m: float = 0.004,
        steps_per_point: int = 10, open_gripper: bool = False,
    ) -> None:
        """Bounded slow arm servo with physical contact; this is not an RRT plan.

        Only the fixed tabletop corridor is allowed. Callers separately choose
        the intentional push corridor or a high, clear stack transport plane.
        The actual blocks remain independent rigid bodies at every step.
        """
        self._check_limits()
        start = self.hand_position()
        if not 0.001 <= step_m <= 0.01 or not 5 <= steps_per_point <= 30:
            raise ValueError("Contact servo speed lies outside the bounded configuration.")
        if any(not math.isfinite(float(value)) for value in (*start, *position)):
            raise ValueError("Contact servo coordinates must be finite.")
        for point in (start, position):
            if not (0.34 <= point[0] <= 0.68 and -0.43 <= point[1] <= 0.43 and 0.11 <= point[2] <= 0.40):
                raise ValueError("Contact servo leaves the bounded tabletop workspace.")
        distance = math.dist(start, position)
        if distance > 0.70:
            raise ValueError("Contact servo segment is too long.")
        count = max(1, math.ceil(distance / step_m))
        if count > 250:
            raise ValueError("Contact servo exceeds its waypoint limit.")
        self._guard_contact_state()
        if open_gripper:
            self.robot.control_dofs_position(self._np.array([0.04, 0.04]), self.finger_dofs)
        for index in range(1, count + 1):
            fraction = index / count
            point = [a + fraction * (b - a) for a, b in zip(start, position)]
            qpos = self.robot.inverse_kinematics(
                link=self.hand, pos=self._np.array(point), quat=self._np.array([0, 1, 0, 0]),
                init_qpos=self.robot.get_qpos(), max_samples=5, max_solver_iters=30,
            )
            self.robot.control_dofs_position(qpos[:7], self.arm_dofs)
            self._step(steps_per_point)
            self._guard_contact_state()
        self._step(50)
        self._assert_hand_near(position, tolerance=0.015)

    def _settled_stack(self, base_id: str, top_id: str) -> tuple[dict[str, list[float]], dict[str, Any]]:
        samples = []
        for _ in range(5):
            self._step(20)
            positions = self.object_positions()
            self._guard_contact_state(positions)
            samples.append(positions)
        geometry = self._stack_measurements(samples[-1], base_id, top_id)
        spread = max(math.dist(sample[object_id], samples[0][object_id])
                     for sample in samples for object_id in (base_id, top_id))
        geometry["settle_displacement_m"] = spread
        geometry["stack_verified"] = bool(geometry["stack_verified"] and spread <= 0.005
                                          and all(self._stack_measurements(sample, base_id, top_id)["stack_verified"]
                                                  for sample in samples))
        return samples[-1], geometry

    def _align_stack_release(
        self, object_id: str, support_id: str, support_reference: list[float],
    ) -> tuple[float, float, float]:
        """Correct held-object error through arm PD before opening the fingers.

        Command corrections accumulate from the previous command, compensating
        the measured grasp/IK offset. Four corrections and 3cm total correction
        travel are hard limits; simulator body poses are read, never assigned.
        """
        command = (support_reference[0], support_reference[1],
                   support_reference[2] + self.block_size_m + self.grasp_hand_offset_m + 0.018)
        self._contact_move(command)
        correction_travel = 0.0
        for iteration in range(5):
            positions = self.object_positions()
            self._guard_contact_state(positions)
            support, held = positions[support_id], positions[object_id]
            if math.dist(support, support_reference) > 0.005:
                raise RuntimeError("Support cube moved during release alignment; stopping.")
            desired = [support[0], support[1], support[2] + self.block_size_m + 0.003]
            error = [target - actual for target, actual in zip(desired, held)]
            xy_error = math.hypot(error[0], error[1])
            self._event("stack_release_alignment", iteration=iteration, desired_object_position=desired,
                        object_position=held, support_position=support, hand_command=list(command),
                        error_m=error, correction_travel_m=correction_travel)
            if xy_error <= 0.002 and abs(error[2]) <= 0.001:
                self._event("stack_before_release", object_id=object_id, support_id=support_id,
                            object_position=held, support_position=support, hand_position=self.hand_position(),
                            xy_error_m=xy_error, release_gap_m=held[2] - support[2] - self.block_size_m,
                            correction_count=iteration)
                return command
            distance = math.sqrt(sum(value * value for value in error))
            if iteration == 4 or not math.isfinite(distance) or correction_travel + distance > 0.03:
                raise RuntimeError("Held cube could not align within 2mm XY and 3mm release gap in bounded corrections.")
            correction_travel += distance
            command = tuple(value + correction for value, correction in zip(command, error))
            self._cartesian_contact_line(command, step_m=0.002)
        raise AssertionError("Unreachable release alignment state.")

    def _pair_diagnostic(self, stage: str, base_id: str, top_id: str) -> None:
        """Compact actual pose/contact snapshot; never changes physics state."""
        def values(value: Any) -> Any:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return value.tolist() if hasattr(value, "tolist") else list(value)

        try:
            contacts = {}
            for key in (base_id, top_id):
                measured = self.robot.get_contacts(with_entity=self.objects[key])
                contacts[key] = {name: values(measured[name])[:16]
                                 for name in ("link_a", "link_b", "position", "force_a", "force_b")}
            self._event(stage, hand_position=self.hand_position(), hand_quaternion=values(self.hand.get_quat()),
                        finger_positions=values(self.robot.get_dofs_position(self.finger_dofs)),
                        object_quaternions={key: values(self.objects[key].get_quat()) for key in (base_id, top_id)},
                        object_positions=self.object_positions(), contacts=contacts)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            # Diagnostics cannot turn a physics failure into success or mask it.
            self._event(stage, diagnostic_error=f"{type(exc).__name__}: {exc}")

    def _gentle_open(self) -> None:
        """Ramp finger separation from its measured value while holding the arm.

        An immediate 4cm command produced a measured release impulse in the
        small-block calibration. This bounded position ramp changes neither
        object state nor the arm target and remains ordinary contact dynamics.
        """
        measured = self.robot.get_dofs_position(self.finger_dofs)
        if hasattr(measured, "detach"):
            measured = measured.detach().cpu().numpy()
        start = [float(value) for value in measured]
        if len(start) != 2 or any(not math.isfinite(value) or not -0.001 <= value <= 0.045 for value in start):
            raise RuntimeError("Cannot gently release without two valid measured finger positions.")
        self._event("gentle_release_begin", finger_positions=start, ramp_steps=125)
        for index in range(1, 26):
            targets = [value + (0.04 - value) * index / 25 for value in start]
            self.robot.control_dofs_position(self._np.array(targets), self.finger_dofs)
            self._step(5)
        self._event("gentle_release_end")

    def _gentle_close(self) -> None:
        """Position-ramp the physical finger extensions before force holding."""
        measured = self.robot.get_dofs_position(self.finger_dofs)
        if hasattr(measured, "detach"):
            measured = measured.detach().cpu().numpy()
        start = [float(value) for value in measured]
        if len(start) != 2 or any(not math.isfinite(value) or not -0.001 <= value <= 0.045 for value in start):
            raise RuntimeError("Cannot gently close without two valid measured finger positions.")
        start = [max(0.0, value) for value in start]
        target = [min(value, self.block_size_m / 2 - 0.001) for value in start]
        self._event("gentle_grasp_begin", finger_positions=start, target_positions=target, ramp_steps=100)
        for index in range(1, 21):
            command = [value + (end - value) * index / 20 for value, end in zip(start, target)]
            self.robot.control_dofs_position(self._np.array(command), self.finger_dofs)
            self._step(5)
        self._step(50)
        self._event("gentle_grasp_ready_for_force")

    def push_out(self, object_id: str, source_target_id: str) -> dict[str, Any]:
        """Push a cube off tray A toward the clear +Y staging area on the plane."""
        if source_target_id != "left_tray":
            raise ValueError("The verified geometric push corridor starts at tray A only.")
        positions_before = self._begin_contact_primitive(object_id)
        start = positions_before[object_id]
        target = next(target for target in DEFAULT_TARGETS if target.target_id == source_target_id)
        half = self.block_size_m / 2
        tray_top = target.position[2] + target.size[2] / 2
        if not (abs(start[0] - target.position[0]) <= 0.07
                and abs(start[1] - target.position[1]) <= 0.07 and abs(start[2] - (tray_top + half)) <= 0.01):
            raise RuntimeError("Push source cube is not settled inside tray A.")
        x, staging_y = start[0], 0.08
        # Check the low push corridor before approaching; red is also checked
        # throughout actual stepping. The far side of A is reached from above.
        if any(key != object_id and abs(value[0] - x) < self.block_size_m + 0.025
               and staging_y - 0.055 < value[1] < start[1] + 0.09
               for key, value in positions_before.items()):
            raise RuntimeError("Another block occupies the bounded push corridor.")
        behind_y = start[1] + 0.065
        self._event("push_approach", object_id=object_id, source_target_id=source_target_id,
                    positions_before=positions_before, staging_xy=[x, staging_y])
        self._move_hand((x, behind_y, 0.32), open_gripper=True)
        self.robot.control_dofs_position(self._np.array([0.0, 0.0]), self.finger_dofs)
        self._step(50)
        push_clearance = 0.176 if self.finger_extensions_enabled else 0.1145
        push_height = max(tray_top + push_clearance, half + self.grasp_hand_offset_m)
        push_end_y = staging_y + half + 0.008
        self._contact_move((x, behind_y, push_height))
        self._event("physical_push")
        self._cartesian_contact_line((x, push_end_y, push_height), step_m=0.003)
        # Withdraw in the opposite direction before lifting the gripper.
        self._cartesian_contact_line((x, push_end_y + 0.035, push_height), step_m=0.003)
        self._contact_move((x, push_end_y + 0.035, 0.32), open_gripper=True)
        self._step(100)
        positions_after = self.object_positions()
        self._guard_contact_state(positions_after)
        final = positions_after[object_id]
        displacement = math.hypot(final[0] - start[0], final[1] - start[1])
        outside = (abs(final[0] - target.position[0]) > target.size[0] / 2 + half
                   or abs(final[1] - target.position[1]) > target.size[1] / 2 + half)
        verified = bool(outside and displacement >= 0.10 and abs(final[2] - half) <= 0.008
                        and abs(final[0] - x) <= 0.04 and abs(final[1] - staging_y) <= 0.04)
        result = {"action": "push_out", "object_id": object_id, "source_target_id": source_target_id,
                  "position_before": start, "position_after": final, "positions_before": positions_before,
                  "positions_after": positions_after, "displacement_m": displacement,
                  "outside_source": bool(outside), "push_verified": verified,
                  "controller": "closed fingers; bounded horizontal contact IK; independent rigid-body physics"}
        self._event("push_measured", **result)
        if not verified:
            raise RuntimeError("Physical push did not put the cube fully outside tray A in the staging area.")
        return result

    def stack(self, object_id: str, support_id: str) -> dict[str, Any]:
        """Grasp one cube and release it onto another; verify a settled stack."""
        positions_before = self._begin_contact_primitive(object_id, support_id)
        start, support = positions_before[object_id], positions_before[support_id]
        if not abs(support[2] - self.block_size_m / 2) <= 0.008 or not (0.37 <= support[0] <= 0.63 and 0.03 <= support[1] <= 0.13):
            raise RuntimeError("Support cube must be settled in the clear staging area.")
        self._event("stack_approach", object_id=object_id, support_id=support_id, positions_before=positions_before)
        self._move_hand((start[0], start[1], 0.32), open_gripper=True)
        self._contact_move((start[0], start[1], start[2] + self.grasp_hand_offset_m), open_gripper=True)
        self._pair_diagnostic("stack_pickup_before_close", support_id, object_id)
        force = float(self.config.get("grasp_force", 2.0))
        if self.finger_extensions_enabled:
            self._gentle_close()
        self.robot.control_dofs_force(self._np.array([-force, -force]), self.finger_dofs)
        self._step(100)
        self._pair_diagnostic("stack_pickup_after_close", support_id, object_id)
        self._contact_move((start[0], start[1], 0.32))
        lifted = self.object_positions()[object_id][2] - start[2] > 0.05
        if not lifted:
            raise RuntimeError("Stack grasp did not lift the top cube.")
        self._holding_object_id = object_id
        self._move_hand((support[0], support[1], 0.32))
        current = self.object_positions()
        if math.dist(current[support_id], support) > 0.005:
            raise RuntimeError("Support cube moved before stacking; stopping.")
        self._event("stack_place_contact")
        release_command = self._align_stack_release(object_id, support_id, support)
        self._pair_diagnostic("stack_before_gentle_release", support_id, object_id)
        self._gentle_open()
        self._step(100)
        self._pair_diagnostic("stack_after_gentle_release", support_id, object_id)
        self._holding_object_id = None
        self._contact_move((release_command[0], release_command[1], 0.32), open_gripper=True)
        positions_after, geometry = self._settled_stack(support_id, object_id)
        geometry["stack_verified"] = bool(geometry["stack_verified"] and geometry["relative_xy_m"] <= 0.004)
        result = {"action": "stack", "object_id": object_id, "support_id": support_id,
                  "positions_before": positions_before, "positions_after": positions_after,
                  "lift_verified": bool(lifted), **geometry,
                  "controller": "physical finger grasp; gravity-supported release onto measured support cube"}
        self._event("stack_measured", **result)
        if not geometry["stack_verified"]:
            raise RuntimeError("Released cubes did not form a stable physical stack.")
        return result

    def move_stack(self, base_id: str, top_id: str, target_id: str) -> dict[str, Any]:
        """Physically pinch the stacked pair and carry both without a weld.

        The fingers can contact both cubes near their seam. This is deliberately
        described as a pair grasp, not as evidence of base-only support during
        motion. Independent measured lift and stack geometry are mandatory.
        """
        targets = {target.target_id: target for target in DEFAULT_TARGETS}
        if target_id != "right_tray":
            raise ValueError("Stack transport is bounded to the empty tray B corridor.")
        positions_before = self._begin_contact_primitive(base_id, top_id)
        initial_geometry = self._stack_measurements(positions_before, base_id, top_id)
        if not initial_geometry["stack_verified"] or initial_geometry["relative_xy_m"] > 0.004:
            raise RuntimeError("Move-stack requires an already aligned physical stack.")
        base = positions_before[base_id]
        target = targets[target_id]
        x, y = target.position[:2]
        if any(key not in {base_id, top_id} and math.hypot(value[0] - x, value[1] - y) < 0.10
               for key, value in positions_before.items()):
            raise RuntimeError("Tray B is not empty enough for the stacked pair.")
        # The palm-to-finger-ledge pocket is ~28.45mm high. A 25mm top cube
        # fits this pocket; 30/40mm pair trials showed wedging and lower-cube
        # creep. Dimensions are explicit scene/task inputs, never changed
        # mid-run. Independent body/contact diagnostics preserve the evidence.
        default_offset = 0.165 if self.finger_extensions_enabled else {0.025: 0.110, 0.03: 0.115, 0.04: 0.125}[self.block_size_m]
        hand_offset = float(self.config.get("pair_grasp_offset_m", default_offset))
        pair_force = float(self.config.get("pair_grasp_force", 4.0))
        grasp_x = (base[0] + positions_before[top_id][0]) / 2
        grasp_y = (base[1] + positions_before[top_id][1]) / 2
        self._event("stack_pair_approach", base_id=base_id, top_id=top_id, target_id=target_id,
                    positions_before=positions_before, hand_base_offset_m=hand_offset,
                    block_size_m=self.block_size_m, pair_grasp_force_n=pair_force)
        self._move_hand((grasp_x, grasp_y, 0.34), open_gripper=True)
        pre_grasp = self.object_positions()
        pre_grasp_geometry = self._stack_measurements(pre_grasp, base_id, top_id)
        if not pre_grasp_geometry["stack_verified"] or pre_grasp_geometry["relative_xy_m"] > 0.004:
            raise RuntimeError("Stack drifted out of 4mm alignment before pair descent; stopping.")
        grasp_x = (pre_grasp[base_id][0] + pre_grasp[top_id][0]) / 2
        grasp_y = (pre_grasp[base_id][1] + pre_grasp[top_id][1]) / 2
        self._cartesian_contact_line((grasp_x, grasp_y, base[2] + hand_offset), step_m=0.003, open_gripper=True)
        self._pair_diagnostic("stack_pair_before_close", base_id, top_id)
        if self.finger_extensions_enabled:
            self._gentle_close()
        self.robot.control_dofs_force(self._np.array([-pair_force, -pair_force]), self.finger_dofs)
        self._step(100)
        self._pair_diagnostic("stack_pair_after_close", base_id, top_id)
        after_grasp = self.object_positions()
        if not self._stack_measurements(after_grasp, base_id, top_id)["stack_verified"]:
            raise RuntimeError("Pair grasp displaced the stack; refusing to lift.")
        self._carried_stack = (base_id, top_id)
        self._event("stack_pair_physical_lift")
        self._cartesian_contact_line((grasp_x, grasp_y, 0.34), step_m=0.004)
        lifted_positions = self.object_positions()
        lift_heights = {key: lifted_positions[key][2] - positions_before[key][2] for key in (base_id, top_id)}
        lifted = all(value >= 0.05 for value in lift_heights.values())
        if not lifted:
            raise RuntimeError("Physical pair grasp did not lift both cubes together.")
        self._pair_diagnostic("stack_pair_after_lift", base_id, top_id)
        self._event("stack_pair_lift_measured", lift_heights_m=lift_heights,
                    positions=lifted_positions, stack_lift_verified=bool(lifted))
        # Transport both independent bodies slowly through a high clear plane;
        # do not pass either body to the single-entity planning attachment API.
        self._cartesian_contact_line((x, y, 0.34), step_m=0.004)
        tray_top = target.position[2] + target.size[2] / 2
        self._cartesian_contact_line((x, y, tray_top + self.block_size_m / 2 + hand_offset + 0.002), step_m=0.004)
        self._pair_diagnostic("stack_pair_before_release", base_id, top_id)
        self._gentle_open()
        self._step(100)
        self._carried_stack = None
        self._cartesian_contact_line((x, y, 0.34), step_m=0.004, open_gripper=True)
        positions_after, geometry = self._settled_stack(base_id, top_id)
        both_inside = all(abs(positions_after[key][0] - x) <= 0.07 and abs(positions_after[key][1] - y) <= 0.07
                          for key in (base_id, top_id))
        base_settled = abs(positions_after[base_id][2] - (tray_top + self.block_size_m / 2)) <= 0.008
        verified = bool(both_inside and base_settled and geometry["stack_verified"])
        result = {"action": "move_stack", "base_id": base_id, "top_id": top_id, "target_id": target_id,
                  "positions_before": positions_before, "positions_after": positions_after,
                  "lift_heights_m": lift_heights, "stack_lift_verified": bool(lifted),
                  "both_in_target": bool(both_inside and base_settled), "placement_verified": verified, **geometry,
                  "controller": "physical finger-pad grasp; slow Cartesian arm PD; no object weld or fixed joint",
                  "finger_extensions_enabled": self.finger_extensions_enabled}
        self._event("stack_transport_measured", **result)
        if not verified:
            raise RuntimeError("Stacked pair did not settle together inside tray B.")
        return result

    def close(self) -> None:
        writer, scene = self.writer, self.scene
        self.writer = None
        self.scene = None
        self.camera = None
        self.robot = None
        self.objects.clear()
        self._holding_object_id = None
        self._protected_reference = None
        self._carried_stack = None
        first_error = None
        try:
            if writer is not None:
                writer.close()
        except BaseException as exc:
            first_error = exc
        try:
            if scene is not None and hasattr(scene, "destroy"):
                scene.destroy()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error
