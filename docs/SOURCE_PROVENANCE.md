# Source provenance

Evidence checked on 2026-10-29. This record identifies dependencies and references; it contains no robot success-rate claim.

## Implementation attribution

| Project file or function | Source relationship |
|---|---|
| `src/vla_trial/simulation.py` | Project controller informed by the official Genesis Franka grasp tutorial: Panda asset name, seven arm and two finger joints, gain/force limits, downward hand quaternion, gripper commands and grasp-height offset. Added scene layout, vertical contact motion, task sequencing, bounded execution, real-frame recording and measured checks. |
| `src/vla_trial/finger_extensions.py` | Project generator for optional collision-enabled finger pads in a derived Panda MJCF. The upstream robot description remains separately credited; added extension primitives and generation logic are project modifications. |
| `src/vla_trial/qwen.py` | Uses Qwen3-VL through the Transformers APIs. Adds this task's instruction prompt, real scene image, strict action vocabulary and local model reference. No model training or weight modification is part of the project. |
| `src/vla_trial/schema.py`, `evaluation.py` and task configuration | Project-defined action validation, protected-object rule, goal contract and trajectory scoring. A model's `done` response is not a success criterion. |
| `src/vla_trial/graphics_binding.py` | Project implementation using documented Khronos EGL and CUDA driver identity queries; no upstream EGL implementation is bundled. |
| Browser and execution modules | Project interface and orchestration built on installed Gradio and simulator/model dependencies. |

The controller reference is [Genesis v1.4.4 `IK_motion_planning_grasp.py`](https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/v1.4.4/examples/tutorials/IK_motion_planning_grasp.py). The package license is [Genesis v1.4.4 Apache-2.0](https://github.com/Genesis-Embodied-AI/genesis-world/blob/v1.4.4/LICENSE). No complete upstream tutorial file is included in this project archive.

## Robot asset evidence

The runtime loads `xml/franka_emika_panda/panda.xml` from Genesis's installed asset distribution. Its README states that the MJCF is derived from the publicly available Franka ROS URDF description and identifies Apache-2.0. The original [Franka ROS license](https://github.com/frankarobotics/franka_ros/blob/develop/LICENSE) was checked separately. The asset's README describes mesh conversions and collision geometry preparation by its upstream authors; this project does not claim those transformations as its own work.

The following SHA-256 digests identify the installed Genesis 1.4.4 files inspected for this course run. They identify those files only, not the entire dependency distribution or every mesh.

| File within the installed dependency | SHA-256 |
|---|---|
| `genesis/assets/xml/franka_emika_panda/README.md` | `d5710c80899f4b6c8c89915b097a26f962c11fb35a692350f37aef8a534896aa` |
| `genesis/assets/xml/franka_emika_panda/panda.xml` | `e502e07fcef6219024be02dfb9cd65d5c194b4767adf381ca51f10172ff8a32d` |
| `genesis_world-1.4.4.dist-info/licenses/LICENSE` | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |

The public source archive does not include the robot MJCF, mesh binaries or a copy of the installed environment. Images and video created by the project render the upstream robot asset alongside project-created primitive blocks and trays.

### Optional project finger extensions

The multistep implementation's `src/vla_trial/finger_extensions.py` generates a derived MJCF with additional
collision-enabled finger-pad geometry. This is a project-specific modification
to the upstream Panda description, not a claim that the standard Panda asset
already has these fingers. The generator reads the installed original asset and
creates a separate runtime file under the run's `generated-assets/` directory;
it must not rewrite the installed MJCF or meshes. References to installed mesh
paths are local runtime data. Generated assets are excluded from the public
evidence whitelist, while the generator source is included in the source archive.
The upstream robot and its existing geometry retain the provenance above, while
the added primitive extension geometry and generation logic are project work.
The current helper adds one collision box and one visual box per existing finger
body, with dimensions 50 mm wide × 6 mm thick × 60 mm long. A density of
1,100 kg/m³ gives 19.8 g additional mass per finger; the generator combines mass,
center of mass and full inertia with the original finger using the parallel-axis
theorem. It does not add robot joints. Earlier narrow-pad calibration used
different geometry and must not be described as using this version.
Manipulated blocks remain independent rigid bodies rather than being attached
to the robot. Runs record whether the modification was enabled and its grasp
offsets, and preserve the generator's identity in the source manifest. The
50 × 6 × 60 mm version was used in the completed formal multistep simulation
on 2026-10-29. The extensions have not been fabricated or tested on a physical
robot; simulation success does not establish hardware performance.

## Model evidence

Model: `Qwen/Qwen3-VL-8B-Instruct`.

Snapshot: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

The [exact revision's model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md) declares `apache-2.0`. The [exact revision's file listing](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) has no separate LICENSE file. The inspected existing cache likewise has no LICENSE or README file. The license evidence therefore comes from the official revision-specific model card; a license file was not fabricated or added to that cache.

The existing base snapshot is used for inference. Model weights are not copied, fine-tuned, merged with adapters or redistributed in this project. This record does not claim that every model shard was independently hashed. Other users can obtain the same revision from the linked official model repository and provide their own local path.

## Version boundaries

Genesis 1.4.4, Transformers 5.12.0, Accelerate 1.14.0, Quadrants 1.3.3, Gradio 6.30.0, ImageIO 2.38.1 and imageio-ffmpeg 0.6.0 were verified from installed package metadata.

The initial camera and model smoke tests used PyTorch 2.7.1+cu128 and Torchvision 0.22.1+cu128. The project's dependency specification was then updated to PyTorch 2.8.0+cu128 and Torchvision 0.23.0+cu128 for Genesis compatibility. Formal task results must name the actual verified environment used after this change. Historical smoke tests should not be relabeled as having used the newer versions.

This source record and the dependency inventory are separate from formal experiment results, grading, publication and submission status.
