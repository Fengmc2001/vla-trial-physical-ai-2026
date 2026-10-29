# Third-party sources and notices

Checked on 2026-10-29. This project combines a pretrained visual-language model with a simulator and conventional robot control. It does not train a new end-to-end vision-language-action model. The name 尝试VLA describes the project direction, not a claim of a new pretrained VLA model.

## Components used

| Component | Version or revision | Role and license evidence |
|---|---|---|
| [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world) | 1.4.4 | Physics, inverse kinematics, path planning and camera rendering. The installed distribution's license and the [v1.4.4 license](https://github.com/Genesis-Embodied-AI/genesis-world/blob/v1.4.4/LICENSE) identify Apache-2.0. |
| [Franka Emika Panda MJCF and meshes](https://github.com/Genesis-Embodied-AI/genesis-world/tree/main/genesis/assets/xml/franka_emika_panda) | Assets supplied with Genesis 1.4.4 | Robot description and appearance. Its installed README identifies Apache-2.0 and credits the [Franka ROS description](https://github.com/frankarobotics/franka_ros/tree/develop/franka_description), whose [license](https://github.com/frankarobotics/franka_ros/blob/develop/LICENSE) was also checked. |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Pretrained image-and-instruction inference. The [model card at this exact revision](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md) declares `apache-2.0`. Neither the inspected cache nor the upstream file listing at this revision contains a separate LICENSE file; this is a model-card declaration, not a claim that a cached license file was verified. |
| [Transformers](https://github.com/huggingface/transformers) / [Accelerate](https://github.com/huggingface/accelerate) | 5.12.0 / 1.14.0 | Model loading, processing and inference. Installed package metadata and license files identify Apache-2.0. |
| [PyTorch](https://pytorch.org/) / [Torchvision](https://github.com/pytorch/vision) | Project pins: 2.8.0+cu128 / 0.23.0+cu128 | Tensor runtime and image support. PyTorch declares BSD-3-Clause; Torchvision carries its BSD license. Package binaries and transitive components have their own notices. |
| [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) | 1.3.3 | Genesis simulation kernels. Installed package metadata and license text identify Apache-2.0. |
| [Gradio](https://github.com/gradio-app/gradio) | 6.30.0 | Browser interface. Installed package license identifies Apache-2.0. |
| [ImageIO](https://github.com/imageio/imageio) / [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | 2.38.1 / 0.6.0 | Frame streaming and video encoding interface. Installed metadata identifies BSD-2-Clause for these Python packages. The FFmpeg executable and codecs retain their separate licensing; the Python wrapper license is not a license for every codec. |

The dependency pins are installation specifications. Runtime results must identify the environment actually used; installing the listed versions alone does not demonstrate a successful robot task. `requirements.lock.txt` records the complete dependency set after environment verification.

## Control references and adaptations

`src/vla_trial/simulation.py` uses the approach, joint indexing, PD gain values, hand orientation and approximate grasp offset described in the official [Genesis grasp tutorial at v1.4.4](https://raw.githubusercontent.com/Genesis-Embodied-AI/genesis-world/v1.4.4/examples/tutorials/IK_motion_planning_grasp.py). The initial arm posture also follows the conventional Panda posture used by the official [Genesis grasp environment example](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/manipulation/grasp_train.py). These references are credited to the Genesis contributors and retain their upstream Apache-2.0 licensing.

The course implementation adds its own colored-block and tray scene, restricted JSON action schema, Qwen image-and-language integration, protected-object rule, bounded motion phases, live browser workflow, execution records and independent position-based scoring. Robot motion uses privileged simulator coordinates for IK and control; it is not learned visual pose estimation. The upstream tutorial script itself, the Genesis library and the Panda asset files are not vendored into this source package.

The multistep implementation also supports project-created collision-enabled finger
extensions. `src/vla_trial/finger_extensions.py` derives a separate MJCF from the installed Panda
asset, adding primitive collision geometry to the robot's fingers while leaving
the upstream installation unchanged. This is a modified simulated gripper, not
the standard upstream Panda finger geometry. The upstream asset remains credited
to its original authors; the added extension geometry and generation logic are
project contributions. Blocks retain independent physical bodies. The source
package distributes the generator rather than the generated MJCF or upstream
meshes. This modification was used in the completed multistep simulation, but
has no physical-hardware validation.

The current generated pads measure 50 × 6 × 60 mm (width × thickness × length).
Their added mass and combined center of mass/inertia are represented in the
derived robot description; the change is not only a visual overlay. Geometry
and results from earlier calibration configurations remain separate.

The EGL identity implementation consults the Khronos specifications for [device DRM identifiers](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_drm.txt), [CUDA device handles](https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt) and [persistent device identifiers](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_persistent_id.txt). These are API references, not a copied EGL implementation.

## Distribution scope

The public source package contains project source, configuration templates, dependency specifications and documentation. It excludes the Python environment, model weights, tokenizer cache, robot meshes, upstream libraries, FFmpeg binaries, private machine configuration and research data. Dependencies and assets are obtained separately through their upstream distributions. The scene's blocks and trays are primitive shapes created by this project; rendered robot imagery depicts the credited Panda asset.

Anyone redistributing upstream code, weights, assets or binary dependencies separately must retain the license and notice material applicable to that distribution. This inventory does not replace those texts or assign a new license to third-party material. No endorsement by Qwen, Genesis, Franka Robotics or the course organizers is implied.

Additional source identity evidence is recorded in `docs/SOURCE_PROVENANCE.md`.
