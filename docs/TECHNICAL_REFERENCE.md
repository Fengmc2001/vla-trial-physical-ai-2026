# Technical reference

The project overview and current four-page report are linked from [README](../README.md). This page contains the detailed architecture, environment setup and execution procedure.

## Task and architecture

The scene is created by the program. It contains a robot arm, blue/yellow/red blocks, a cyan tray A, and a purple tray B. The current instruction is preserved exactly in `configs/tasks/multistep_stack_mixed.json`:

> 青いブロックをトレーAへ、黄色いブロックをトレーBへ移してください。赤いブロックは動かさないでください。
> And then please push the blue block out of the A , and put the yellow block B on the block A.
> 最后把它们一起放到B区域中。一直都不要动红色那个块。

The intended sequence is: sort blue into A and yellow into B; physically push blue out of A; place yellow on blue; move the two blocks together into B while keeping their stack intact. Red must remain still throughout. Here “the block A” means the blue block previously placed in A.

The validated v2 scene uses 25 mm cubes and a modified Panda gripper; the saved task contract records the geometry and scoring bounds. The retained v1 demonstration used 4 cm cubes. Geometry is part of each task setting; this fixed-scene implementation does not establish handling of arbitrary object sizes or shapes.

Each v2 run also records its actual time step, substeps, no-slip solver iterations, friction model and per-finger grasp forces in `runtime-environment.json`. Stack placement uses bounded simulator-position feedback and gradual finger opening. The final report reads the recorded numerical settings; a later controller change cannot silently replace the settings of an earlier run.

V2 adds custom collision-enabled finger extensions to a generated Panda MJCF so both blocks can be contacted by the gripper. The validated pads measure 50 × 6 × 60 mm (width × thickness × length), adding 19.8 g per finger with combined mass/inertia. This geometry is a project modification, not the upstream standard Panda fingers. Runs record whether extensions are enabled and the grasp-height offsets. The installed upstream asset remains unchanged; the blocks remain separate physical bodies. The modified gripper has not been built or validated on hardware.

Qwen3-VL-8B-Instruct receives the actual scene image, the full instruction and descriptions of four predefined primitives: `pick_place`, `push_out`, `stack` and `move_stack`. It returns one JSON plan with object/target IDs and a final `done`; no answer plan is supplied in the model prompt. The strict validator permits repeated operations only in multistep mode, caps the plan at 12 actions, and rejects protected objects in every manipulated/support role. Genesis provides physical simulation, camera rendering, inverse kinematics, path planning and joint control. The currently implemented contact corridors and primitives are tailored to this fixed scene.

The controller uses **simulator ground-truth object coordinates** to construct grasp, push and placement motions. The programmer defines the physical primitives; the VLM chooses their IDs and order. This is a VLM planner plus conventional controllers with privileged state, not a trained end-to-end VLA or a pixel-to-joint policy. No additional model training is performed. Object motion must come from physical simulation.

The v2 evaluator checks the recorded trajectory and actual execution timestamps against the independently declared contract: initial sorting, pushing outside A without a grasp-like lift, stacking, intact transport and a stable final stack in B. It checks red's displacement across all recorded samples. A model saying `done` does not establish success. Changed instructions are custom tasks and cannot inherit the preset contract's success label. One fixed mixed Japanese/English/Chinese example is not a multilingual generalization benchmark. The earlier v1 contract remains in `configs/tasks/minimal_sort_ja.json`; results from the two tasks are not pooled.

## Install and inspect without running a robot

The dependency lock targets Linux x86_64 and Python 3.11, with PyTorch 2.8.0 and CUDA 12.8 wheels. Use an independent project environment:

```bash
bash scripts/bootstrap.sh
bash scripts/cpu_check.sh
bash scripts/start_preview.sh
```

The installer does not execute inference or physics. The ordinary preview on `127.0.0.1:7866` shows metadata, a schematic and handwritten schema examples. It cannot start a GPU job and is not task-execution evidence. Default configuration stays disabled.

To reuse an existing original Qwen snapshot without copying weights, create the ignored `configs/local.json`:

```json
{"model": {"path": "YOUR_EXISTING_ORIGINAL_QWEN_SNAPSHOT"}}
```

The tested model revision is `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. A new host still needs its own model-load and inference validation; metadata checks alone are insufficient. Weights, research adapters, datasets and model caches are not distributed with this code.

## Watch the actual simulation and enter instructions

On the configured execution host:

```bash
bash scripts/start_live.sh
```

This starts a CPU-only browser service on `127.0.0.1:7867`, limited to two hours. In another terminal on your own computer, forward that loopback port through an existing SSH connection:

```bash
ssh -N -S none -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:17867:127.0.0.1:7867 YOUR_CONFIGURED_SSH_ALIAS
```

Open `http://127.0.0.1:17867` locally. The page shows real camera frames, the current stage, and finalized recordings. Its instruction box accepts text; **填入当前课题指令** restores the selected preset. Private `site.json` selects v2 with `active_task: "configs/tasks/multistep_stack_mixed.json"`; absent that field the older v1 preset remains the default. The **执行任务** button becomes available only after the selected task is validated, authorization is configured, and no other course run holds the execution lock. Each run resets the scene and launches a separately admitted process. The webpage itself does not import the model or simulator. Closing a tab does not stop a running task; the stop button can stop only a task launched by that interface instance.

See [中文使用说明](USER_GUIDE_ZH.md) for the scene, supported commands and status meanings. No public browser share link or full remote desktop is required.

## Configure an allocated execution host

Installation does not authorize GPU execution. Both modes need a private, ignored `configs/site.json` with the actual host, operator allocation reference, current date, full GPU UUIDs and matching PCI identities.

| Admission mode | Intended allocation | Validation status |
|---|---|---|
| `shared_adapter` | Approved co-running under a site's existing policy, task registry and shared locks | Used for the validated v1 and v2 demonstrations; depends on a private site policy library |
| `dedicated_exclusive` | Two distinct GPUs already allocated exclusively to the operator | Public implementation with 16 passing CPU tests; not yet validated on hardware |

For a new host with an existing exclusive allocation, `configs/site.example.json` provides the portable mode's configuration fields. It ships with `dedicated_exclusive_allocation_confirmed=false`, `interactive_enabled=false` and invalid placeholders. Copy it to the ignored site configuration without replacing an existing setup, then fill in the allocation information. The confirmation flag records permission obtained elsewhere; it does not reserve a GPU or grant access. The current full runner needs **two distinct allocated GPUs**, because the simulator remains alive while a separate process runs inference. Dedicated mode rejects every existing compute process, including the operator's own; only a verified system Xorg graphics process may remain. It does not require the private research-policy library and must not be used as a shortcut for shared-server co-running.

The existing shared mode in `src/vla_trial/admission.py` uses the site's `full_snapshot`, `proc_identity`, task registrations and lock layout, plus a checked SHA-256 of that policy library. It recognizes specific job types and therefore needs review and adaptation at another site. The private policy library and actual site configuration are excluded from source export.

Both modes perform fresh resource and process checks, a 30-second recheck, a final check before CUDA, and renderer-device binding. Current conservative per-process budgets are **18 GiB for inference and 6 GiB for simulation**, with at least 10% of the selected GPU's memory left free. Admission additionally requires 64 GiB available host RAM and 32 GiB free project storage; these are this runner's gates, not universal model requirements. A sampled process watcher records peaks and requests shutdown on a wrong device or exceeded budget; it is not a hard allocation limit. Account separately for model-weight storage on disk.

Keep interactive execution disabled until the camera, controller and model work on the new host. After confirming allocation and configuring admission, use these bounded stages for validation:

```bash
# Actual scene/camera validation, without model inference or task execution.
timeout --signal=TERM --kill-after=20s 1500s nice -n 19 \
  .venv/bin/python -m vla_trial.execute --stage preview \
  --project-root "$PWD" --wall-seconds 1440

# Current v2 task: actual image -> Qwen -> ordered physical execution.
timeout --signal=TERM --kill-after=20s 1500s nice -n 19 \
  .venv/bin/python -m vla_trial.execute --stage full \
  --project-root "$PWD" --wall-seconds 1440 \
  --task configs/tasks/multistep_stack_mixed.json
```

For the earlier v1 task, select `--task configs/tasks/minimal_sort_ja.json`. The `control` stage uses a handwritten controller-validation plan for the selected task; `infer` performs model-only inference on a supplied image. Neither alone proves an AI-controlled robot task. Runs write camera observations, plans, actual action milestones, trajectories, evaluation, videos and private resource receipts under ignored `outputs/`. Do not publish that directory wholesale: internal receipts can contain machine and process details. The public repository can be inspected and CPU-tested independently; the complete GPU workflow has not been demonstrated on a fresh, unrelated machine.

