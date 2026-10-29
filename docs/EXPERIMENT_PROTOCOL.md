# Multistep task demonstration protocol — 2026-10-29

One formal v2 run completed on 2026-10-29 with actual Qwen inference and all seven
independently recomputed milestone checks passing. The frozen configuration had
one formal launch, one completion and no formal failures; ten earlier development
failures remain separate. This is a case demonstration, not an estimate of
general performance. The earlier v1 sorting result is retained separately.

The completed run recorded 1,202 trajectory samples over 120.0 seconds of
observed simulation time. Blue moved 175.989 mm during the push with a maximum
rise of 0.073 mm. Final stack horizontal offset was 0.082 mm and height difference
24.946 mm. Red's maximum recorded displacement was below 0.001 mm, at numerical
noise scale. These results are sampled simulator coordinates, not hardware
accuracy or proof of behavior between samples. Full wall time was 222.33 seconds;
measured process GPU peaks were 17,616 MiB for inference and 873 MiB for simulation.

## Fixed task and geometry

Use `configs/tasks/multistep_stack_mixed.json` and preserve its exact mixed
Japanese/English/Chinese instruction. Interpret its later reference to “block A”
as the blue block previously placed in A. The required sequence is:

1. Place blue in cyan tray A and yellow in purple tray B. These two initial
   placements may occur in either order, but both must be complete before pushing.
2. Physically push blue completely outside A; yellow remains in B.
3. Place yellow on top of blue outside A.
4. Move the existing stack together into B, retaining its geometry throughout.

Red remains protected in all stages and cannot be used as a support. The completed
v2 run used 25 mm cubes and 50 × 6 × 60 mm collision-enabled finger pads; the
separate v1 setting uses 4 cm cubes. The controller receives simulator ground-truth coordinates.
Qwen selects and orders predefined physical primitives rather than generating
continuous joint actions or learning a new VLA policy.

Freeze the task contract, geometry, controller settings and model configuration
before a formal run. If development changes geometry or control, preserve the
old run with its own configuration; do not relabel its outcome under new rules.
The saved run contract is authoritative for all numeric bounds and tolerances.
The completed run used a 0.01-second time step, four physics substeps, five
no-slip iterations, a pyramidal friction cone, 2 N per finger for single-block
grasping and 4 N for the pair, with 165 mm grasp-height offsets. Geometry and
control were frozen after a successful controller-only calibration and before
the first formal AI run.

Save the actual time step, physics substeps, no-slip solver iterations,
friction-cone model, cube size and normal/pair grasp force per finger in
`runtime-environment.json` under `simulation_settings`. These are calibrated
numerical/contact settings, not learned control parameters. The stack controller
uses simulator-position feedback with bounded correction count and travel, then
ramps finger opening to release slowly. Preserve the corresponding control
events; do not infer these settings from a later source default. The v2 report
requires the recorded values and checks cube size against the scored contract.
Calibration is specific to this scene and does not establish real-robot behavior.

When custom finger extensions are enabled, additionally record
`finger_extensions`, `grasp_offset_m` and `pair_grasp_offset_m`. Distinguish
the generated, collision-enabled finger geometry from upstream standard Panda
fingers. Do not modify the installed upstream asset or attach manipulated blocks
to the robot; blocks remain independent rigid bodies. Preserve the geometry
helper in the source manifest. A complete run with this modification is a
separate configuration from earlier standard-finger attempts. No physical
fabrication or hardware validation is established by this simulation.
Use the recorded `physical_finger_extensions` control event for the actual pad
dimensions and added mass; the report must not substitute the current helper's
constants for an older run. Current extension control closes gradually, settles,
then switches to force holding, and releases with a gradual opening ramp.

## Independent milestone evaluation

Record positions throughout motion at nominal 0.1-second simulation intervals;
gaps exceeding the contract limit (currently 0.2 seconds) invalidate scoring.
Record each actually executed primitive with its arguments, simulation-time
`started_s`, `ended_s` and `completed` state. These milestones must come from
execution, not from copying the model's plan. Include at least 0.5 seconds of
settling before each successful action end.

| Check | Required evidence |
|---|---|
| Initial placements | Each block settles in its declared tray; both are in place before pushing |
| Push out of A | Blue begins in A, moves horizontally, ends fully outside the tray footprint with clearance, and settles on the plane; its maximum center rise stays within the contract's push limit |
| Stack | Yellow is above blue with the contract's horizontal and height tolerances, outside A, and stable |
| Transport | Relative stack geometry holds across every recorded sample during the whole `move_stack` interval, including grasp and release |
| Final state | Both blocks settle inside B with blue below yellow and a stable stack |
| Protection | Red's maximum displacement from its initial position stays within the contract limit over the entire recorded trajectory |

`evaluation_multistep.py` checks the ordered milestones, complete trajectory
and independent contract. A lifted transfer cannot substitute for pushing; a
stack reconstructed only at the end cannot substitute for intact transport.
The final B positions alone do not establish that earlier stages were completed.
The evaluator measures sampled geometry; it does not independently prove contact
forces or rule out motion between recorded samples. Video and controller records
provide complementary evidence of the physical execution.

## Trials and claims

Classify a run's purpose before launch. Keep `preview`, handwritten `control`
and standalone `infer` development checks separate from formal `full` trials.
After controller calibration and the task freeze, record every formal launch,
including admission, inference, plan-validation and motion failures. Do not
discard failed attempts or retrospectively rename them as calibration.

Use the task identifier in any trial summary. V2 summaries must specify
`task_id: "multistep_stack_mixed_v1"` and give total, success, failure and
unscored counts, plus separate development notes. Never pool the v1 sorting and
v2 multistep denominators. Without a complete trial ledger, report the selected
run as a case demonstration instead of presenting a success rate. One fixed
mixed-language instruction is not evidence of general multilingual competence.
Custom-instruction UI runs cannot automatically inherit the preset task score.

The recorded ten development failures comprise eight controller calibration
attempts, one attempt through the older action schema, and one pre-execution
resource admission refusal. A later successful controller calibration and
standalone model check are development checks, not additional formal AI trials.

## Public evidence and artifact rebuilding

The sanitized `examples/validated_multistep_run/` bundle has been prepared and
independently rescored. Its model raw output, executed plan, observation hash,
trajectory, geometry events and runtime fields pass the report builder's gates.
The three-page Japanese PDF was built from this public copy. Repository
publication is a separate operation and is not established by this local check.

Keep original operational records private. After a complete real v2 run is
verified, prepare a separate sanitized `examples/validated_multistep_run/` copy;
do not overwrite `examples/validated_run/`, which identifies the earlier v1 run.
Retain numeric trajectory/action evidence and hashes unchanged. Remove host,
operator login, PID, GPU UUID, filesystem paths and unrelated process information
from public status/environment metadata; preserve the scientific configuration,
device model, measured memory peaks and timestamps needed to explain the run.

A complete public v2 artifact-rebuild bundle contains:

- Task contract, evaluation, trajectory, actual action milestones, action results
  and controller events.
- Initial/final PNG observations and their coordinate metadata, plus the original
  `simulation.mp4` used for the video.
- Executed plan, original model response, input-image SHA-256, model revision and
  generation settings.
- Sanitized status, event timestamps and runtime versions for both simulator and
  inference; source manifest and any task-specific trial summary.
- A short README identifying the task, any redactions, the selected outcome and
  what the bundle can reproduce.

Exclude admission receipts, resource-monitor dumps, private configuration,
shell logs and model weights. The existing v1 public example omits some inference
event metadata and the original MP4; it supports inspection/rescoring, not a full
video/PDF rebuild by itself. Check both evidence builders against the sanitized
v2 copy before claiming that the public bundle can rebuild its artifacts.

The source exporter includes only explicit public filenames under the two
versioned example directories, including the v2 original MP4. It independently
rescans the v2 score and rejects private machine/operator metadata. Optional
`--include-delivery` adds exactly the reviewed course-named MP4 and PDF to the
archive's `delivery/` folder after the public v2 bundle is ready. It does not
recursively copy raw output directories or publish the archive.

## Video and report

The completed local submission video is 70.5 seconds. It preserves the same
run's full 121-second raw recording at labeled 2× speed, with ten seconds of
introductory/result cards. Initialization and model waiting are disclosed
omissions. The final video passed full decoding and visual review after its title-font fix;
the exact Japanese/English/Chinese instruction is readable. All four pages of the current concise monochrome explanation were rendered and visually reviewed. These artifact checks do not mean YouTube or OmniCampus submission
has occurred.

Use one identified real run for the entire robot sequence. Introductory/result
stills are explanatory holds; initialization and model waiting may be omitted
with disclosure. State playback speed relative to recorded simulation time and
keep any speed-up within the course's permitted 2×. Do not stitch successful
stages from different runs into one apparent trial. The final MP4 must last
60–180 seconds; a planned duration is not a verified exported duration.

The Japanese PDF must describe the mixed instruction, predefined primitives,
privileged coordinates, frozen geometry, actual milestone results and remaining
limitations. Build it only from complete real evidence and independently
recompute the scores. Inspect every rendered page and the final video before
publishing. Local artifact creation, public publication and course submission
are separate states.

## Public release

The completed code, sanitized evidence, 70.5-second final video and current four-page monochrome explanation are published at https://github.com/Fengmc2001/vla-trial-physical-ai-2026. Public visibility and anonymous access were checked. The clean export passed 170 CPU tests; a fresh GPU-machine reproduction has not been claimed. The user has uploaded the video to https://www.youtube.com/watch?v=hS34eNkm5VM. OmniCampus submission remains pending while the report is reviewed; the video title still needs to match the account name.


## Monochrome academic report

The revised PDF retains the same genuine run and numerical results. It contains four numbered figures and three tables, with system architecture, actual grayscale before/after images, the recorded manipulation sequence, and plotted trajectories. Figure annotations project saved coordinates through the recorded fixed camera; crops and times are disclosed. No generated replacement scenes or new robot trials are used. The original report is retained locally and in Git history.


## Current concise explanation and additional repetition case

The current course PDF is four pages and is explained by the editable source in `report/`. It replaces the six-page academic-style report as the submission artifact. The retained detailed report is historical.

For the new instruction 抓起蓝色方块然后放下。重复三回, actual Qwen inference returned three identical blue-to-A `pick_place` actions followed by `done`. All three manipulations completed, and each had verified lift and placement. Each successor read the updated scene position, matching the previous result. This run is `custom_task_unscored` with `success=null`; it is not a second successful instance of the fixed five-action task. Core model/parser/execution/controller hashes match the earlier run; `admission.py` differs. An additional recorded response missing `done` was rejected before any `moving` phase. These selected examples do not estimate a general success rate.
