# ATHLETE: Learning Reactive Humanoid Tennis via Trajectory-Guided Motion Matching

**English** | [简体中文](README.zh-CN.md)

Code for training, simulating and deploying the ATHLETE reactive whole-body
tennis policy on a Unitree G1 humanoid.

A time-adaptive **teacher** tracks dynamically corrected whole-body references
while adapting its motion phase to the contact deadline. A reactive **student**
is distilled from the teacher with online DAgger and privileged intent
alignment, then fine-tuned with PPO and tennis-specific rewards. At deployment
the student responds only to robot state, ball observations and the landing
target — no reference motions, online motion retrieval or explicit planner.

```
reference motions ──▶ teacher (phase + deadline) ──▶ student (DAgger + PPO) ──▶ ONNX ──▶ ROS 2 (sim / G1)
```

## Release status

The code is released in stages. This repository currently contains:

**Released**

- [x] Training code: time-adaptive teacher, TPPO student distillation with PPO
  fine-tuning, and full-flight fine-tuning tasks (`athlete/`)
- [x] Simulation and evaluation: play with native / browser (viser) viewers,
  ONNX student playback, evaluation scripts
- [x] Deployment stack for MuJoCo simulation and the real Unitree G1 (`deploy/`)
- [x] Released student policy, PyTorch checkpoint and exported ONNX
  (`deploy/policies/m14_11/`)
- [x] Frozen teacher checkpoint (`checkpoints/teacher_model_29999.pt`)
- [x] G1 + racket robot models and meshes (`robots/`)
- [x] 164 video-derived strike clips with strike-frame annotations
  (`data/djvkovic_npz_data_v5/`)
- [x] Trajectory-Guided Motion Matching (TGMM): motion-database construction,
  target-trajectory optimization and kinematic reference generation
  (`motion_matching/`)
- [x] Local G1 LAFAN1 CSV-to-NPZ converter (`scripts/prepare_lafan_locomotion.py`);
  third-party motion files are not redistributed
- [x] Unit tests for training and deployment

**Coming soon**

- [ ] Reference-motion datasets used for training (`rollout_1000epis`,
  `rollout_1000epis_relabelled`, ~430 MB each)
- [ ] Full-flight fine-tuned policy checkpoint

## Repository layout

| Path | Contents |
|---|---|
| `athlete/` | Python package: tasks, MDP terms, TPPO distillation, training / play scripts, tests |
| `athlete/src/athlete/goal_cond_tracking/` | Environment configs (`config/g1/`), MDP terms (`mdp/`), RL runner and TPPO (`rl/`) |
| `athlete/src/athlete/motion_sets/` | Motion library and motion-set TOMLs passed with `--motion-config` |
| `motion_matching/` | Trajectory-Guided Motion Matching: database builder, CT-OC target-trajectory optimizer, reference generator |
| `motion_db/` | Motion-matching resources (34-body augmentation mapping; generated databases are written here) |
| `deploy/` | ROS 2 Jazzy deployment stack (MuJoCo sim and real G1), see [Deployment](#deployment) |
| `deploy/policies/m14_11/` | Released student policy (`.pt` + `.onnx`) and its robot assets |
| `checkpoints/` | Frozen teacher checkpoint used for distillation and fine-tuning |
| `robots/` | G1 + racket MuJoCo models and meshes |
| `data/djvkovic_npz_data_v5/` | 164 video-derived G1 strike clips (97 forehand, 67 backhand) with strike-frame metadata |
| `scripts/` | Checkpoint export and evaluation utilities |

## Installation

Training uses Python 3.13 and [uv](https://docs.astral.sh/uv/); a CUDA 12.8
capable GPU is required.

```bash
git clone <this repository> athlete-humanoid-tennis
cd athlete-humanoid-tennis
uv sync
```

Run every training-side command from the repository root with `uv run`.
The deployment stack uses a separate ROS 2 environment (see [Deployment](#deployment)).

## Data

- **Strike clips** (`data/djvkovic_npz_data_v5/`): G1 motions reconstructed from
  ordinary tennis-practice video and retargeted to the robot, resampled to
  50 fps. Each `*.npz` stores joint and body states; the `*.json` sidecar holds
  the annotated `strike_frame`.
- **Reference-motion dataset** (`artifacts/rollout_1000epis*/`): 1,000
  dynamically corrected approach-and-strike references produced by
  trajectory-guided motion matching and simulated imitation rollouts. The
  teacher uses `rollout_1000epis`; the student uses the relabelled
  `rollout_1000epis_relabelled`. Both are distributed separately because of
  their size; place them under `artifacts/` (download link to be added).

## Reference generation (TGMM)

Trajectory-Guided Motion Matching composes reusable locomotion with
video-derived strikes into approach-and-strike references. The pipeline has
three stages; stages 1 and 2 are released in `motion_matching/`.

The 16 G1 locomotion motions required by stage 1 are **not included in this
repository**. They originate from [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
and were retargeted to G1 by the maintainers of the
[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset),
then converted from their 30 fps CSV files to 50 fps NPZ with Isaac Lab forward
kinematics for the original experiments. Ubisoft's [CC BY-NC-ND 4.0 license](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt)
does not grant permission to redistribute adapted motion data, so the converted
NPZ files remain local and are ignored by Git. Obtain the upstream data directly
and follow the applicable licenses. To make compatible local NPZ files with
the included MuJoCo converter (body velocities can differ from the original
Isaac Lab logs, so this is not a bitwise reproduction of the training input):

```bash
# In the ATHLETE repository root; install the Hugging Face CLI separately.
hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
  --include 'g1/run*.csv' --include 'g1/walk*.csv' \
  --local-dir /path/to/lafan1-retargeted

uv run python scripts/prepare_lafan_locomotion.py \
  --input-dir /path/to/lafan1-retargeted/g1 \
  --output-dir data/lafan_npz_data/runandwalk
```

The resulting directory should contain four `run*.npz` and twelve `walk*.npz`
files. Alternatively, copy previously converted NPZ files there. The database
builder below consumes this directory directly.

Install the optional dependency first:

```bash
uv sync --extra motion-matching
```

**1. Build the motion-matching database** from LAFAN1 locomotion
(`data/lafan_npz_data/runandwalk/`, 16 run/walk sequences retargeted to the G1)
and the strike clips:

```bash
uv run python motion_matching/build_official_mm_db.py \
  --input-dir data/lafan_npz_data/runandwalk \
  --strike-dir data/djvkovic_npz_data_v5 \
  --output-dir motion_db/official_tennis_runwalk_hand33
```

**2. Generate kinematic references.** For each randomly sampled ball, a CasADi
continuous-time optimal-control problem plans the root trajectory toward a
strike entry, motion matching follows it with locomotion clips, and the
selected strike clip is stitched in with inertialization:

```bash
uv run python motion_matching/tennis_official_mm_mj.py \
  --db_file motion_db/official_tennis_runwalk_hand33 \
  --config motion_matching/tennis_official_mm_config.yaml \
  --collect 1000 --headless
```

Episodes are written to `data/generated/collected_episodes/` as `ep_XXXX.npz`
(the 34-body layout of the strike clips) with a JSON sidecar holding the strike
frame and ball target. Run without `--collect --headless` (optionally with
`--interactive --ghost`) to explore the generator in the MuJoCo viewer.

**3. Dynamic correction by simulated imitation (not included).** A
motion-tracking policy trained in Isaac Lab tracks every kinematic reference in
physics simulation, and the executed motions are recorded at 50 Hz in the same
format, yielding physically consistent references. Collecting once keeps the
planned ball targets (`rollout_1000epis`, used by the teacher); collecting with
ball relabelling rewrites each target to the racket position actually reached
at the strike frame and recomputes it in the frame-0 pelvis frame
(`rollout_1000epis_relabelled`, used by the student), so that target labels
match the executed motion. The resulting datasets will be released (see
[Release status](#release-status)).

## Quick start: run the released policy

Play the released student in the training simulator with a browser viewer
(open the printed URL, by default <http://localhost:8080>):

```bash
uv run python -m athlete.scripts.play \
  Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RelativeSweet-BallDelay-RootLanding-Unitree-G1 \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml \
  --checkpoint-file deploy/policies/m14_11/model_36499.pt \
  --num-envs 1 --viewer viser --physical-ball True --fast-play False
```

Use `--viewer native` for a local MuJoCo window. To run the exported ONNX
policy through the ROS 2 stack instead, see [Deployment](#deployment).

## Training

All tasks are registered in `athlete/src/athlete/goal_cond_tracking/config/g1/`;
list them with `uv run list-envs`. Training logs go to `logs/rsl_rl/`.

### 1. Teacher

The teacher tracks the reference motions with phase-acceleration control and
deadline projection:

```bash
uv run train Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_first200_phase.toml \
  --env.scene.num-envs 8192 --agent.max-iterations 30000
```

The resulting teacher is provided as `checkpoints/teacher_model_29999.pt`.

### 2. Student distillation and PPO fine-tuning

The student is trained with TPPO (`athlete/src/athlete/goal_cond_tracking/rl/tppo.py`),
which combines online DAgger action distillation from the frozen teacher with
PPO on tennis rewards:

```bash
uv run train \
  Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RelativeSweet-BallDelay-RootLanding-Unitree-G1 \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml \
  --agent.algorithm.teacher-checkpoint checkpoints/teacher_model_29999.pt \
  --env.scene.num-envs 8192 --env.scene.terrain.num-envs 8192 \
  --agent.logger tensorboard --gpu-ids "[0]"
```

### 3. Full-flight fine-tuning

Continues from a trained student and keeps each episode running through the
complete post-strike ball flight, rewarding the actual landing point:

```bash
uv run train Mjlab-Tennis-SmallCourt-M14-11-FullFlight-ActualLanding-Std0-FootForce20-Unitree-G1 \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml \
  --resume-checkpoint deploy/policies/m14_11/model_36499.pt --agent.resume True \
  --agent.algorithm.teacher-checkpoint checkpoints/teacher_model_29999.pt \
  --env.scene.num-envs 8192 --env.scene.terrain.num-envs 8192 \
  --env.commands.motion.incoming-ball-failure-trajectory-probability 0.05 \
  --agent.max-iterations 30000 --agent.save-interval 500 \
  --agent.logger tensorboard --gpu-ids "[0]"
```

Variants with a return-home speed term and post-hit stillness are registered in
`full_flight_return_home.py`.

### Exporting a checkpoint to ONNX

```bash
uv run python scripts/export_intent_checkpoint.py --help
```

## Released policy

`deploy/policies/m14_11/` contains the final checkpoint of the ATHLETE reactive
student policy, trained to completion (36,499 iterations).

### Files

- `model_36499.pt`: original training checkpoint, including training state.
- `model_36499.onnx`: exported student, including the z_S intent encoder. It
  needs neither the teacher nor future reference inputs.
- `params/agent.yaml`, `params/env.yaml`: parameter snapshots saved by the
  training run (kept for reference).
- `robot/g1.xml`: the G1 + racket model used in training, with mesh paths made
  relative to this package.
- `robot/g1_with_racket_collision.xml`: additionally contains the racket
  collision geometry and nominal inertia that training creates at runtime.
- `robot/meshes/`: all 36 meshes referenced by the XML, including the racket.
- `robot/LICENSE.Unitree`: the BSD-3-Clause license of the Unitree G1 meshes.
- `manifest.json`: provenance, file sizes and SHA-256 checksums.

### Interface

- ONNX input `obs`: 631-D; output `actions`: 29-D.
- The input concatenates the current student observation (136-D), the state
  history (465-D) and the ball history (30-D).
- Includes the global root XYZ; the landing target is expressed in the
  real-time root frame; the sweet-spot velocity is defined relative to the pelvis.
- Control period 20 ms; training physics step 2.5 ms with decimation 8.
- At the end of training the landing-target curriculum std is 1.0 m. Ball
  observations were delayed during training by 20 ± 10 ms (clipped to
  0–50 ms); the perception buffer is not part of the ONNX graph.

The XML files describe the robot only: the court, ball launching, PD settings
and observation assembly are provided by the training and deployment code, and
randomized physics parameters are not baked into the XML. This package has not
been validated for unattended real-robot operation.

Verified: the ONNX outputs match the PyTorch student of this checkpoint; the
packaged base XML reproduces the original model arrays; the collision XML
compiles on its own.

## Deployment

Asynchronous ROS 2 stack that runs the exported ATHLETE student policy in MuJoCo
or on a Unitree G1. Simulation and hardware share the same control node,
observation builder and FSM; only the backend changes. Commands below are run
from the repository root; `source deploy/setup_deploy.zsh` changes into
`deploy/`, so the remaining paths in a session are relative to `deploy/`.

The released policy is `deploy/policies/m14_11/model_36499.onnx` (631-D observation,
29-D joint action, 50 Hz). Its deploy configuration is
`deploy/configs/g1_tppo_student_m14_11.yaml`.

### 0. Environment

The deploy stack uses its own Python 3.12 virtual environment on top of ROS 2
Jazzy (ROS 2 Jazzy pins Python 3.12, while training uses Python 3.13):

    bash deploy/setup_jazzy_env.sh      # one-time: create deploy/.venv
    source deploy/setup_deploy.zsh      # every shell; also cd's into deploy/

### 1. Simulation

Starts MuJoCo and the student policy. Keyboard: `a` = home, `x` = control,
`b` = damp, `q` = quit.

    source deploy/setup_deploy.zsh
    bash run_tppo.sh sim g1_tppo_student_m14_11.yaml

In simulation, landing targets are sampled per ball around (5, 0) m in the
start frame with the curriculum std stored in `sim_landing_target_sampling`; the
yellow ring of radius 0.5 m marks the target, and `deploy_robot/landing_target`
forwards it to the student, which converts it to the real-time root frame.
Policy ticks, the FSM and command timeouts use MuJoCo simulation time, so viewer
stalls do not consume the timeout budget.

### 2. Motion-capture bridge (real robot only)

`estimation/mocap/natnet_policy_bridge.py` converts NatNet `/rigid_bodies`
(`G1_pelvis` and `Ball` rigid bodies from an OptiTrack/Motive driver) into the
pelvis pose, ball position and ball velocity used by the policy. Measured ball
positions are published immediately; a 120 Hz timer fills in predictions only
when no new valid frame has arrived. The real-robot launcher already starts the
bridge, so run it on its own only for debugging:

    source deploy/setup_deploy.zsh
    python estimation/mocap/natnet_policy_bridge.py --config g1_tppo_student_m14_11.yaml

Visualize the bridge output in RViz (fixed frame `world`):

    rviz2 -d rviz/natnet_policy_bridge.rviz

### 3. Real robot

Starts the mocap bridge, the policy node and the Unitree hardware backend, and
allows LowCmd to be sent. The real-robot pipeline is read-only unless
`TPPO_ENABLE_REAL_ROBOT=YES` is set and the `enable` mode is passed.
Remote: `B` = damp, `A` = home, `Y` = calibrate pelvis, `X` = control;
keyboard `q` enters damp and exits.

    source deploy/setup_deploy.zsh
    export TPPO_ENABLE_REAL_ROBOT=YES
    bash run_tppo.sh real g1_tppo_student_m14_11.yaml <network-interface> enable --record

On the real robot the command watchdog uses a 100 ms wall-clock timeout. After a
hard tilt protection event, quit and restart rather than pressing `x` again.

**Safety.** Test in simulation first, keep the robot on a gantry for the first
real-robot runs, and keep the damp button within reach. This policy has not
been safety-validated for unattended operation.

### 4. Recording and replay

Append `--record` to a sim or real command to record robot state, policy inputs
and outputs, and mocap data under `deploy/recordings/<time>_<backend>_<config>/`.

Replay a recorded deployment in MuJoCo:

    python estimation/mocap/replay_deploy_bag_mujoco.py \
      recordings/<recording-dir> --config g1_tppo_student_m14_11.yaml

Replay only the recorded ball stream:

    python estimation/mocap/replay_bag_ball.py \
      recordings/<recording-dir> --config g1_tppo_student_m14_11.yaml

## Tests

```bash
uv run pytest athlete/tests
```

Deployment tests run in the deploy environment:
`source deploy/setup_deploy.zsh && python -m pytest tests`.

## Acknowledgements

We thank the authors of the following projects:

- [TaskNPoint](https://github.com/wernerb43/tasknpoint): goal-conditioned motion-tracking foundation of this code base
- [mjlab](https://github.com/mujocolab/mjlab): GPU-accelerated MuJoCo simulation and RL environment framework
- [Instinct-RL](https://github.com/project-instinct/instinct_rl): the TPPO algorithm adapted for teacher-student distillation
- [Motion-Matching](https://github.com/orangeduck/Motion-Matching) by Daniel Holden: the motion-matching core ported in `motion_matching/`
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): reinforcement-learning runner
- [Unitree](https://github.com/unitreerobotics/unitree_ros): G1 robot model and meshes
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset): source of the locomotion references

## License

See [`LICENSE.md`](LICENSE.md). Third-party components keep their own licenses;
in particular, the TPPO implementation adapted from Instinct-RL is licensed
under CC BY-NC 4.0, which restricts commercial use.
