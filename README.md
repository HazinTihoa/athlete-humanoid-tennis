# ATHLETE: Learning Reactive Humanoid Tennis via Trajectory-Guided Motion Matching

**English** | [简体中文](README.zh-CN.md)

[Project page & interactive demo](https://hazintihoa.github.io/ATHLETE/) ·
[Paper](https://arxiv.org/abs/2610.13572)

Training and deployment code for reactive whole-body tennis on the Unitree G1.

## Data downloads

### Source motions

- **G1-retargeted tennis strikes:** [164 NPZ clips and JSON annotations](data/djvkovic_npz_data_v5/) — 97 forehand, 67 backhand. Human motions are reconstructed from tennis videos using [GVHMR](https://github.com/zju3dv/GVHMR), then retargeted to the G1 using [GMR](https://github.com/YanjieZe/GMR). Included in this repository at `data/djvkovic_npz_data_v5/`.
- **G1-retargeted LAFAN1 locomotion:** [download from Hugging Face](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1). Use the 16 `run*.csv` / `walk*.csv` files and convert them with `scripts/prepare_lafan_locomotion.py` into `data/lafan_npz_data/runandwalk/`. The upstream data are provided separately by their maintainers.

### Generated motions

- **Student references:** [download from Google Drive](https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K?usp=sharing). TGMM-generated approach-and-strike motions recorded from mimic policy rollouts, with relabelled ball targets. Place the NPZ/JSON files directly under `artifacts/rollout_1000epis_relabelled/`.

## 1. Environment setup

- **Deployment simulation:** Ubuntu 24.04, system Python 3.12, ROS 2 Jazzy and [uv](https://docs.astral.sh/uv/getting-started/installation/). Install [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) first; `/opt/ros/jazzy/setup.bash` must exist. The windowed demo needs a desktop display.
- **Training:** a separate Python 3.13 environment and a CUDA 12.8-capable NVIDIA GPU. It does not use the deployment environment.

Set up the simulation environment:

```bash
git clone https://github.com/HazinTihoa/athlete-humanoid-tennis.git
cd athlete-humanoid-tennis
uv venv --no-project --python /usr/bin/python3 deploy/.venv
uv pip install --python deploy/.venv/bin/python -r deploy/requirements-jazzy.txt
```

Run subsequent commands from this repository root. The simulation environment needs no Unitree hardware SDK and no reference-motion dataset.

For training only, also install the locked project dependencies:

```bash
uv python install 3.13
uv sync --frozen
```

## 2. Quick demo: simulation deployment

Run the released Student ONNX policy in the MuJoCo ROS 2 simulator:

```bash
bash -c 'source /opt/ros/jazzy/setup.bash; bash deploy/run_tppo.sh sim g1_tppo_student_m14_11.yaml'
```

In the terminal, press **a** to enter home, wait at least 3 seconds, then press **x** to enable policy control. Balls are launched automatically. Press **b** to damp or **q** to quit. The policy is `deploy/policies/m14_11/model_36499.onnx` (631 observation values, 29 joint actions, 50 Hz).

Without a desktop display, use this 8-second startup check; it launches the simulator in damp mode and exits automatically:

```bash
bash -c 'source /opt/ros/jazzy/setup.bash; bash deploy/run_tppo.sh sim g1_tppo_student_m14_11.yaml --headless --duration-s 8'
```

## 3. Training workflow

1. **Generate references:** use `motion_matching/` to build the TGMM database and collect kinematic approach-and-strike motions. The 164 strike clips are included; the 16 G1 LAFAN1 run/walk clips must be obtained separately and converted using `scripts/prepare_lafan_locomotion.py`. Physical correction produces the Teacher and Student reference datasets.
2. **Train the Teacher:** phase-adaptive tracking of the generated references.
3. **Train the Student:** online DAgger distillation from the Teacher, followed by PPO fine-tuning. Full-flight task configurations support subsequent landing-based fine-tuning.

**Reference data:**

- **Student — [download from Google Drive](https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K?usp=sharing):** `rollout_1000epis_relabelled` contains generated approach-and-strike motions recorded from mimic policy rollouts in physics simulation, with relabelled ball targets. Place the NPZ/JSON files directly under `artifacts/rollout_1000epis_relabelled/`. The Student command below uses `ep_0000`–`ep_0199` and the included frozen Teacher checkpoint.
- **Teacher — not yet published:** Teacher retraining requires NPZ/JSON pairs `ep_0000`–`ep_0199` under `artifacts/rollout_1000epis/` with the original planned targets. The relabelled Student dataset does not replace these Teacher labels.

TOML files specify dataset paths; they do not contain motion data. The simulation demo above requires neither dataset.

Teacher:

```bash
uv run train Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_first200_phase.toml \
  --env.scene.num-envs 8192 --agent.max-iterations 30000 \
  --agent.logger tensorboard --gpu-ids "[0]"
```

Student (uses the included frozen Teacher checkpoint):

```bash
uv run train Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RelativeSweet-BallDelay-RootLanding-Unitree-G1 \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml \
  --agent.algorithm.teacher-checkpoint checkpoints/teacher_model_29999.pt \
  --env.scene.num-envs 8192 --env.scene.terrain.num-envs 8192 \
  --agent.logger tensorboard --gpu-ids "[0]"
```

The commands above use 8,192 parallel environments; reduce the environment counts to fit GPU memory. Logs and checkpoints are written under `logs/rsl_rl/`. Full training is not part of the quick demo.

## 4. Deployment workflow

- **Simulation:** use the released ONNX model and `deploy/configs/g1_tppo_student_m14_11.yaml` with the quick demo above. To export a new checkpoint, inspect the supported arguments with `uv run python scripts/export_intent_checkpoint.py --help`.
- **Real G1:** the same policy/control configuration uses the hardware backend and external motion capture for robot and ball state. ROS 2 Jazzy, the Unitree SDK wrapper, a configured robot network and the NatNet bridge are required. The SDK wrapper is not bundled, so hardware setup is not complete from a fresh clone.
- **Recording:** add `--record` to the simulation command; this additionally requires ROS 2 rosbag2 with MCAP support. Recordings are saved under `deploy/recordings/`.

## Release status

### Published

- [x] **Training code:** Teacher, Student distillation and PPO fine-tuning (`athlete/`).
- [x] **Deploy code:** MuJoCo simulation and G1 deployment (`deploy/`).
- [x] **TGMM:** database construction and kinematic reference generation (`motion_matching/`).
- [x] **G1-retargeted tennis dataset:** 164 NPZ clips with JSON annotations — 97 forehand and 67 backhand (`data/djvkovic_npz_data_v5/`).
- [x] **Generated Student reference dataset:** mimic policy rollouts with relabelled ball targets — [Google Drive](https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K?usp=sharing).
- [x] **Student checkpoint and ONNX:** `model_36499.pt` and `model_36499.onnx` (`deploy/policies/m14_11/`).

### TODO

- [ ] **Generated Teacher reference dataset:** `rollout_1000epis`, 1,000 motions with the original planned targets.
- [ ] **Detailed reproduction guide:** environment setup, data preparation, TGMM, training, evaluation and deployment.

## Acknowledgements

We thank the authors of the following projects:

- [GVHMR](https://github.com/zju3dv/GVHMR): human motion reconstruction from the source tennis videos
- [GMR](https://github.com/YanjieZe/GMR): retargeting the reconstructed human motions to the Unitree G1

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
