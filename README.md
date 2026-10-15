# ATHLETE: Learning Reactive Humanoid Tennis via Trajectory-Guided Motion Matching

**English** | [简体中文](README.zh-CN.md)

[Project page & interactive demo](https://hazintihoa.github.io/ATHLETE/) ·
[Paper](https://arxiv.org/abs/2610.13572)

Training and deployment code for reactive whole-body tennis on the Unitree G1.

## Data downloads

### Source motions

- **G1-retargeted tennis strikes:** [164 NPZ clips and JSON annotations](data/djvkovic_npz_data_v5/) — 97 forehand, 67 backhand. Human motions are reconstructed from tennis videos using [GVHMR](https://github.com/zju3dv/GVHMR), then retargeted to the G1 using [GMR](https://github.com/YanjieZe/GMR). Included in this repository at `data/djvkovic_npz_data_v5/`.

Original dataset replay: three forehand and three backhand clips, shown at 1× speed. These are kinematic replays of the 164-clip source dataset, not mimic-policy rollouts.

| Forehand (`fh_01`, `fh_32`, `fh_64`) | Backhand (`bh_01`, `bh_22`, `bh_45`) |
| --- | --- |
| [![Forehand replay](media/strike-replays/forehand.gif)](media/strike-replays/forehand.mp4) | [![Backhand replay](media/strike-replays/backhand.gif)](media/strike-replays/backhand.mp4) |

[Forehand MP4](media/strike-replays/forehand.mp4) · [Backhand MP4](media/strike-replays/backhand.mp4)

- **G1-retargeted LAFAN1 locomotion:** [download from Hugging Face](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1). Use the 16 `run*.csv` / `walk*.csv` files and convert them with `scripts/prepare_lafan_locomotion.py` into `data/lafan_npz_data/runandwalk/`. The upstream data are provided separately by their maintainers.

### Generated motions

- **Student references:** [download from Google Drive](https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K?usp=sharing). TGMM-generated approach-and-strike motions recorded from mimic policy rollouts, with relabelled ball targets. Place the NPZ/JSON files directly under `artifacts/rollout_1000epis_relabelled/`.

## Demonstration videos

Clips from the project page: the homepage rally and mouse-drag ball interaction in simulation. Click a preview to play the full MP4.

| Real-world rally | Interactive simulation |
| --- | --- |
| [![Real-world rally](media/project-demos/real-rally.gif)](https://hazintihoa.github.io/ATHLETE/assets/video/hero-7strikes-1080.mp4) | [![Mouse-drag ball interaction in simulation](media/project-demos/simulation-drag.gif)](https://hazintihoa.github.io/ATHLETE/assets/video/sim-rally1.mp4) |

## 1. Environment setup

The Quick Demo uses the same Python environment as training: Python 3.13.14 and the dependencies pinned in `uv.lock`. It does not require ROS 2, a Unitree SDK, or the separate real-robot deployment environment. Tested locally on Ubuntu 24.04 with an NVIDIA GPU; the windowed demo needs a desktop display. The ONNX policy runs on CPU; incoming-ball planning uses the training environment's Torch/Warp stack.

```bash
git clone https://github.com/HazinTihoa/athlete-humanoid-tennis.git
cd athlete-humanoid-tennis
uv python install 3.13.14
uv venv --python 3.13.14 .venv
uv sync --frozen
```

Run subsequent commands from the repository root. `uv run` automatically uses `.venv`; manual activation is not needed. Real-robot deployment is configured separately (see the deployment TODO below).

## 2. Quick demo: MuJoCo without ROS

The ONNX model and robot assets are included. No reference-motion download, LAFAN1 dataset, or Teacher checkpoint is needed. G1 starts from a fixed standing pose, and physical incoming balls are sampled near the robot without motion matching.

```bash
uv run python scripts/demo_mujoco.py
```

The included **GPU3-2 Full-flight + return-home-speed** model, `deploy/policies/gpu3_2_model45000/model_45000.onnx`, runs at 50 Hz with 631 observation values and 29 joint actions. The robot starts under policy control and balls launch automatically. The landing target is 5 m forward in the startup frame; the policy receives it in the current root frame and receives pelvis-relative racket velocity, matching this model's training configuration. Close the MuJoCo window to stop.

For a 20-second headless simulation that also executes policy control:

```bash
uv run python scripts/demo_mujoco.py --headless --duration-s 20 --no-realtime
```

For manual feeding, run `uv run python scripts/demo_mujoco.py --manual`, then click a robot-side first-bounce point in the separate court plot. This mode selects a bounce point; it is not the mouse-drag interface shown in the project-page video. Use `--onnx-policy-file PATH` to explicitly select another compatible Student model.

## 3. Training workflow

- [ ] TODO: training instructions.

## 4. Deployment workflow

- [ ] TODO: deployment instructions.

## Release status

### Published

- [x] **Training code:** Teacher, Student distillation and PPO fine-tuning (`athlete/`).
- [x] **Deploy code:** MuJoCo simulation and G1 deployment (`deploy/`).
- [x] **TGMM:** database construction and kinematic reference generation (`motion_matching/`).
- [x] **G1-retargeted tennis dataset:** 164 NPZ clips with JSON annotations — 97 forehand and 67 backhand (`data/djvkovic_npz_data_v5/`).
- [x] **Generated Student reference dataset:** mimic policy rollouts with relabelled ball targets — [Google Drive](https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K?usp=sharing).
- [x] **Student checkpoint and ONNX:** `model_36499.pt` and `model_36499.onnx` (`deploy/policies/m14_11/`).
- [x] **Quick Demo ONNX:** GPU3-2 Full-flight `model_45000.onnx`, saved training configurations and checksums (`deploy/policies/gpu3_2_model45000/`). Its source `.pt` checkpoint is not included.

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
