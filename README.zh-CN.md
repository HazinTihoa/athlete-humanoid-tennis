# ATHLETE：基于轨迹引导运动匹配的人形机器人反应式网球学习

[English](README.md) | **简体中文**

[项目网页与交互演示](https://hazintihoa.github.io/ATHLETE/) · [论文](https://arxiv.org/abs/2610.13572)

Unitree G1 反应式全身网球策略的训练与部署代码。

## 数据下载

### 原始动作数据

- **G1 重定向网球击球动作：**[164 个 NPZ 及 JSON 标注](data/djvkovic_npz_data_v5/)，正手 97 条、反手 67 条。先用 [GVHMR](https://github.com/zju3dv/GVHMR) 从网球视频重建人体动作，再用 [GMR](https://github.com/YanjieZe/GMR) 重定向到 G1。已包含在仓库的 `data/djvkovic_npz_data_v5/` 中。
- **G1 重定向 LAFAN1 走跑动作：**[Hugging Face 下载](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1)。使用其中 16 个 `run*.csv`／`walk*.csv` 文件，经 `scripts/prepare_lafan_locomotion.py` 转换到 `data/lafan_npz_data/runandwalk/`。该上游数据由原维护者单独提供。

### 生成动作数据

- **Student 参考动作：**[Google Drive 下载](https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K?usp=sharing)。TGMM 生成的“移动 + 击球”动作经过 mimic policy rollout 后记录，并重新标注球目标。将 NPZ／JSON 文件直接放入 `artifacts/rollout_1000epis_relabelled/`。

## 1. 环境配置

- **部署仿真：**Ubuntu 24.04、系统 Python 3.12、ROS 2 Jazzy 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。请先安装 [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)，确保 `/opt/ros/jazzy/setup.bash` 存在。有窗口 Demo 需要桌面显示环境。
- **训练：**独立的 Python 3.13 环境，以及支持 CUDA 12.8 的 NVIDIA GPU；不使用部署环境。

配置仿真环境：

```bash
git clone https://github.com/HazinTihoa/athlete-humanoid-tennis.git
cd athlete-humanoid-tennis
uv venv --no-project --python /usr/bin/python3 deploy/.venv
uv pip install --python deploy/.venv/bin/python -r deploy/requirements-jazzy.txt
```

后续命令均在仓库根目录执行。纯仿真不需要 Unitree 硬件 SDK，也不需要参考动作数据集。

仅进行训练时，额外安装锁定的项目依赖：

```bash
uv python install 3.13
uv sync --frozen
```

## 2. 快速 Demo：仿真部署

在 MuJoCo ROS 2 仿真器中运行已发布的 Student ONNX 策略：

```bash
bash -c 'source /opt/ros/jazzy/setup.bash; bash deploy/run_tppo.sh sim g1_tppo_student_m14_11.yaml'
```

在终端按 **a** 进入 home，等待至少 3 秒，再按 **x** 开启策略控制；仿真会自动发球。按 **b** 阻尼，按 **q** 退出。使用的模型为 `deploy/policies/m14_11/model_36499.onnx`，观测维度 631、关节动作维度 29、控制频率 50 Hz。

没有桌面显示时，可运行以下 8 秒启动检查；仿真保持阻尼状态，到时自动退出：

```bash
bash -c 'source /opt/ros/jazzy/setup.bash; bash deploy/run_tppo.sh sim g1_tppo_student_m14_11.yaml --headless --duration-s 8'
```

## 3. 训练流程

- [ ] TODO：补充训练说明。

## 4. 部署流程

- [ ] TODO：补充部署说明。

## 发布状态

### 已发布

- [x] **训练代码：**Teacher、Student 蒸馏与 PPO 微调（`athlete/`）。
- [x] **部署代码：**MuJoCo 仿真与 G1 实机部署（`deploy/`）。
- [x] **TGMM：**数据库构建与运动学参考生成（`motion_matching/`）。
- [x] **G1 重定向网球数据集：**164 个 NPZ 及 JSON 标注，正手 97 条、反手 67 条（`data/djvkovic_npz_data_v5/`）。
- [x] **生成的 Student 参考数据：**经过 mimic policy rollout，并重新标注球目标 — [Google Drive 下载](https://drive.google.com/drive/folders/1CWTzU1l1Bo3vszM7OkWS0bBAl8Cbif2K?usp=sharing)。
- [x] **Student checkpoint 与 ONNX：**`model_36499.pt` 和 `model_36499.onnx`（`deploy/policies/m14_11/`）。

### TODO

- [ ] **生成的 Teacher 参考数据集：**`rollout_1000epis`，1000 条动作，保留原始规划球目标。
- [ ] **详细复现说明：**环境配置、数据准备、TGMM、训练、评估与部署。

## 致谢

感谢以下项目的作者：

- [GVHMR](https://github.com/zju3dv/GVHMR)：从源网球视频重建人体动作
- [GMR](https://github.com/YanjieZe/GMR)：将重建的人体动作重定向到 Unitree G1

- [TaskNPoint](https://github.com/wernerb43/tasknpoint)：本项目目标条件动作跟踪部分的基础
- [mjlab](https://github.com/mujocolab/mjlab)：GPU 加速的 MuJoCo 仿真与强化学习环境框架
- [Instinct-RL](https://github.com/project-instinct/instinct_rl)：Teacher–Student 蒸馏所改编的 TPPO 算法
- [Motion-Matching](https://github.com/orangeduck/Motion-Matching)（Daniel Holden）：`motion_matching/` 中移植的运动匹配核心
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)：强化学习训练框架
- [Unitree](https://github.com/unitreerobotics/unitree_ros)：G1 机器人模型与网格
- [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)：走跑参考动作的数据来源

## 许可证

见 [`LICENSE.md`](LICENSE.md)。第三方组件保留各自的许可证；其中改编自 Instinct-RL 的 TPPO 实现采用
CC BY-NC 4.0 许可证，不可用于商业用途。
