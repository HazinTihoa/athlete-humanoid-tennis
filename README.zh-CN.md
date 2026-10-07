# ATHLETE：基于轨迹引导运动匹配的人形机器人反应式网球学习

[English](README.md) | **简体中文**

本仓库包含在 Unitree G1 人形机器人上训练、仿真和部署 ATHLETE 反应式全身网球策略的代码。

一个**时间自适应的 Teacher** 跟踪经过动力学修正的全身参考动作，同时根据击球截止时间调整动作相位。
**反应式 Student** 通过在线 DAgger 和特权意图对齐从 Teacher 蒸馏得到，再用 PPO 和网球任务奖励进行微调。
部署时，Student 只依赖机器人状态、来球观测和落点目标，不需要参考动作、在线动作检索或显式规划器。

```
参考动作 ──▶ Teacher（相位 + 截止时间）──▶ Student（DAgger + PPO）──▶ ONNX ──▶ ROS 2（仿真 / G1）
```

## 发布状态

代码分阶段发布，本仓库目前包含：

**已发布**

- [x] 训练代码：时间自适应 Teacher、基于 TPPO 的 Student 蒸馏与 PPO 微调、Full-flight 微调任务（`athlete/`）
- [x] 仿真与评估：本地窗口 / 浏览器（viser）可视化的 play、ONNX 策略回放、评估脚本
- [x] MuJoCo 仿真与 Unitree G1 实机的部署框架（`deploy/`）
- [x] 发布的 Student 策略：PyTorch checkpoint 和导出的 ONNX（`deploy/policies/m14_11/`）
- [x] 冻结的 Teacher checkpoint（`checkpoints/teacher_model_29999.pt`）
- [x] G1 + 球拍的机器人模型与网格（`robots/`）
- [x] 164 个从视频重建的击球片段及击球帧标注（`data/djvkovic_npz_data_v5/`）
- [x] 轨迹引导运动匹配（TGMM）：动作数据库构建、目标轨迹优化和运动学参考动作生成（`motion_matching/`）
- [x] 训练端和部署端的单元测试

**即将发布**

- [ ] 训练用的参考动作数据集（`rollout_1000epis`、`rollout_1000epis_relabelled`，各约 430 MB）
- [ ] LAFAN1 走跑数据的下载与转换脚本
- [ ] Full-flight 微调后的策略 checkpoint

## 目录结构

| 路径 | 内容 |
|---|---|
| `athlete/` | Python 包：任务、MDP 项、TPPO 蒸馏、训练 / play 脚本、测试 |
| `athlete/src/athlete/goal_cond_tracking/` | 环境配置（`config/g1/`）、MDP 项（`mdp/`）、RL runner 与 TPPO（`rl/`） |
| `athlete/src/athlete/motion_sets/` | 动作库，以及通过 `--motion-config` 传入的动作集 TOML |
| `motion_matching/` | 轨迹引导运动匹配：数据库构建、CT-OC 目标轨迹优化、参考动作生成 |
| `motion_db/` | 运动匹配资源（34 刚体增广映射；生成的数据库也写在这里） |
| `deploy/` | ROS 2 Jazzy 部署框架（MuJoCo 仿真和 G1 实机），见[部署](#部署) |
| `deploy/policies/m14_11/` | 发布的 Student 策略（`.pt` + `.onnx`）及其机器人资源 |
| `checkpoints/` | 蒸馏和微调使用的冻结 Teacher checkpoint |
| `robots/` | G1 + 球拍的 MuJoCo 模型与网格 |
| `data/djvkovic_npz_data_v5/` | 164 个从视频重建的 G1 击球片段（正手 97、反手 67），附击球帧信息 |
| `scripts/` | checkpoint 导出与评估工具 |

## 安装

训练使用 Python 3.13 和 [uv](https://docs.astral.sh/uv/)，需要支持 CUDA 12.8 的 GPU。

```bash
git clone <this repository> athlete-humanoid-tennis
cd athlete-humanoid-tennis
uv sync
```

所有训练端命令都在仓库根目录下用 `uv run` 执行。部署框架使用单独的 ROS 2 环境（见[部署](#部署)）。

## 数据

- **击球片段**（`data/djvkovic_npz_data_v5/`）：从普通网球练习视频中重建、重定向到机器人上的 G1 动作，
  重采样到 50 fps。每个 `*.npz` 保存关节和刚体状态，同名 `*.json` 记录标注的 `strike_frame`（击球帧）。
- **参考动作数据集**（`artifacts/rollout_1000epis*/`）：由轨迹引导运动匹配和仿真模仿 rollout 生成的
  1000 条经过动力学修正的“移动 + 击球”参考动作。Teacher 使用 `rollout_1000epis`，Student 使用重新标注的
  `rollout_1000epis_relabelled`。两者体积较大，会单独发布；下载后放在 `artifacts/` 下（下载链接待补充）。

## 参考动作生成（TGMM）

轨迹引导运动匹配（TGMM）把可复用的走跑动作和从视频重建的击球动作拼接成“移动 + 击球”参考动作。
整个流程分三步，第 1、2 步的代码在 `motion_matching/` 中。

先安装可选依赖：

```bash
uv sync --extra motion-matching
```

**1. 构建运动匹配数据库**，输入为 LAFAN1 走跑数据（`data/lafan_npz_data/runandwalk/`，16 段重定向到 G1 的
跑步 / 行走序列）和击球片段：

```bash
uv run python motion_matching/build_official_mm_db.py \
  --input-dir data/lafan_npz_data/runandwalk \
  --strike-dir data/djvkovic_npz_data_v5 \
  --output-dir motion_db/official_tennis_runwalk_hand33
```

**2. 生成运动学参考动作。**对每个随机采样的来球，先用 CasADi 求解一个连续时间最优控制问题，规划根节点
到击球切入点的轨迹；运动匹配用走跑片段跟随这条轨迹；最后通过惯性化（inertialization）混合切入选中的击球片段：

```bash
uv run python motion_matching/tennis_official_mm_mj.py \
  --db_file motion_db/official_tennis_runwalk_hand33 \
  --config motion_matching/tennis_official_mm_config.yaml \
  --collect 1000 --headless
```

生成的片段以 `ep_XXXX.npz`（与击球片段相同的 34 刚体格式）保存在 `data/generated/collected_episodes/`，
同名 JSON 记录击球帧和来球目标。去掉 `--collect --headless`（可加 `--interactive --ghost`）可以在 MuJoCo
窗口中交互式地查看生成过程。

**3. 通过仿真模仿进行动力学修正（本仓库不包含）。**用一个在 Isaac Lab 中训练的动作跟踪策略，在物理仿真里
跟踪每条运动学参考动作，并以 50 Hz 按相同格式记录实际执行的动作，得到物理上一致的参考动作。
采集两遍：一遍保留规划的来球目标（`rollout_1000epis`，供 Teacher 使用）；另一遍开启来球重新标注，把每个目标
改写为击球帧时球拍实际到达的位置，并在第 0 帧的骨盆坐标系下重新计算（`rollout_1000epis_relabelled`，
供 Student 使用），使目标标签与实际执行的动作一致。这两个数据集将会发布（见[发布状态](#发布状态)）。

## 快速开始：运行发布的策略

在训练用的仿真器中运行发布的 Student，并用浏览器查看（打开终端输出的地址，默认为 <http://localhost:8080>）：

```bash
uv run python -m athlete.scripts.play \
  Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RelativeSweet-BallDelay-RootLanding-Unitree-G1 \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml \
  --checkpoint-file deploy/policies/m14_11/model_36499.pt \
  --num-envs 1 --viewer viser --physical-ball True --fast-play False
```

使用 `--viewer native` 可以打开本地 MuJoCo 窗口。如果想通过 ROS 2 框架运行导出的 ONNX 策略，见[部署](#部署)。

## 训练

所有任务都注册在 `athlete/src/athlete/goal_cond_tracking/config/g1/` 中，可以用 `uv run list-envs` 列出。
训练日志保存在 `logs/rsl_rl/`。

### 1. Teacher

Teacher 通过相位加速度控制和截止时间投影来跟踪参考动作：

```bash
uv run train Mjlab-PhaseAccel-DeadlineProjection-MultiTarget-Tracking-Flat-Unitree-G1-Root-Pos-Obs \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_first200_phase.toml \
  --env.scene.num-envs 8192 --agent.max-iterations 30000
```

训练得到的 Teacher 已提供为 `checkpoints/teacher_model_29999.pt`。

### 2. Student 蒸馏与 PPO 微调

Student 使用 TPPO（`athlete/src/athlete/goal_cond_tracking/rl/tppo.py`）训练，它结合了从冻结 Teacher
进行的在线 DAgger 动作蒸馏，以及基于网球奖励的 PPO：

```bash
uv run train \
  Mjlab-Tennis-SmallCourt-IntentRefLatent-PPOPlusDistill005-GlobalRoot-RelativeSweet-BallDelay-RootLanding-Unitree-G1 \
  --motion-config athlete/src/athlete/motion_sets/motion_train_configs/rollout_1000epis_relabelled_first200_phase.toml \
  --agent.algorithm.teacher-checkpoint checkpoints/teacher_model_29999.pt \
  --env.scene.num-envs 8192 --env.scene.terrain.num-envs 8192 \
  --agent.logger tensorboard --gpu-ids "[0]"
```

### 3. Full-flight 微调

从训练好的 Student 继续训练，让每个 episode 持续到击球后球的完整飞行结束，并奖励实际落点：

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

带回位速度项和击球后静止项的变体注册在 `full_flight_return_home.py` 中。

### 导出 checkpoint 为 ONNX

```bash
uv run python scripts/export_intent_checkpoint.py --help
```

## 发布的策略

`deploy/policies/m14_11/` 包含 ATHLETE 反应式 Student 策略的最终 checkpoint，完整训练了 36,499 轮。

### 文件

- `model_36499.pt`：原始训练 checkpoint，包含训练状态。
- `model_36499.onnx`：导出的 Student，包含 z_S 意图编码器，不需要 Teacher 或未来参考动作输入。
- `params/agent.yaml`、`params/env.yaml`：训练时保存的参数快照（仅供参考）。
- `robot/g1.xml`：训练使用的 G1 + 球拍模型，网格路径已改为相对本目录。
- `robot/g1_with_racket_collision.xml`：额外包含训练时在运行中创建的球拍碰撞体和标称惯性。
- `robot/meshes/`：XML 引用的全部 36 个网格，包括球拍。
- `robot/LICENSE.Unitree`：Unitree G1 网格的 BSD-3-Clause 许可证。
- `manifest.json`：来源、文件大小和 SHA-256 校验值。

### 接口

- ONNX 输入 `obs` 为 631 维，输出 `actions` 为 29 维。
- 输入由当前 Student 观测（136 维）、状态历史（465 维）和来球历史（30 维）拼接而成。
- 包含全局根节点 XYZ；落点目标在实时根节点坐标系下表示；甜区速度相对骨盆定义。
- 控制周期 20 ms；训练物理步长 2.5 ms，decimation 为 8。
- 训练结束时落点课程的标准差为 1.0 m。训练中来球观测有 20 ± 10 ms 的延迟（截断在 0–50 ms），
  感知缓存不包含在 ONNX 计算图中。

XML 文件只描述机器人本身：球场、发球、PD 参数和观测拼接由训练和部署代码提供，随机化的物理参数也没有
写进 XML。该策略尚未针对无人值守的实机运行做过安全验证。

已验证：ONNX 输出与该 checkpoint 的 PyTorch Student 一致；打包后的基础 XML 与原模型的数组一致；
带碰撞体的 XML 可以独立编译。

## 部署

基于 ROS 2 的异步框架，在 MuJoCo 或 Unitree G1 上运行导出的 ATHLETE Student 策略。仿真和实机共用同一个
控制节点、观测构建和状态机，只是后端不同。下面的命令在仓库根目录执行；`source deploy/setup_deploy.zsh`
会切换到 `deploy/` 目录，之后同一终端里的路径都相对于 `deploy/`。

发布的策略为 `deploy/policies/m14_11/model_36499.onnx`（631 维观测，29 维关节动作，50 Hz），
对应的部署配置为 `deploy/configs/g1_tppo_student_m14_11.yaml`。

### 0. 环境

部署框架在 ROS 2 Jazzy 之上使用独立的 Python 3.12 虚拟环境（ROS 2 Jazzy 固定使用 Python 3.12，
训练使用 Python 3.13）：

    bash deploy/setup_jazzy_env.sh      # 只需一次：创建 deploy/.venv
    source deploy/setup_deploy.zsh      # 每个终端都要执行，同时会进入 deploy/

### 1. 仿真

启动 MuJoCo 和 Student 策略。键盘：`a` = 回到初始姿态，`x` = 进入控制，`b` = 阻尼，`q` = 退出。

    source deploy/setup_deploy.zsh
    bash run_tppo.sh sim g1_tppo_student_m14_11.yaml

仿真中，每个球的落点目标在起始坐标系 (5, 0) m 附近采样，标准差为 `sim_landing_target_sampling` 中保存的
课程值；半径 0.5 m 的黄色圆环标出目标，`deploy_robot/landing_target` 把它发送给 Student，Student 再
转换到实时根节点坐标系。策略推理、状态机和命令超时都使用 MuJoCo 仿真时间，因此可视化窗口卡顿不会消耗超时预算。

### 2. 动捕桥接（仅实机）

`estimation/mocap/natnet_policy_bridge.py` 把 NatNet 的 `/rigid_bodies`（来自 OptiTrack/Motive 驱动的
`G1_pelvis` 和 `Ball` 刚体）转换为策略使用的骨盆位姿、球位置和球速度。实测的球位置会立即发布；
120 Hz 定时器只在没有新的有效帧时补充预测值。实机启动脚本已经包含桥接，只有调试时才需要单独运行：

    source deploy/setup_deploy.zsh
    python estimation/mocap/natnet_policy_bridge.py --config g1_tppo_student_m14_11.yaml

在 RViz 中查看桥接输出（Fixed Frame 设为 `world`）：

    rviz2 -d rviz/natnet_policy_bridge.rviz

### 3. 实机

启动动捕桥接、策略节点和 Unitree 硬件后端，并允许发送 LowCmd。只有设置了 `TPPO_ENABLE_REAL_ROBOT=YES`
并传入 `enable` 模式时，实机流程才会真正发送命令，否则只读。
遥控器：`B` = 阻尼，`A` = 回到初始姿态，`Y` = 校准骨盆，`X` = 进入控制；键盘 `q` 进入阻尼并退出。

    source deploy/setup_deploy.zsh
    export TPPO_ENABLE_REAL_ROBOT=YES
    bash run_tppo.sh real g1_tppo_student_m14_11.yaml <network-interface> enable --record

实机上的命令看门狗使用 100 ms 的墙钟超时。触发严重倾斜保护后，请退出并重新启动，不要反复按 `x`。

**安全提示。**请先在仿真中测试；前几次实机运行时把机器人挂在吊架上，并让阻尼按键随时可按。
该策略尚未针对无人值守运行做过安全验证。

### 4. 录制与回放

在仿真或实机命令末尾加上 `--record`，会把机器人状态、策略输入输出和动捕数据录制到
`deploy/recordings/<时间>_<后端>_<配置名>/`。

在 MuJoCo 中回放录制的部署数据：

    python estimation/mocap/replay_deploy_bag_mujoco.py \
      recordings/<recording-dir> --config g1_tppo_student_m14_11.yaml

只回放录制的来球数据：

    python estimation/mocap/replay_bag_ball.py \
      recordings/<recording-dir> --config g1_tppo_student_m14_11.yaml

## 测试

```bash
uv run pytest athlete/tests
```

部署端的测试在部署环境中运行：`source deploy/setup_deploy.zsh && python -m pytest tests`。

## 致谢

感谢以下项目的作者：

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
