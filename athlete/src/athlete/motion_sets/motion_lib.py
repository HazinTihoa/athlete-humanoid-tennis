"""Motion library — canonical specs for every motion.

Add a new motion here and reference it by name in a motion_sets/*.toml file.
"""

import numpy as np

from athlete.goal_cond_tracking.mdp import MotionCfg, MotionGoalCfg

vel_ori_window = 0.004 # phase window before and after position window
kick_window_x = 0.100
kick_window_y = 0.100
kick_window_z = 0.100


def _collected_tennis_motion(name: str, *, forehand: bool) -> MotionCfg:
  """Template for collected episodes; ball_local and strike phase are per-file."""
  velocity_pitch = float(np.arctan2(1.0, 0.1))
  target_normal_orientation = (
    {"roll": 0.0, "pitch": velocity_pitch, "yaw": 0.0}
    if forehand
    else {"roll": 0.0, "pitch": float(np.pi - velocity_pitch), "yaw": float(np.pi)}
  )
  return MotionCfg(
    name=name,
    sampling_weight=1.0,
    probe_points=[("racket_sweet_spot", 0.5)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_sweet_spot",
        source_type="site",
        target_pos_std={"x": 0.2, "y": 0.3, "z": 0.3},
        target_phase_start=0.496,
        target_phase_end=0.504,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_sweet_spot",
        source_type="site",
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
        target_phase_start=0.48,
        target_phase_end=0.52,
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_sweet_spot",
        source_type="site",
        target_orientation_mean=target_normal_orientation,
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="z",
        target_phase_start=0.48,
        target_phase_end=0.52,
      ),
    ],
  )

MOTION_LIB: dict[str, MotionCfg] = {
  "collected_forehand": _collected_tennis_motion(
    "collected_forehand", forehand=True
  ),
  "collected_backhand": _collected_tennis_motion(
    "collected_backhand", forehand=False
  ),
  "right_kick": MotionCfg(
    name="right_kick",
    sampling_weight=1.0,
    probe_points=[("right_foot", 0.293)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_foot",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.6, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.290,
        target_phase_end=0.297,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.290 - vel_ori_window,
        target_phase_end=0.297 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.290 - vel_ori_window,
        target_phase_end=0.297 + vel_ori_window,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "left_kick": MotionCfg(
    name="left_kick",
    sampling_weight=1.0,
    probe_points=[("left_foot", 0.344)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_foot",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.0, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.340,
        target_phase_end=0.347,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.340 - vel_ori_window,
        target_phase_end=0.347 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.340 - vel_ori_window,
        target_phase_end=0.347 + vel_ori_window,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "right_far_kick": MotionCfg(
    name="right_far_kick",
    sampling_weight=1.0,
    probe_points=[("right_foot", 0.408)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_foot",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -1.2, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.404,
        target_phase_end=0.411,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.404 - vel_ori_window,
        target_phase_end=0.411 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.404 - vel_ori_window,
        target_phase_end=0.411 + vel_ori_window,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "left_far_kick": MotionCfg(
    name="left_far_kick",
    sampling_weight=1.0,
    probe_points=[("left_foot", 0.405)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_foot",
        source_type="site",
        target_pos_mean={"x": 1.0, "y": 0.6, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.401,
        target_phase_end=0.408,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.401 - vel_ori_window,
        target_phase_end=0.408 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.401 - vel_ori_window,
        target_phase_end=0.408 + vel_ori_window,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

########################### Tennis motions ###########################

  "forehand": MotionCfg(
    name="forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.310)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.6, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.306,
        target_phase_end=0.313,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.300,
        target_phase_end=0.320,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.306,
        target_phase_end=0.313,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "backhand": MotionCfg(
    name="backhand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.372)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.6, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.368,
        target_phase_end=0.375,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.360,
        target_phase_end=0.380,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.368,
        target_phase_end=0.375,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_forehand": MotionCfg(
    name="two_step_forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.399)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -1.2, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.394,
        target_phase_end=0.403,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.390,
        target_phase_end=0.410,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.394,
        target_phase_end=0.403,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_backhand": MotionCfg(
    name="two_step_backhand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.452)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 1.2, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.448,
        target_phase_end=0.455,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.440,
        target_phase_end=0.460,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.448,
        target_phase_end=0.455,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_forehand": MotionCfg(
    name="stepback_forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.337)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.1, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.333,
        target_phase_end=0.340,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.328,
        target_phase_end=0.345,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.333,
        target_phase_end=0.340,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_backhand": MotionCfg(
    name="stepback_backhand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.402)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.1, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.398,
        target_phase_end=0.405,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.390,
        target_phase_end=0.410,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.398,
        target_phase_end=0.405,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  ################# new tennis motions with steps ##################
  "one_step_forehand": MotionCfg(
    name="one_step_forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.402)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5589, "y": -0.4391, "z": 0.3129},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.398,
        target_phase_end=0.407,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.383,
        target_phase_end=0.422,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.383,
        target_phase_end=0.422,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "forehand_sidestep": MotionCfg(
    name="forehand_sidestep",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.429)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.4038, "y": 0.2925, "z": 0.3185},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.425,
        target_phase_end=0.433,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.410,
        target_phase_end=0.448,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.410,
        target_phase_end=0.448,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "forehand_two_sidestep": MotionCfg(
    name="forehand_two_sidestep",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.507)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.6079, "y": -1.4088, "z": -0.7081},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.503,
        target_phase_end=0.512,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.488,
        target_phase_end=0.527,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.488,
        target_phase_end=0.527,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "backhand_onehand": MotionCfg(
    name="backhand_onehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.321)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.74, "y": 0.03, "z": 0.2426},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.318,
        target_phase_end=0.324,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.303,
        target_phase_end=0.339,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.303,
        target_phase_end=0.339,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "backhand_onehand_step": MotionCfg(
    name="backhand_onehand_step",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.447)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5105, "y": 0.7421, "z": 0.1194},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.443,
        target_phase_end=0.451,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.428,
        target_phase_end=0.466,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.428,
        target_phase_end=0.466,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  ########################################################## Fast motions ###############################################################

  "forehand_fast": MotionCfg(
    name="forehand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.386)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.6, "z": 0.0},
        target_pos_std={"x": 0.20, "y": 0.30, "z": 0.30},
        target_phase_start=0.381,
        target_phase_end=0.390,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.381 - vel_ori_window,
        target_phase_end=0.390 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.381 - vel_ori_window,
        target_phase_end=0.390 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "backhand_fast": MotionCfg(
    name="backhand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.397)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.6, "z": 0.0},
        target_pos_std={"x": 0.20, "y": 0.30, "z": 0.30},
        target_phase_start=0.393,
        target_phase_end=0.400,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.393 - vel_ori_window,
        target_phase_end=0.400 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.393 - vel_ori_window,
        target_phase_end=0.400 + vel_ori_window,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_forehand_fast": MotionCfg(
    name="two_step_forehand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.427)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -1.2, "z": 0.0},
        target_pos_std={"x": 0.20, "y": 0.30, "z": 0.30},
        target_phase_start=0.423,
        target_phase_end=0.430,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.423 - vel_ori_window,
        target_phase_end=0.430 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.423 - vel_ori_window,
        target_phase_end=0.430 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_backhand_fast": MotionCfg(
    name="two_step_backhand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.452)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 1.2, "z": 0.0},
        target_pos_std={"x": 0.20, "y": 0.30, "z": 0.30},
        target_phase_start=0.448,
        target_phase_end=0.455,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.448 - vel_ori_window,
        target_phase_end=0.455 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.448 - vel_ori_window,
        target_phase_end=0.455 + vel_ori_window,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_forehand_fast": MotionCfg(
    name="stepback_forehand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.355)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.0, "z": 0.0},
        target_pos_std={"x": 0.20, "y": 0.30, "z": 0.30},
        target_phase_start=0.351,
        target_phase_end=0.358,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.351 - vel_ori_window,
        target_phase_end=0.358 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.351 - vel_ori_window,
        target_phase_end=0.358 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_backhand_fast": MotionCfg(
    name="stepback_backhand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.402)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.1, "z": 0.0},
        target_pos_std={"x": 0.20, "y": 0.30, "z": 0.30},
        target_phase_start=0.398,
        target_phase_end=0.405,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.398 - vel_ori_window,
        target_phase_end=0.405 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.398 - vel_ori_window,
        target_phase_end=0.405 + vel_ori_window,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  #########################################
  #                                       #
  #          box pickups to floor         #
  #                                       #
  #########################################


  "pickup_bench_to_floor": MotionCfg(
    name="pickup_bench_to_floor",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.412), ("left_palm", 0.412)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7876, "y": 0.1359, "z": -0.328},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.412 - vel_ori_window,
        target_phase_end=0.412 + vel_ori_window,
      ),
      # Left hand when box is grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.762, "y": 0.422, "z": -0.282},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.412 - vel_ori_window,
        target_phase_end=0.412 + vel_ori_window,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",
        target_pos_mean={"x": -0.026, "y": 0.286, "z": 0.046},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.02, "z": 0.02},
        target_phase_start=0.412,   # from when the box is grabbed
        target_phase_end=0.904,     # until the box is set down
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.405, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": 0.5, "z": 0.0}, # pushing on the box to the left (y positive)
        target_vel_std={"x": 0.0, "y": 0.2, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="left_palm",
        source_type="site",
        target_phase_start=0.405, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": -0.5, "z": 0.0}, # pushing on the box to the right (y negative)
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=3.85,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.359, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_orientation_mean={"roll": -0.444, "pitch": 0.531, "yaw": 0.121},  # this is the init frame orientation at the box grab frame
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.1},    # small std dev
        orientation_axis="y",
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=3.85,
        source_link="left_palm",
        source_type="site",
        target_phase_start=0.359, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_orientation_mean={"roll": -0.208, "pitch": 0.4220, "yaw": 0.232},  # this is the init frame orientation at the box grab frame
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.1},    # small std dev
        orientation_axis="y",
      ),
    ],
  ),

  "pickup_bench_to_floor_2": MotionCfg(
    name="pickup_bench_to_floor_2",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.338), ("left_palm", 0.338)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7732, "y": 0.1720, "z": 0.0325},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left hand when box is grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.6513, "y": 0.4940, "z": -0.0271},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",
        target_pos_mean={"x": -0.026, "y": 0.286, "z": 0.046},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.02, "z": 0.02},
        target_phase_start=0.338,   # from when the box is grabbed
        target_phase_end=0.9008,     # until the box is set down
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.338, # a bit before when the box was grabbed
        target_phase_end=0.9008, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": 0.5, "z": 0.0}, # pushing on the box to the left (y positive)
        target_vel_std={"x": 0.0, "y": 0.2, "z": 0.0},
      ),
    ],
  ),

  "pickup_bench_to_floor_3": MotionCfg(
    name="pickup_bench_to_floor_3",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.338), ("left_palm", 0.338)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7876, "y": 0.1359, "z": -0.328},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left hand when box is grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.762, "y": 0.422, "z": -0.282},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",
        target_pos_mean={"x": -0.026, "y": 0.286, "z": 0.046},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.02, "z": 0.02},
        target_phase_start=0.338,   # from when the box is grabbed
        target_phase_end=0.9008,     # until the box is set down
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.338, # a bit before when the box was grabbed
        target_phase_end=0.9008, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": 0.5, "z": 0.0}, # pushing on the box to the left (y positive)
        target_vel_std={"x": 0.0, "y": 0.2, "z": 0.0},
      ),
    ],
  ),

  "pickup_box_1": MotionCfg(
    name="pickup_box_1",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.412), ("right_palm", 0.867)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7876, "y": 0.1359, "z": -0.328},
        target_pos_std={"x": 0.2, "y": 0.2, "z": 0.05},
        target_phase_start=0.412,
        target_phase_end=0.430,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=2.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.2, "z": 0.02},
        target_phase_start=0.420,   # from when the box is grabbed
        target_phase_end=0.867,     # until the box is set down
      ),

    ],
  ),

  ########################### New soccer motions ###########################

  # "right_kick": MotionCfg(
  #   name="right_kick",
  #   sampling_weight=1.0,
  #   probe_points=[("right_foot", 0.293)],
  #   sub_targets=[
  #     MotionGoalCfg(
  #       goal_type="position",
  #       goal_weight=100.0,
  #       source_link="right_foot",
  #       source_type="site",
  #       target_pos_mean={"x": 0.5, "y": -0.6, "z": -0.70},
  #       target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
  #       target_phase_start=0.290,
  #       target_phase_end=0.297,
  #     ),
  #     MotionGoalCfg(
  #       goal_type="velocity",
  #       goal_weight=10.0,
  #       source_link="right_foot",
  #       source_type="site",
  #       target_phase_start=0.285,
  #       target_phase_end=0.302,
  #       target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
  #       target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
  #     ),
  #     MotionGoalCfg(
  #       goal_type="orientation",
  #       goal_weight=10.0,
  #       source_link="right_foot",
  #       source_type="site",
  #       target_phase_start=0.290,
  #       target_phase_end=0.297,
  #       target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
  #       target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
  #       orientation_axis="y",
  #     ),
  #   ],
  # ),

  "one_step_right_kick": MotionCfg(
    name="one_step_right_kick",
    sampling_weight=1.0,
    probe_points=[("right_foot", 0.293)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_foot",
        source_type="site",
        target_pos_mean={"x": 0.3206, "y": -0.3557, "z": -0.7514},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.309 - 0.004,
        target_phase_end=0.309 + 0.004,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.309 - 0.015,
        target_phase_end=0.309 + 0.015,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.309 - 0.015,
        target_phase_end=0.309 + 0.015,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "one_step_left_kick": MotionCfg(
    name="one_step_left_kick",
    sampling_weight=1.0,
    probe_points=[("left_foot", 0.403)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_foot",
        source_type="site",
        target_pos_mean={"x": 0.4484, "y": 0.4820, "z": -0.7188},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.403 - 0.004,
        target_phase_end=0.403 + 0.004,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.403 - 0.015,
        target_phase_end=0.403 + 0.015,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.403 - 0.015,
        target_phase_end=0.403 + 0.015,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "two_step_right_kick": MotionCfg(
    name="two_step_right_kick",
    sampling_weight=1.0,
    probe_points=[("right_foot", 0.407)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_foot",
        source_type="site",
        target_pos_mean={"x": 0.4856, "y": -0.7632, "z": -0.7502},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.407 - 0.004,
        target_phase_end=0.407 + 0.004,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.407 - 0.015,
        target_phase_end=0.407 + 0.015,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.407 - 0.015,
        target_phase_end=0.407 + 0.015,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "two_step_left_kick": MotionCfg(
    name="two_step_left_kick",
    sampling_weight=1.0,
    probe_points=[("left_foot", 0.474)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_foot",
        source_type="site",
        target_pos_mean={"x": 0.5222, "y":0.8246, "z": -0.6967},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.474 - 0.004,
        target_phase_end=0.474 + 0.004,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.474 - 0.015,
        target_phase_end=0.474 + 0.015,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.474 - 0.015,
        target_phase_end=0.474 + 0.015,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "two_baby_step_right_kick": MotionCfg(
    name="two_baby_step_right_kick",
    sampling_weight=1.0,
    probe_points=[("right_foot", 0.403)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_foot",
        source_type="site",
        target_pos_mean={"x": 0.4812, "y": -0.0485, "z": -0.7371},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.403 - 0.004,
        target_phase_end=0.403 + 0.004,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.403 - 0.015,
        target_phase_end=0.403 + 0.015,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.403 - 0.015,
        target_phase_end=0.403 + 0.015,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  #################################### pickup 2 motions ####################################
  "cast_pickup_1": MotionCfg(
    name="cast_pickup_1",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.378)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.6104, "y": -0.2074, "z": -0.4884},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.378 - 0.01,
        target_phase_end=0.378 + 0.01,
      ),

      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.6104, "y": 0.2, "z": -0.4884},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.378 - 0.01,
        target_phase_end=0.378 + 0.01,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.25, "z": 0.02},
        target_phase_start=0.0,   # from when the box is grabbed
        target_phase_end=1.0,     # until the box is set down
      ),
    ],
  ),

  "cast_pickup_2": MotionCfg(
    name="cast_pickup_2",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.313)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.5372, "y": -1.2038, "z": -0.3706},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.313 - 0.01,
        target_phase_end=0.313 + 0.01,
      ),

      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.5372, "y": -0.8038, "z": -0.3706},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.313 - 0.01,
        target_phase_end=0.313 + 0.01,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.25, "z": 0.02},
        target_phase_start=0.0,   # from when the box is grabbed
        target_phase_end=1.0,     # until the box is set down
      ),
    ],
  ),

  "cast_pickup_3": MotionCfg(
    name="cast_pickup_3",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.334)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7894, "y": 0.9037, "z": -0.4059},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.334 - 0.01,
        target_phase_end=0.334 + 0.01,
      ),

      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.7894, "y": 1.3037, "z": -0.4059},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.334 - 0.01,
        target_phase_end=0.334 + 0.01,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=2.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.25, "z": 0.02},
        target_phase_start=0.0,   # from when the box is grabbed
        target_phase_end=1.0,     # until the box is set down
      ),
    ],
  ),

  "cast_pickup_4": MotionCfg(
    name="cast_pickup_4",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.323)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 0.9881, "y": -0.1486, "z": -0.4271},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.323 - 0.01,
        target_phase_end=0.323 + 0.01,
      ),

      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 0.9881, "y": 0.3486, "z": -0.4271},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.323 - 0.01,
        target_phase_end=0.323 + 0.01,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.25, "z": 0.02},
        target_phase_start=0.0,   # from when the box is grabbed
        target_phase_end=1.0,     # until the box is set down
      ),
    ],
  ),

  "cast_pickup_5": MotionCfg(
    name="cast_pickup_5",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.372)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 0.7715, "y": 0.9096, "z": -0.3840},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.372 - 0.01,
        target_phase_end=0.372 + 0.01,
      ),

      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 0.7715, "y": 1.3096, "z": -0.3840},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.372 - 0.01,
        target_phase_end=0.372 + 0.01,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.25, "z": 0.02},
        target_phase_start=0.0,   # from when the box is grabbed
        target_phase_end=1.0,     # until the box is set down
      ),
    ],
  ),

  "cast_pickup_6": MotionCfg(
    name="cast_pickup_6",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.337)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 0.4943, "y": -1.0652, "z": -0.3898},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.337 - 0.01,
        target_phase_end=0.337 + 0.01,
      ),

      MotionGoalCfg(
        goal_type="position",
        goal_weight=50.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 0.4943, "y": -0.6652, "z": -0.3898},
        target_pos_std={"x": 0.3, "y": 0.3, "z": 0.1},
        target_phase_start=0.337 - 0.01,
        target_phase_end=0.337 + 0.01,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.25, "z": 0.02},
        target_phase_start=0.0,   # from when the box is grabbed
        target_phase_end=1.0,     # until the box is set down
      ),
    ],
  ),

  ##########################################################################################
  ##  MLE Estimates
  ##########################################################################################

  "mle_volley_1": MotionCfg(
    name="mle_volley_1",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.402)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.1842, "y": -0.2966, "z": 0.4117},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.402 - vel_ori_window,
        target_phase_end=0.402 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.402 - vel_ori_window,
        target_phase_end=0.402 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.402 - vel_ori_window,
        target_phase_end=0.402 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_1": MotionCfg(
    name="mle_forehand_1",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.393)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.1764, "y": 0.6351, "z": 0.4313},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.393 - vel_ori_window,
        target_phase_end=0.393 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.393 - vel_ori_window,
        target_phase_end=0.393 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.393 - vel_ori_window,
        target_phase_end=0.393 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_1": MotionCfg(
    name="mle_backhand_1",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.352)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5616 , "y": 0.5608, "z": 0.4420},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.352 - vel_ori_window,
        target_phase_end=0.352 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.352 - vel_ori_window,
        target_phase_end=0.352 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.352 - vel_ori_window,
        target_phase_end=0.352 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_2": MotionCfg(
    name="mle_backhand_2",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.381)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5628 , "y": 0.2436 , "z": 0.6045},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.381 - vel_ori_window,
        target_phase_end=0.381 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.381 - vel_ori_window,
        target_phase_end=0.381 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.381 - vel_ori_window,
        target_phase_end=0.381 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_3": MotionCfg(
    name="mle_backhand_3",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.368)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.4631 , "y": 0.3539, "z": 0.6471},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.368 - vel_ori_window,
        target_phase_end=0.368 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.368 - vel_ori_window,
        target_phase_end=0.368 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.368 - vel_ori_window,
        target_phase_end=0.368 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_4": MotionCfg(
    name="mle_backhand_4",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.474)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.4482 , "y": 0.5473 , "z": 0.6198},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.474 - vel_ori_window,
        target_phase_end=0.474 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.474 - vel_ori_window,
        target_phase_end=0.474 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.474 - vel_ori_window,
        target_phase_end=0.474 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_5": MotionCfg(
    name="mle_backhand_5",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.478)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":-0.0133 , "y": 0.5569 , "z": 0.4147},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.478 - vel_ori_window,
        target_phase_end=0.478 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.478 - vel_ori_window,
        target_phase_end=0.478 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.478 - vel_ori_window,
        target_phase_end=0.478 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_6": MotionCfg(
    name="mle_backhand_6",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.214)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.6826 , "y": 0.0199 , "z": 0.3869},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.214 - vel_ori_window,
        target_phase_end=0.214 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.214 - vel_ori_window,
        target_phase_end=0.214 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.214 - vel_ori_window,
        target_phase_end=0.214 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_7": MotionCfg(
    name="mle_backhand_7",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.194)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.6473 , "y": -0.0206 , "z": 0.5069},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.194 - vel_ori_window,
        target_phase_end=0.194 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.194 - vel_ori_window,
        target_phase_end=0.194 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.194 - vel_ori_window,
        target_phase_end=0.194 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_2": MotionCfg(
    name="mle_forehand_2",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.148)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.0863 , "y": 0.0079 , "z": 0.5123},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.148 - vel_ori_window,
        target_phase_end=0.148 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.148 - vel_ori_window,
        target_phase_end=0.148 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.148 - vel_ori_window,
        target_phase_end=0.148 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_3": MotionCfg(
    name="mle_forehand_3",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.418)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.1426 , "y": -0.4146 , "z": 0.5819},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.418 - vel_ori_window,
        target_phase_end=0.418 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.418 - vel_ori_window,
        target_phase_end=0.418 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.418 - vel_ori_window,
        target_phase_end=0.418 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_4": MotionCfg(
    name="mle_forehand_4",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.833)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":-0.0712 , "y": -0.8251 , "z": 0.6604},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.833 - vel_ori_window,
        target_phase_end=0.833 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.833 - vel_ori_window,
        target_phase_end=0.833 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.833 - vel_ori_window,
        target_phase_end=0.833 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_5": MotionCfg(
    name="mle_forehand_5",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.321)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.1824 , "y": -0.8362 , "z": 0.4264},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.321 - vel_ori_window,
        target_phase_end=0.321 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.321 - vel_ori_window,
        target_phase_end=0.321 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.321 - vel_ori_window,
        target_phase_end=0.321 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_6": MotionCfg(
    name="mle_forehand_6",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.354)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":-0.0005 , "y": 0.1260 , "z": 0.4984},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.354 - vel_ori_window,
        target_phase_end=0.354 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.354 - vel_ori_window,
        target_phase_end=0.354 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.354 - vel_ori_window,
        target_phase_end=0.354 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_7": MotionCfg(
    name="mle_forehand_7",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.354)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":-0.0005 , "y": 0.1260 , "z": 0.4984},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.354 - vel_ori_window,
        target_phase_end=0.354 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.354 - vel_ori_window,
        target_phase_end=0.354 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.354 - vel_ori_window,
        target_phase_end=0.354 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_8": MotionCfg(
    name="mle_forehand_8",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.482)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":-0.0798 , "y": 0.6537 , "z": 0.4121},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "djvkovic_fh_01_forehand": MotionCfg(
    name="djvkovic_fh_01_forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.482)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_sweet_spot",
        source_type="site",
        target_pos_mean={"x": 0.0, "y": 0.0, "z": 0.0},
        target_pos_std={"x": 0.20, "y": 0.20, "z": 0.20},
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_sweet_spot",
        source_type="site",
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_sweet_spot",
        source_type="site",
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="z",
      ),
    ],
  ),

  "mle_backhand_8": MotionCfg(
    name="mle_backhand_8",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.861)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.0526 , "y": 0.7584 , "z": 0.6577},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.861 - vel_ori_window,
        target_phase_end=0.861 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.861 - vel_ori_window,
        target_phase_end=0.861 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.861 - vel_ori_window,
        target_phase_end=0.861 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_backhand_9": MotionCfg(
    name="mle_backhand_9",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.482)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.0526 , "y": 0.7584 , "z": 0.6577},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.482 - vel_ori_window,
        target_phase_end=0.482 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_9": MotionCfg(
    name="mle_forehand_9",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.317)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":-0.4435 , "y": 0.4073 , "z": 0.3654},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.317 - vel_ori_window,
        target_phase_end=0.317 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.317 - vel_ori_window,
        target_phase_end=0.317 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.317 - vel_ori_window,
        target_phase_end=0.317 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_10": MotionCfg(
    name="mle_forehand_10",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.424)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.3265 , "y": -1.2223 , "z": 0.6763},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.424 - vel_ori_window,
        target_phase_end=0.424 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.424 - vel_ori_window,
        target_phase_end=0.424 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.424 - vel_ori_window,
        target_phase_end=0.424 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_11": MotionCfg(
    name="mle_forehand_11",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.356)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.5489 , "y": -0.6000 , "z": 0.5093},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.356 - vel_ori_window,
        target_phase_end=0.356 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.356 - vel_ori_window,
        target_phase_end=0.356 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.356 - vel_ori_window,
        target_phase_end=0.356 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_12": MotionCfg(
    name="mle_forehand_12",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.334)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.1269 , "y":-0.4261 , "z": 0.5101},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.334 - vel_ori_window,
        target_phase_end=0.334 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.334 - vel_ori_window,
        target_phase_end=0.334 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.334 - vel_ori_window,
        target_phase_end=0.334 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_forehand_13": MotionCfg(
    name="mle_forehand_13",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.294)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.3424 , "y":0.5207 , "z": 0.5854},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.294 - vel_ori_window,
        target_phase_end=0.294 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.294 - vel_ori_window,
        target_phase_end=0.294 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.294 - vel_ori_window,
        target_phase_end=0.294 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "mle_volley_2": MotionCfg(
    name="mle_volley_2",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.316)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x":0.1604 , "y":-0.4141 , "z": 0.5388},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.316 - vel_ori_window,
        target_phase_end=0.316 + vel_ori_window,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.316 - vel_ori_window,
        target_phase_end=0.316 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.316 - vel_ori_window,
        target_phase_end=0.316 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  

}
