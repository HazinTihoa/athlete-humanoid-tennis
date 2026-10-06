"""Instantaneous point velocity relative to a translating, rotating frame."""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv


def point_velocity_relative_to_frame(
    point_pos_w, point_vel_w, frame_pos_w, frame_quat_w,
    frame_lin_vel_w, frame_ang_vel_w,
):
    relative_w = (
        point_vel_w - frame_lin_vel_w
        - torch.linalg.cross(frame_ang_vel_w, point_pos_w - frame_pos_w, dim=-1)
    )
    rotation = frame_quat_w.expand(relative_w.shape[:-1] + (4,))
    return quat_apply(quat_inv(rotation), relative_w)
