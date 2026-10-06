"""Batched joint FK and deployment-style finite-difference site observations."""

from __future__ import annotations

import mujoco
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul


class RacketJointFK:
    """Compile only the root-to-site hinge chain; evaluate all envs on device."""

    def __init__(self, model, joint_names, *, root_name, site_name, device, dtype):
        root_id = model.body(root_name).id
        site = model.site(site_name)
        chain = []
        body = int(site.bodyid[0])
        while body != root_id:
            if body == 0:
                raise ValueError(f"{site_name} is not a descendant of {root_name}")
            chain.append(body)
            body = int(model.body_parentid[body])
        indices = {name: i for i, name in enumerate(joint_names)}
        tensor = lambda value: torch.as_tensor(value.copy(), device=device, dtype=dtype)
        self.links = []
        for body in reversed(chain):
            joints = []
            start = int(model.body_jntadr[body])
            for j in range(start, start + int(model.body_jntnum[body])):
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("Racket FK requires hinge joints below the root")
                name = model.joint(j).name.split("/")[-1]
                joints.append(
                    (
                        indices[name],
                        tensor(model.jnt_pos[j]),
                        tensor(model.jnt_axis[j]),
                        float(model.qpos0[model.jnt_qposadr[j]]),
                    )
                )
            self.links.append(
                (tensor(model.body_pos[body]), tensor(model.body_quat[body]), joints)
            )
        self.site_pos = tensor(model.site_pos[site.id])

    def position_b(self, joint_pos):
        n = joint_pos.shape[0]
        pos = joint_pos.new_zeros(n, 3)
        quat = joint_pos.new_zeros(n, 4)
        quat[:, 0] = 1
        for offset, orientation, joints in self.links:
            pos = pos + quat_apply(quat, offset.expand(n, -1))
            quat = quat_mul(quat, orientation.expand(n, -1))
            for index, pivot, axis, reference in joints:
                pivot = pivot.expand(n, -1)
                pivot_w = pos + quat_apply(quat, pivot)
                angle = (joint_pos[:, index] - reference)[:, None] * 0.5
                rotation = torch.cat((angle.cos(), angle.sin() * axis), dim=-1)
                quat = quat_mul(quat, rotation)
                pos = pivot_w - quat_apply(quat, pivot)
        return pos + quat_apply(quat, self.site_pos.expand(n, -1))


class FKDifferenceState:
    """Once-per-control-step estimate; partial resets never produce velocity spikes."""

    def __init__(self, count, device, dtype, smoothing=0.35):
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        self.smoothing = smoothing
        self.previous = torch.zeros(count, 3, device=device, dtype=dtype)
        self.velocity = torch.zeros_like(self.previous)
        self.last_step = torch.full((count,), -1, device=device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.last_step[ids] = -1
        self.velocity[ids] = 0

    def update(self, position_w, step, step_dt):
        dt = (step - self.last_step).to(position_w.dtype) * step_dt
        changed = self.last_step != step
        valid = (self.last_step >= 0) & (dt >= 1e-4) & (dt <= 0.2)
        raw = (position_w - self.previous) / dt.clamp_min(1e-4)[:, None]
        filtered = self.smoothing * raw + (1 - self.smoothing) * self.velocity
        next_velocity = torch.where(valid[:, None], filtered, self.velocity)
        self.velocity = torch.where(changed[:, None], next_velocity, self.velocity)
        self.previous = torch.where(changed[:, None], position_w, self.previous)
        self.last_step.fill_(step)
        return self.velocity


class TennisSweetSpotFK(ManagerTermBase):
    """Position from joint FK; world position difference + EMA, then root rotation."""

    def __init__(self, cfg, env):
        super().__init__(env)
        command_name = cfg.params["command_name"]
        site_name = cfg.params["site_name"]
        smoothing = cfg.params["smoothing"]
        command = env.command_manager.get_term(command_name)
        robot = command.robot
        key = (command_name, site_name, smoothing)
        states = getattr(env, "_racket_fk_observations", None)
        if states is None:
            states = {}
            env._racket_fk_observations = states
        if key not in states:
            model = env.sim.mj_model
            # Resolve names from the compiled scene, not XML joint order.
            root_name = f"robot/{command.cfg.anchor_body_name}"
            compiled_site = f"robot/{site_name}"
            fk = RacketJointFK(
                model,
                robot.joint_names,
                root_name=root_name,
                site_name=compiled_site,
                device=env.device,
                dtype=robot.data.joint_pos.dtype,
            )
            difference = FKDifferenceState(
                env.num_envs, env.device, robot.data.joint_pos.dtype, smoothing
            )
            states[key] = {"fk": fk, "difference": difference, "step": None}
        self.state = states[key]

    def reset(self, env_ids=None):
        self.state["difference"].reset(env_ids)
        self.state["step"] = None

    @torch.no_grad()
    def __call__(self, env, command_name, quantity, site_name, smoothing):
        del site_name, smoothing
        command = env.command_manager.get_term(command_name)
        step = int(env.common_step_counter)
        if self.state["step"] != step:
            pos_b = self.state["fk"].position_b(command.robot.data.joint_pos)
            root_quat = command.robot_anchor_quat_w
            pos_w = command.robot_anchor_pos_w + quat_apply(root_quat, pos_b)
            vel_w = self.state["difference"].update(pos_w, step, env.step_dt)
            self.state["position"] = pos_b
            self.state["linear_velocity"] = quat_apply(quat_inv(root_quat), vel_w)
            self.state["step"] = step
        if quantity == "position":
            return self.state["position"]
        if quantity == "linear_velocity":
            return self.state["linear_velocity"]
        raise ValueError(f"Unknown FK quantity: {quantity}")
