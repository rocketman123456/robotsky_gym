# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from robotsky_lab.envs.base.base_env import BaseEnv
from robotsky_lab.envs.robotsky.robotsky_wq_config import (
    RobotSkyWQFlatEnvCfg,
    RobotSkyWQRoughEnvCfg,
)


class RobotSkyWQEnv(BaseEnv):
    """Wheel-legged robot environment for RobotSky WQ.

    Joint control strategy:
      - Leg joints (Roll / Hip / Knee): position control with ``action_scale``
      - Wheel joints: velocity control with ``wheel_action_scale``

    Observation strategy:
      - Wheel joint *positions* are zeroed out (they accumulate freely and carry no useful info)
      - Wheel joint *velocities* are included (feedback for velocity control)
    """

    def __init__(self, cfg: RobotSkyWQFlatEnvCfg | RobotSkyWQRoughEnvCfg, headless: bool):
        self.cfg: RobotSkyWQFlatEnvCfg | RobotSkyWQRoughEnvCfg
        super().__init__(cfg, headless)

    # ------------------------------------------------------------------
    # Buffer initialisation (overrides)
    # ------------------------------------------------------------------

    def init_buffers(self):
        # Resolve wheel / leg joint IDs *before* calling the parent, because
        # the parent's init_buffers → init_obs_buffer → compute_current_observations
        # call chain already relies on these attributes.
        wheel_joint_cfg = SceneEntityCfg("robot", joint_names=self.cfg.robot.wheel_joint_names)
        wheel_joint_cfg.resolve(self.scene)
        self.wheel_joint_ids: list = list(wheel_joint_cfg.joint_ids)

        num_joints: int = self.robot.data.default_joint_pos.shape[1]
        self.leg_joint_ids: list = [i for i in range(num_joints) if i not in self.wheel_joint_ids]
        self.wheel_action_scale: float = self.cfg.robot.wheel_action_scale

        super().init_buffers()

    def init_obs_buffer(self):
        # Let the base class build the noise vector (assumes all joint_pos are active)
        super().init_obs_buffer()
        if self.add_noise:
            wheel_ids_t = torch.tensor(self.wheel_joint_ids, device=self.device)
            noise_scales = self.cfg.noise.noise_scales

            # obs layout: [ang_vel(3), gravity(3), cmd(3), joint_pos(N), joint_vel(N), ...]
            # 1. Zero wheel position noise (wheel positions are zeroed in obs)
            self.noise_scale_vec[9 + wheel_ids_t] = 0.0

            # 2. Patch wheel velocity noise to use the wheel_vel obs_scale
            #    (base class applied joint_vel scale uniformly for all joint_vel entries)
            joint_vel_start = 9 + self.num_actions
            self.noise_scale_vec[joint_vel_start + wheel_ids_t] = (
                noise_scales.joint_vel * self.obs_scales.wheel_vel
            )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command

        # Leg position deviation from default; wheel positions zeroed (no positional info)
        joint_pos = (robot.data.joint_pos - robot.data.default_joint_pos).clone()
        joint_pos[:, self.wheel_joint_ids] = 0.0

        # All joint velocities included; wheel velocity gives velocity-control feedback.
        # Leg joints and wheel joints use separate obs scales so that both
        # contribute similarly to the observation despite very different speed ranges.
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        joint_vel_scaled = joint_vel.clone()
        joint_vel_scaled[:, self.leg_joint_ids] *= self.obs_scales.joint_vel
        joint_vel_scaled[:, self.wheel_joint_ids] *= self.obs_scales.wheel_vel

        action = self.action_buffer._circular_buffer.buffer[:, -1, :]

        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel_scaled,
                action * self.obs_scales.actions,
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact],
            dim=-1,
        )

        return current_actor_obs, current_critic_obs

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        # Leg joints → position target
        leg_pos_target = clipped_actions[:, self.leg_joint_ids] * self.action_scale + self.robot.data.default_joint_pos[:, self.leg_joint_ids]
        # Wheel joints → velocity target
        wheel_vel_target = clipped_actions[:, self.wheel_joint_ids] * self.wheel_action_scale

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(leg_pos_target, joint_ids=self.leg_joint_ids)
            self.robot.set_joint_velocity_target(wheel_vel_target, joint_ids=self.wheel_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras
