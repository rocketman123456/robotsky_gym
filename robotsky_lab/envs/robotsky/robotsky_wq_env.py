# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

import math

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

        self.rf_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "RF_Roll_Joint",
                "RF_Hip_Joint",
                "RF_Knee_Joint",
                "RF_Wheel_Joint",
            ],
            preserve_order=True,
        )
        self.lf_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "LF_Roll_Joint",
                "LF_Hip_Joint",
                "LF_Knee_Joint",
                "LF_Wheel_Joint",
            ],
            preserve_order=True,
        )
        self.rb_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "RB_Roll_Joint",
                "RB_Hip_Joint",
                "RB_Knee_Joint",
                "RB_Wheel_Joint",
            ],
            preserve_order=True,
        )
        self.lb_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "LB_Roll_Joint",
                "LB_Hip_Joint",
                "LB_Knee_Joint",
                "LB_Wheel_Joint",
            ],
            preserve_order=True,
        )
        self.joint_names = self.robot.data.joint_names
        print(f"Joint names: {self.joint_names}")
        self.isaac2urdf_idx = self.rf_leg_ids + self.lf_leg_ids + self.rb_leg_ids + self.lb_leg_ids
        self.urdf2isaac_idx = [self.isaac2urdf_idx.index(i) for i in range(len(self.joint_names))]
        print(f"ISAAC to URDF indices: {self.isaac2urdf_idx}")
        print(f"URDF to ISAAC indices: {self.urdf2isaac_idx}")

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
            self.noise_scale_vec[joint_vel_start + wheel_ids_t] = noise_scales.joint_vel * self.obs_scales.wheel_vel

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
    # 对角 trot：在裁剪后的全关节 action 上对腿关节叠加正弦（开环辅助抬脚）
    # ------------------------------------------------------------------

    def _apply_diagonal_trot_overlay(self, clipped_actions: torch.Tensor) -> torch.Tensor:
        """在策略输出上叠加对角 trot 正弦；不改变观测中的 action（buffer 仍为原策略输出）。

        假定 command 布局为 UniformVelocityCommand: [vx, vy, wz]（机体系线速度 + 绕竖轴角速度）。
        """
        cfg = self.cfg.robot.diagonal_trot
        if not cfg.enable:
            return clipped_actions

        cmd = self.command_generator.command
        vx, vy, wz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
        v_xy = torch.sqrt(vx * vx + vy * vy + 1e-9)

        # 门控：静止/微指令时不叠加；侧移、前进、转圈均通过 v_xy 或 wz 打开
        cmd_drive = v_xy + cfg.wz_cmd_weight * torch.abs(wz)
        gate = torch.clamp((cmd_drive - cfg.cmd_deadband) / max(cfg.cmd_blend, 1e-6), 0.0, 1.0)

        # 周期：speed_blend 越大 period 越短（步频越高）；饱和映射避免数值发散
        speed_blend = v_xy / max(cfg.speed_ref, 1e-6) + torch.abs(wz) / max(cfg.ang_ref, 1e-6)
        speed_blend = torch.clamp(speed_blend, 0.0, 20.0)
        period = cfg.period_max_s - (cfg.period_max_s - cfg.period_min_s) * (speed_blend / (1.0 + speed_blend))
        # 相位：按控制步长积分等价写法；reset 后 episode_length_buf=0 相位归零
        phase = 2.0 * math.pi * self.episode_length_buf.to(dtype=torch.float32, device=self.device) * self.step_dt
        phase = phase / period
        s_a = torch.sin(phase).clamp(0.0, 1.0) * gate
        s_b = torch.sin(phase + math.pi).clamp(0.0, 1.0) * gate  # 另一对角组，相差 π

        overlay = torch.zeros_like(clipped_actions)
        amps = (cfg.roll_amp, cfg.hip_amp, cfg.knee_amp)
        # rf_leg_ids 等前 3 项为 Roll/Hip/Knee，第 4 项为轮关节，此处只写腿关节索引
        # for i in range(3):
        #     overlay[:, self.rf_leg_ids[i]] += amps[i] * s_a
        #     overlay[:, self.lb_leg_ids[i]] += amps[i] * s_a
        #     overlay[:, self.lf_leg_ids[i]] += amps[i] * s_b
        #     overlay[:, self.rb_leg_ids[i]] += amps[i] * s_b

        # roll
        overlay[:, self.rf_leg_ids[0]] += amps[0] * s_a
        overlay[:, self.lb_leg_ids[0]] += amps[0] * s_a
        overlay[:, self.lf_leg_ids[0]] += amps[0] * s_b
        overlay[:, self.rb_leg_ids[0]] += amps[0] * s_b

        # hip
        overlay[:, self.rf_leg_ids[1]] += amps[1] * s_a
        overlay[:, self.lb_leg_ids[1]] += amps[1] * s_a
        overlay[:, self.lf_leg_ids[1]] += amps[1] * s_b
        overlay[:, self.rb_leg_ids[1]] += amps[1] * s_b

        # knee
        overlay[:, self.rf_leg_ids[2]] -= 2.0 * amps[2] * s_a
        overlay[:, self.lb_leg_ids[2]] -= 2.0 * amps[2] * s_a
        overlay[:, self.lf_leg_ids[2]] -= 2.0 * amps[2] * s_b
        overlay[:, self.rb_leg_ids[2]] -= 2.0 * amps[2] * s_b

        return torch.clip(clipped_actions + overlay, -self.clip_actions, self.clip_actions)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        delayed_actions = torch.zeros_like(delayed_actions)
        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        # 对角 trot 正弦叠在腿关节 action 上；轮关节列保持为策略输出
        clipped_actions = self._apply_diagonal_trot_overlay(clipped_actions)

        # Leg joints → position target
        leg_pos_target = clipped_actions[:, self.leg_joint_ids] * self.action_scale + self.robot.data.default_joint_pos[:, self.leg_joint_ids]
        # Wheel joints → velocity target
        wheel_vel_target = clipped_actions[:, self.wheel_joint_ids] * self.wheel_action_scale + self.robot.data.default_joint_pos[:, self.wheel_joint_ids]

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(leg_pos_target, joint_ids=self.leg_joint_ids)
            self.robot.set_joint_position_target(wheel_vel_target, joint_ids=self.wheel_joint_ids)
            # self.robot.set_joint_velocity_target(wheel_vel_target, joint_ids=self.wheel_joint_ids)
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
