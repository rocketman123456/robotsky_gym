from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.srobot.srobot_biped_config import SrobotBipedFlatEnvCfg, SrobotBipedRoughEnvCfg
import torch


class SrobotBipedEnv(BaseEnv):
    def __init__(self, cfg: SrobotBipedFlatEnvCfg | SrobotBipedRoughEnvCfg, hedless):
        self.cfg: SrobotBipedFlatEnvCfg | SrobotBipedRoughEnvCfg

        self.cycle_time = 0.5
        self.dt = 0.005
        super().__init__(cfg, hedless)

    def _get_phase(self):
        phase = self.episode_length_buf * self.dt / self.cycle_time
        return phase

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                # sin_pos,
                # cos_pos,
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat([current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1)

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        current_actor_obs = torch.clip(current_actor_obs, -self.clip_obs, self.clip_obs)
        current_critic_obs = torch.clip(current_critic_obs, -self.clip_obs, self.clip_obs)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1) - self.height_scanner.data.ray_hits_w[..., 2] - self.cfg.normalization.height_scan_offset
            )
            height_scan = torch.clip(height_scan, -self.clip_obs, self.clip_obs) * self.obs_scales.height_scan
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
        return actor_obs, critic_obs
