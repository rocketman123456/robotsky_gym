from typing import Union
import torch


class JointModel:
    def __init__(
        self,
        num_envs,
        num_joints,
        tau_max: Union[float, torch.Tensor],
        q_dot_max: Union[float, torch.Tensor],
        q_dot_max_at_tau_max: Union[float, torch.Tensor] = 1.0,
        max_q_offset: Union[float, torch.Tensor] = 0.0,
        min_backlash: Union[float, torch.Tensor] = 0.0,
        max_backlash: Union[float, torch.Tensor] = 0.0,
        backlash_activation: Union[float, torch.Tensor] = 0.1,
        q_sigma_0: Union[float, torch.Tensor] = 0.0,
        q_sigma_1: Union[float, torch.Tensor] = 0.0,
        min_friction_static: Union[float, torch.Tensor] = 0.0,
        max_friction_static: Union[float, torch.Tensor] = 0.0,
        min_friction_dynamic: Union[float, torch.Tensor] = 0.0,
        max_friction_dynamic: Union[float, torch.Tensor] = 0.0,
        friction_activation: Union[float, torch.Tensor] = 0.1,
        device="cpu",
    ):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.max_q_offset = max_q_offset
        self.min_backlash = min_backlash
        self.max_backlash = max_backlash
        self.device = device

        self.q_offset = torch.zeros(num_envs, num_joints, device=device)

        self.tau_max = torch.zeros(num_envs, num_joints, device=device) + tau_max

        self.q_dot_max_at_tau_max = torch.zeros(num_envs, num_joints, device=device) + q_dot_max_at_tau_max
        self.q_dot_max = torch.zeros(num_envs, num_joints, device=device) + q_dot_max

        self.min_friction_static = min_friction_static
        self.max_friction_static = max_friction_static
        self.min_friction_dynamic = min_friction_dynamic
        self.max_friction_dynamic = max_friction_dynamic
        self.friction_static = torch.zeros(num_envs, num_joints, device=device)
        self.friction_dynamic = torch.zeros(num_envs, num_joints, device=device)

        self.friction_activation = torch.zeros(num_envs, num_joints, device=device) + friction_activation

        self.backlash = torch.zeros(num_envs, num_joints, device=device)
        self.backlash_activation = torch.zeros(num_envs, num_joints, device=device) + backlash_activation
        self.q_sigma_0 = torch.zeros(num_envs, num_joints, device=device) + q_sigma_0
        self.q_sigma_1 = torch.zeros(num_envs, num_joints, device=device) + q_sigma_1

        self.backlash_travel = torch.zeros(num_envs, num_joints, device=device)

    def sample_params(self, env_ids=slice(None)):
        self.q_offset[env_ids, :] = (torch.rand_like(self.q_offset[env_ids, :]) * 2 - 1) * self.max_q_offset
        self.friction_static[env_ids, :] = (torch.rand_like(self.friction_static[env_ids, :]) * 2 - 1) * (
            self.max_friction_static - self.min_friction_static
        ) + self.min_friction_static
        self.friction_dynamic[env_ids, :] = (torch.rand_like(self.friction_dynamic[env_ids, :]) * 2 - 1) * (
            self.max_friction_dynamic - self.min_friction_dynamic
        ) + self.min_friction_dynamic
        self.backlash[env_ids, :] = (torch.rand_like(self.backlash[env_ids, :]) * 2 - 1) * (self.max_backlash - self.min_backlash) + self.min_backlash

    def calc_friction(self, qdot):
        tau_f = self.friction_static * torch.tanh(qdot / self.friction_activation)
        tau_f += self.friction_dynamic * qdot
        return tau_f

    def clamp_tau_by_vel(self, tau: torch.Tensor, qdot: torch.Tensor):
        tau_max = (qdot - self.q_dot_max) * self.tau_max / (self.q_dot_max_at_tau_max - self.q_dot_max)
        tau_max = tau_max.clamp(max=self.tau_max)
        tau_min = (qdot + self.q_dot_max) * (-self.tau_max) / (-self.q_dot_max_at_tau_max + self.q_dot_max)
        tau_min = tau_min.clamp(min=-self.tau_max)
        return tau.clamp(min=tau_min, max=tau_max)

    def calc_backlash(self, cmd: torch.Tensor):
        backlash_periods = 5
        self.backlash_travel += cmd.sign()
        self.backlash_travel.clamp_(min=-backlash_periods, max=backlash_periods)
        return self.backlash_travel.abs() >= backlash_periods

    def sample_q_sigma(self, qdot: torch.Tensor):
        q_sigma = self.q_sigma_0 + self.q_sigma_1 * torch.abs(qdot)
        return torch.randn_like(qdot) * q_sigma

    def estimate(self, q_des, q, qdot, kp, kd, tau_0):
        q_e = q + self.q_offset + self.sample_q_sigma(qdot)
        # # use last tau to calculate backlash
        # q_h = q_e + self.calc_backlash(tau_0)

        tau_m = kp * (q_des - q_e) - kd * qdot
        tau_b = self.calc_backlash(q_des - q_e) * tau_m
        tau_f = self.calc_friction(qdot)
        tau = self.clamp_tau_by_vel(tau_b, qdot) - tau_f
        return tau, q_e
