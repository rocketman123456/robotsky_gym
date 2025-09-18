import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os
from random import random

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .bipedal_walker_config import BipedalWalkerCfg


class BipedalWalker(LeggedRobot):

    def check_angle_deviation_termination(self):
        """Check if the hip angles too much
        TODO: the other angle and trunk orientation
        """
        hip_saggital_indices = [1, 6]  # 髋部侧摆自由度
        hip_transversal_indices = [2, 7]  # 髋部内外旋自由度
        # 髋关节侧摆角度大于30度则终止
        hip_saggital_ang = torch.any(torch.abs(self.simulator.dof_pos[:, hip_saggital_indices]) > torch.pi / 4, dim=1)
        hip_transversal_ang = torch.any(torch.abs(self.simulator.dof_pos[:, hip_transversal_indices]) > 0.15, dim=1)
        self.reset_buf |= hip_saggital_ang
        self.reset_buf |= hip_transversal_ang

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.check_angle_deviation_termination()

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.default_dof_pos = self.simulator.default_dof_pos.squeeze(0)

        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, device=self.device)
        dof_pos[:, [0, 5]] = self.default_dof_pos[[0, 5]] + gs_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device)  # saggital
        dof_pos[:, [1, 6]] = self.default_dof_pos[[1, 6]] + gs_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device)  # frontal
        dof_pos[:, [2, 7]] = self.default_dof_pos[[2, 7]] + gs_rand_float(-0.05, 0.05, (len(env_ids), 2), device=self.device)  # transversal
        dof_pos[:, [3, 8]] = gs_rand_float(0.0, torch.pi / 2, (len(env_ids), 2), device=self.device)  # knee
        dof_pos[:, [4, 9]] = gs_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)  # ankle

        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _reward_no_fly(self):
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        return 1.0 * single_contact
