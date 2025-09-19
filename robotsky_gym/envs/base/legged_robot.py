# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
from warnings import WarningMessage
import numpy as np
import os

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.solvers.avatar_solver import AvatarSolver


import torch
from torch import Tensor
from typing import Tuple, Dict

from robotsky_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR, envs
from robotsky_gym.utils.torch_utils import *
from robotsky_gym.envs.base.base_task import BaseTask
from robotsky_gym.utils.terrain import Terrain
from robotsky_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from robotsky_gym.utils.helpers import class_to_dict
from robotsky_gym.utils.gs_utils import (
    gs_rand_float,
    gs_quat_mul,
    gs_inv_quat,
    gs_transform_by_quat,
    gs_euler2quat,
    gs_quat2euler,
    gs_quat_conjugate,
)
from .legged_robot_config import LeggedRobotCfg

# gs.init(logging_level="warning")


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'mps' 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.headless = headless
        self.physics_engine = physics_engine
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # if not self.headless:
        #     self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        # step physics and render each frame
        self.actions[:] = actions[:]
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(self.num_envs), self.action_delay].clone()
        # self.simulator.step(actions)

        if self.cfg.sim.use_implicit_controller:  # use embedded pd controller
            target_dof_pos = self._compute_target_dof_pos(exec_actions)
            self.torques = self._compute_torques(exec_actions)
            clip_target_dof_pos = torch.clip(target_dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
            self.robot.control_dofs_position(clip_target_dof_pos, self.motor_dofs)
            self.scene.step()
        else:
            for _ in range(self.cfg.control.decimation):  # use self-implemented pd controller
                self.torques = self._compute_torques(exec_actions)
                if self.num_build_envs == 0:
                    torques = self.torques.squeeze()
                    self.robot.control_dofs_force(torques, self.motor_dofs)
                else:
                    self.robot.control_dofs_force(self.torques, self.motor_dofs)
                self.scene.step()
                self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
                self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.post_physics_step()

        # for _ in range(self.cfg.control.decimation):
        #     self._apply_action(actions)
        #     self.scene.step()
        # self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        obs_naninf_mask = torch.isnan(self.obs_buf) | torch.isinf(self.obs_buf)
        env_ids_obs = obs_naninf_mask.any(dim=1).nonzero(as_tuple=False).flatten()
        rew_naninf_mask = torch.isnan(self.rew_buf) | torch.isinf(self.rew_buf)
        env_ids_rew = rew_naninf_mask.nonzero(as_tuple=False).flatten()
        env_ids = torch.unique(torch.cat([env_ids_obs, env_ids_rew]))

        if torch.isnan(self.obs_buf).any() or torch.isinf(self.obs_buf).any():
            self.obs_buf[:] = self.last_obs_buf.clone()
            # self.reset_idx(env_ids)
        if torch.isnan(self.rew_buf).any() or torch.isinf(self.rew_buf).any():
            self.rew_buf[:] = self.last_rew_buf.clone()
        else:
            self.last_rew_buf[:] = self.rew_buf.clone()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities

        self.update_buffer()
        # self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        # self.simulator.last_dof_vel[:] = self.simulator.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def update_buffer(self):
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
        inv_base_quat = inv_quat(self.base_quat)
        self.last_base_ang_vel[:] = self.base_ang_vel[:]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=torch.float,
        )
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.root_states = torch.cat((self.base_pos[:], self.base_quat[:], self.base_lin_vel[:], self.base_ang_vel[:]), dim=-1)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.links_pos[:] = self.robot.get_links_pos()

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > 10
        # self.reset_buf |= torch.abs(self.base_euler[:, 0]) > 10
        self.time_out_buf = self.episode_length_buf > self.max_episode_length - 10  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # proj_grav_over_limit = self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity
        # self.reset_buf |= proj_grav_over_limit

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self._resample_commands(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # self.simulator.reset_idx(env_ids)

        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)

        # reset buffers
        self.llast_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_base_ang_vel[env_ids] = 0.0
        self.last_base_lin_vel[env_ids] = 0.0

        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            # self.extras["episode"]["terrain_level"] = torch.mean(self.simulator.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.0
            self.action_queue[env_ids] = 0.0
            self.action_delay[env_ids] = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (len(env_ids),),
                device=self.device,
                requires_grad=False,
            )

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]

            if torch.isnan(rew).any():
                rew = torch.zeros_like(rew)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """Computes observations"""
        # TODO : use simulator instead
        self.last_obs_buf[:] = self.obs_buf.clone()
        self.last_privileged_obs_buf = self.privileged_obs_buf.clone() if self.privileged_obs_buf is not None else None

        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay / self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.projected_gravity,
                    self.commands[:, :3] * self.commands_scale,
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    self.actions,
                    self.last_actions,
                    # self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                    # self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                    # self.simulator.projected_gravity,
                    # self.commands[:, :3] * self.commands_scale,
                    # (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
                    # self.simulator.dof_vel * self.obs_scales.dof_vel,
                    # self.actions,
                    # self.last_actions,
                    # self.simulator._friction_values,  # 1
                    # self.simulator._added_base_mass,  # 1
                    # self.simulator._base_com_bias,  # 3
                    # self.simulator._rand_push_vels[:, :2],  # 2
                ),
                dim=-1,
            )
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = (
                    torch.clip(self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 - self.simulator.measured_heights, -1, 1.0)
                    * self.obs_scales.height_measurements
                )
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        if torch.isnan(self.obs_buf).any() or torch.isinf(self.obs_buf).any():
            self.obs_buf = self.last_obs_buf.clone()

        if self.privileged_obs_buf is not None:
            if torch.isnan(self.privileged_obs_buf).any() or torch.isinf(self.privileged_obs_buf).any():
                self.privileged_obs_buf = self.last_privileged_obs_buf.clone()

    def set_camera(self, pos, lookat):
        """Set camera position and direction"""
        self.floating_camera.set_pose(pos=pos, lookat=lookat)
        # cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        # cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        # self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        gs.init(logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_FPS=False,
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1.0 / self.dt * self.cfg.control.decimation),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=min(self.cfg.viewer.num_rendered_envs, self.num_envs)),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
            ),
            show_viewer=self.show_viewer,
        )
        # create ground
        if mesh_type in ["heightfield", "trimesh"]:
            raise NotImplementedError("Heightfield and trimesh terrains are not yet supported")
            # self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            envs_asset_path = LEGGED_GYM_ROOT_DIR + "/resources/envs/plane/plane.urdf"
            self.scene.add_entity(gs.morphs.URDF(file=envs_asset_path, fixed=True))
        elif mesh_type == "heightfield":
            pass
            # self._create_heightfield()
        elif mesh_type == "trimesh":
            pass
            # self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            elif isinstance(solver, AvatarSolver):
                continue
            self.rigid_solver = solver

        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()

        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == "plane":
            self.terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type == "heightfield":
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type == "heightfield":
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0  # give a small margin(1.0m)
            self.terrain_x_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type == "plane":  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length / 2 + 1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length / 2 - 1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length / 2 + 1  # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length / 2 - 1
        self._create_envs()

    # ------------- Callbacks --------------

    def _setup_camera(self):
        """Set camera position and direction"""
        self.floating_camera = self.scene.add_camera(
            res=(1280, 960),
            pos=np.array(self.cfg.viewer.pos),
            lookat=np.array(self.cfg.viewer.lookat),
            fov=40,
            GUI=True,
        )

        self._recording = False
        self._recorded_frames = []

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.simulator.base_pos[env_ids, :2] - self.simulator.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.simulator.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.simulator.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.simulator.terrain_levels[env_ids] = torch.where(
            self.simulator.terrain_levels[env_ids] >= self.simulator.max_terrain_level,
            torch.randint_like(self.simulator.terrain_levels[env_ids], self.simulator.max_terrain_level),
            torch.clip(self.simulator.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.simulator.env_origins[env_ids] = self.simulator.terrain_origins[self.simulator.terrain_levels[env_ids], self.simulator.terrain_types[env_ids]]

    def _process_rigid_shape_props(self, props, env_id):
        """Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device="cpu")
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)

        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        self.dof_vel_limits = torch.zeros_like(self.dof_pos_limits)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.asset.max_joint_velocity:
                if dof_name in name:
                    self.dof_vel_limits[i] = self.cfg.asset.max_joint_velocity[dof_name]
                    found = True
            if not found:
                self.dof_vel_limits[i] = 200.0

        for i in range(self.dof_vel_limits.shape[0]):
            # soft limits
            m = (self.dof_vel_limits[i, 0] + self.dof_vel_limits[i, 1]) / 2
            r = self.dof_vel_limits[i, 1] - self.dof_vel_limits[i, 0]
            self.dof_vel_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_vel_limit
            self.dof_vel_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_vel_limit

        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            # self.simulator.get_heights()
            # self.measured_heights = self._get_heights()
            # raise NotImplementedError("Measuring heights not yet implemented")
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
            # self.simulator.push_robots()
            # raise NotImplementedError("Pushing robots not yet implemented")

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        torques = self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.batched_d_gains * self.dof_vel
        return torques.to(self.device)

    def _compute_target_dof_pos(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        target_dof_pos = actions_scaled + self.default_dof_pos

        return target_dof_pos.to(self.device)

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(-0.3, 0.3, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.0

        # dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        # dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        # dof_pos[:, :] = self.simulator.default_dof_pos[:] + torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions), self.device)
        # self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

    def _apply_action(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            target_dof_pos = actions_scaled + self.default_dof_pos
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        elif control_type == "V":
            raise NotImplementedError("Velocity control not yet implemented")
        elif control_type == "T":
            raise NotImplementedError("Torque control not yet implemented")
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """

        if self.custom_origins:
            self.base_pos[env_ids] = self.base_init_pos
            self.base_pos[env_ids] += self.env_origins[env_ids]
            self.base_pos[env_ids, :2] += gs_rand_float(-1.0, 1.0, (len(env_ids), 2), self.device)
        else:
            self.base_pos[env_ids] = self.base_init_pos
            self.base_pos[env_ids] += self.env_origins[env_ids]
        self.robot.set_pos(self.base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)

        # base quat
        self.base_quat[env_ids] = self.base_init_quat.reshape(1, -1)
        base_euler = gs_rand_float(-0.1, 0.1, (len(env_ids), 3), self.device)  # roll, pitch [-0.1, 0.1]
        base_euler[:, 2] = gs_rand_float(*self.cfg.init_state.yaw_angle_range, (len(env_ids),), self.device)  # yaw angle
        self.base_quat[env_ids] = gs_quat_mul(gs_euler2quat(base_euler), self.base_quat[env_ids])
        self.robot.set_quat(self.base_quat[env_ids], zero_velocity=False, envs_idx=env_ids)
        self.robot.zero_all_dofs_velocity(env_ids)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)

        # reset root states - velocity
        self.base_lin_vel[env_ids] = gs_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        self.base_ang_vel[env_ids] = gs_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        base_vel = torch.concat([self.base_lin_vel[env_ids], self.base_ang_vel[env_ids]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=env_ids)

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        # max_vel = self.cfg.domain_rand.max_push_vel_xy
        # self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        # in Genesis, base link also has DOF, it's 6DOF if not fixed.
        dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
        push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
        dofs_vel[:, :2] += push_vel
        self.robot.set_dofs_velocity(dofs_vel)

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.utils_terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.0)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.0, self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.0  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""

        # self.gym.refresh_net_contact_force_tensor(self.sim)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.contact_forces = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.link_contact_forces = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)

        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=gs.tc_float)
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.int)

        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=torch.int)

        self.base_init_pos = torch.tensor(self.cfg.init_state.pos, device=self.device)
        self.base_init_quat = torch.tensor(self.cfg.init_state.rot, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.last_obs_buf = torch.zeros_like(self.obs_buf)
        self.last_privileged_obs_buf = None
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.last_rew_buf = torch.zeros_like(self.rew_buf)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # x vel, y vel, yaw vel, heading
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this

        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)

        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
        inv_base_quat = inv_quat(self.base_quat)

        self.root_states = torch.cat((self.base_pos[:], self.base_quat[:], self.base_lin_vel[:], self.base_ang_vel[:]), dim=-1)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        self.root_vel = self.root_states[:, 7:13]
        self.last_root_vel = self.root_states[:, 7:13]

        self.last_feet_z = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.links_pos = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=torch.float)
        self.feet_heights = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=torch.float)

        self.continuous_push = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int,
        )
        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # PD control
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping

        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        self._process_dof_props()

        if self.cfg.terrain.measure_heights:
            # self.simulator._init_height_points()
            raise NotImplementedError("Measuring heights not yet implemented")
        self.measured_heights = 0

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1] + 1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.action_delay = torch.randint(
                self.cfg.domain_rand.ctrl_delay_step_range[0],
                self.cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (self.num_envs,),
                device=self.device,
                requires_grad=False,
            )

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                height_field=self.utils_terrain.height_field_raw,
            )
        )

        self.height_samples = torch.tensor(self.utils_terrain.heightsamples).view(self.utils_terrain.tot_rows, self.utils_terrain.tot_cols).to(self.device)

        # hf_params = gymapi.HeightFieldParams()
        # hf_params.column_scale = self.terrain.cfg.horizontal_scale
        # hf_params.row_scale = self.terrain.cfg.horizontal_scale
        # hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        # hf_params.nbRows = self.terrain.tot_cols
        # hf_params.nbColumns = self.terrain.tot_rows
        # hf_params.transform.p.x = -self.terrain.cfg.border_size
        # hf_params.transform.p.y = -self.terrain.cfg.border_size
        # hf_params.transform.p.z = 0.0
        # hf_params.static_friction = self.cfg.terrain.static_friction
        # hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        # hf_params.restitution = self.cfg.terrain.restitution

        # self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        # self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # def _create_trimesh(self):
    #     """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
    #     #"""
    #     tm_params = gymapi.TriangleMeshParams()
    #     tm_params.nb_vertices = self.terrain.vertices.shape[0]
    #     tm_params.nb_triangles = self.terrain.triangles.shape[0]
    #
    #     tm_params.transform.p.x = -self.terrain.cfg.border_size
    #     tm_params.transform.p.y = -self.terrain.cfg.border_size
    #     tm_params.transform.p.z = 0.0
    #     tm_params.static_friction = self.cfg.terrain.static_friction
    #     tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
    #     tm_params.restitution = self.cfg.terrain.restitution
    #     self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order="C"), self.terrain.triangles.flatten(order="C"), tm_params)
    #     self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
            2.1 creates the environment,
            2.2 calls DOF and Rigid shape properties callbacks,
            2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

        self.base_init_pos = to_torch(self.cfg.init_state.pos, device=self.device, requires_grad=False)
        self.base_init_quat = to_torch(self.cfg.init_state.rot, device=self.device, requires_grad=False)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=asset_path,
                links_to_keep=self.cfg.asset.links_to_keep,
                merge_fixed_links=True,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        self.scene.build(n_envs=self.num_envs)

        self._get_env_origins()

        # self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]
        if self.cfg.asset.dof_names is not None:
            self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.cfg.asset.dof_names]
        else:
            raise ValueError("DOF names not defined in the config file")

        # print("####Dof names auto 2: ", self.dof_names_auto)
        self.num_bodies = self.robot.n_links
        self.dof_names = []
        self.joints_obj = self.robot.joints
        for joint in self.joints_obj:
            if joint.name in self.cfg.asset.dof_names:
                self.dof_names.append(joint.name)
        if len(self.dof_names) != len(self.motor_dofs):
            raise ValueError("Number of DOF names and motor DOFs do not match!")
        self.num_dofs = len(self.dof_names)  # minux base joint
        self.num_dof = self.num_dofs

        body_names = []
        self.links_obj = self.robot.links
        self.body_names = [link.name for link in self.links_obj]
        print("####Body names: ", self.body_names)
        print("####Dof names : ", self.dof_names)
        # PD control parameters

        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.cfg.asset.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.feet_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        # feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        # self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        # self.feet_indices_world_frame = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        # index = 0
        # for name in feet_names:
        #     self.feet_indices[index] = self.robot.get_link(name).idx_local
        #     self.feet_indices_world_frame[index] = self.robot.get_link(name).idx
        #     index = index + 1
        assert len(self.feet_indices) > 0

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.penalised_contact_indices_world_frame = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        index = 0
        for name in penalized_contact_names:
            self.penalised_contact_indices[index] = self.robot.get_link(name).idx_local
            self.penalised_contact_indices_world_frame[index] = self.robot.get_link(name).idx
            index = index + 1

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.termination_contact_indices_world_frame = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        index = 0
        for name in termination_contact_names:
            self.termination_contact_indices[index] = self.robot.get_link(name).idx_local
            self.termination_contact_indices_world_frame[index] = self.robot.get_link(name).idx
            index = index + 1

    def find_link_indices(self, names):
        link_indices = list()
        for link in self.robot.links:
            flag = False
            for name in names:
                if name in link.name:
                    flag = True
            if flag:
                link_indices.append(link.idx - self.robot.link_start)

        return link_indices

    def _randomize_friction(self, env_ids=None):
        """Randomize friction of all links"""
        min_friction, max_friction = self.cfg.domain_rand.friction_range

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) * (max_friction - min_friction) + min_friction
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids=None):
        """Randomize base mass"""
        min_mass, max_mass = self.cfg.domain_rand.added_mass_range
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass
        self.rigid_solver.set_links_mass_shift(
            added_mass,
            [
                base_link_id,
            ],
            env_ids,
        )

    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.cfg.domain_rand.com_displacement_range
        base_link_id = 1

        com_displacement = gs.rand((len(env_ids), 1, 3), dtype=float) * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(
            com_displacement,
            [
                base_link_id,
            ],
            env_ids,
        )

    def _parse_cfg(self, cfg):
        print("self.sim_params.sim: ", self.cfg.sim.dt)  # elf.sim_params["sim"])
        self.dt = self.cfg.control.decimation * self.cfg.sim.dt  # self.sim_params["sim"]["dt"]
        if self.cfg.sim.use_implicit_controller:  # use embedded PD controller
            self.sim_dt = self.dt
            self.sim_substeps = self.cfg.control.decimation
        else:  # use explicit PD controller
            self.sim_dt = self.dt / self.cfg.control.decimation
            self.sim_substeps = 1
        # self.debug = self.cfg.env.debug
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.dof_names = self.cfg.asset.dof_names
        self.simulate_action_latency = self.cfg.domain_rand.simulate_action_latency
        self.debug = self.cfg.env.debug

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        self.scene.clear_debug_objects()
        height_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points)
        height_points[0, :, 0] += self.base_pos[0, 0]
        height_points[0, :, 1] += self.base_pos[0, 1]
        height_points[0, :, 2] = self.measured_heights[0, :]
        # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        self.scene.draw_debug_spheres(height_points[0, :], radius=0.03, color=(0, 0, 1, 0.7))  # only draw for the first env

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device), (self.num_envs / self.cfg.terrain.num_cols), rounding_mode="floor"
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.utils_terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = min((self.cfg.terrain.plane_length / 2) / (num_rows - 1), (self.cfg.terrain.plane_length / 2) / (num_cols - 1))
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
            self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (
                self.base_pos[env_ids, :3]
            ).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.base_pos[:, :3]).unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points / self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        """Penalize action smoothness"""
        action_smoothness_cost = torch.sum(torch.square(self.actions - 2 * self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0.0, max=1.0), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # raise NotImplementedError("feet_air_time reward not yet implemented")
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        return 1.0 * single_contact

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.0), dim=1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        raise NotImplementedError("Stumble reward not yet implemented")
        # return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
        #     5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_joint_pos(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_distance(self):
        reward = 0
        for i in range(self.feet_state.shape[1] - 1):
            for j in range(i + 1, self.feet_state.shape[1]):
                feet_distance = torch.norm(self.feet_state[:, i, :2] - self.feet_state[:, j, :2], dim=-1)
            reward += torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(self.simulator.feet_pos[:, :, 2] - self.cfg.rewards.foot_clearance_target - self.cfg.rewards.foot_height_offset),
            dim=-1,
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)

    def _reward_foot_landing_vel(self):
        z_vels = self.simulator.feet_vel[:, :, 2]
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 0.1
        about_to_land = (
            ((self.simulator.feet_pos[:, :, 2] - self.cfg.rewards.foot_height_offset) < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        )
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_survival(self):
        return (~self.reset_buf).float() * self.dt
