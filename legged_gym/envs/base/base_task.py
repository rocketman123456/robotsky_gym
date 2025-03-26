# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch
import time
import abc


# Base class for RL tasks
class BaseTask:

    def __init__(self, cfg, sim_params, physics_engine, sim_device, graphics_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        if graphics_device is not None:
            self.graphics_device_id = graphics_device  # self.sim_device_id
        else:
            self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_actor_obs = cfg.env.num_actor_obs
        self.num_critic_obs = cfg.env.num_critic_obs
        self.num_actions = cfg.env.num_actions

        self.enable_env_encoder = cfg.env.enable_env_encoder
        self.enable_adaptation_module = cfg.env.enable_adaptation_module
        self.evaluate_teacher = cfg.env.evaluate_teacher
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_obs_history = cfg.env.num_obs_history
        self.num_action_history = cfg.env.num_action_history
        self.num_latent_dim = cfg.env.num_latent_dim

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.actor_obs_buf = torch.zeros(self.num_envs, self.num_actor_obs, device=self.device, dtype=torch.float)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device, dtype=torch.float)

        if cfg.env.multi_critics:
            self.rew1_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
            self.rew2_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        else:
            self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.enable_env_encoder:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
            self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs_history, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_history_buf = torch.zeros(self.num_envs, self.num_action_history, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.privileged_obs_buf = None
            self.obs_history_buf = None
            self.action_history_buf = None

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1600
            camera_props.height = 900
            camera_props.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "free_cam")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")
        self.free_cam = True
        self.lookat_id = 0
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)
        # self.button_pressed = False

    def __del__(self):
        """Cleanup in the end."""
        try:
            if self.sim is not None:
                self.gym.destroy_sim(self.sim)
            if self.viewer is not None:
                self.gym.destroy_viewer(self.viewer)
        except:
            pass

    def destroy(self):
        pass

    """
    Properties.
    """

    def get_actor_obs(self):
        return self.actor_obs_buf

    def get_critic_obs(self):
        return self.critic_obs_buf

    def get_privileged_obs(self):
        return self.privileged_obs_buf

    def get_obs_history(self):
        return self.obs_history_buf

    def get_action_history(self):
        return self.action_history_buf

    """
    Operations.
    """

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.multi_critics:
            actor_obs, critic_obs, privileged_obs, obs_history_buf, action_history_buf, _, _, _, _ = self.step(
                torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
            )
        else:
            actor_obs, critic_obs, privileged_obs, obs_history_buf, action_history_buf, _, _, _ = self.step(
                torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
            )
        return actor_obs, critic_obs, privileged_obs, obs_history_buf, action_history_buf

    def step(self, actions):
        """Apply input action on the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, dict]:
                A tuple containing the observations, privileged observations, rewards, dones and
                extra information (metrics).
        """
        raise NotImplementedError

    def lookat(self, i):
        look_at_pos = self.root_states[i, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            if not self.free_cam:
                self.lookat(self.lookat_id)

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        for evt in self.gym.query_viewer_action_events(self.viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()
            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    """
    Protected Methods.
    """

    @abc.abstractmethod
    def create_sim(self):
        """Creates simulation, terrain and environments"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Resets the MDP for given environment instances.

        Args:
            env_ids (torch.Tensor): A tensor containing indices of environment instances to reset.
        """
        raise NotImplementedError
