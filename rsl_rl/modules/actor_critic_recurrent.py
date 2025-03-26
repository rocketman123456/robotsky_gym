# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import unpad_trajectories
from rsl_rl.utils.torch_utils import get_activation


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        enable_env_encoder,
        num_privileged_obs,
        num_obs_history,
        num_action_history,
        num_actions,
        num_latent_dim,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        env_encoder_hidden_dims=[256, 128],
        adaptation_hidden_dims=[256, 32],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()))

        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.enable_env_encoder = enable_env_encoder

        if self.enable_env_encoder:
            mlp_input_dim_a = rnn_hidden_size  # num_actor_obs + num_latent_dim
            mlp_input_dim_c = rnn_hidden_size  # num_critic_obs + num_latent_dim
        else:
            mlp_input_dim_a = rnn_hidden_size  # num_actor_obs
            mlp_input_dim_c = rnn_hidden_size  # num_critic_obs

        if self.enable_env_encoder:
            mlp_input_dim_encoder = num_privileged_obs
            mlp_input_dim_adap = num_obs_history  # + num_action_history
        else:
            mlp_input_dim_encoder = 0
            mlp_input_dim_adap = 0

        # Policy
        self.create_actor(mlp_input_dim_a, actor_hidden_dims, num_actions, activation)

        # Value function
        self.create_critic(mlp_input_dim_c, critic_hidden_dims, activation)

        # create env encoder
        if self.enable_env_encoder:
            self.create_env_encoder(mlp_input_dim_encoder, env_encoder_hidden_dims, num_latent_dim, activation)

        # create adaptation module
        if self.enable_env_encoder:
            self.create_adaptation_module(mlp_input_dim_adap, adaptation_hidden_dims, num_latent_dim, activation)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False  # type: ignore

        activation = get_activation(activation)

        if enable_env_encoder:
            self.memory_a = Memory(num_actor_obs + num_latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_c = Memory(num_critic_obs + num_latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            # self.memory_a = Memory(num_actor_obs + num_latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=num_actor_obs + num_latent_dim)
            # self.memory_c = Memory(num_critic_obs + num_latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=num_critic_obs + num_latent_dim)
        else:
            self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            # self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=num_actor_obs)
            # self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=num_critic_obs)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, actor_obs, masks=None, hidden_states=None):
        input_a = self.memory_a(actor_obs, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_with_env(self, actor_obs, privileged_obs, masks=None, hidden_states=None):
        latent = self.env_encoder(privileged_obs)
        obs = torch.cat((actor_obs, latent), dim=-1)
        input_a = self.memory_a(obs, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_with_adaptation(self, actor_obs, obs_history, masks=None, hidden_states=None):
        latent = self.adaptation_module((obs_history))
        obs = torch.cat((actor_obs, latent), dim=-1)
        input_a = self.memory_a(obs, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def evaluate_with_env(self, critic_obs, privileged_obs, masks=None, hidden_states=None):
        latent = self.env_encoder(privileged_obs)
        obs = torch.cat((critic_obs, latent), dim=-1)
        input_c = self.memory_c(obs, masks, hidden_states)
        value = self.critic(input_c.squeeze(0))
        return value

    def evaluate_with_adaptation(self, critic_obs, obs_history, masks=None, hidden_states=None):
        latent = self.adaptation_module((obs_history))
        obs = torch.cat((critic_obs, latent), dim=-1)
        input_c = self.memory_c(obs, masks, hidden_states)
        value = self.critic(input_c.squeeze(0))
        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256, device="cuda:0"):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
        self.is_recurrent = True

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
