# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from rsl_rl.utils.torch_utils import get_activation
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent, Memory


class ActorCriticRecurrentMultiCritic(ActorCriticRecurrent):
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
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        # super(ActorCritic, self).__init__()

        super().__init__(
            num_actor_obs,
            num_critic_obs,
            enable_env_encoder,
            num_privileged_obs,
            num_obs_history,
            num_action_history,
            num_actions,
            num_latent_dim,
            actor_hidden_dims,
            critic_hidden_dims,
            env_encoder_hidden_dims,
            adaptation_hidden_dims,
            activation,
            rnn_type,
            rnn_hidden_size,
            rnn_num_layers,
            init_noise_std,
            **kwargs,
        )

        self.critic = None
        self.memory_c = None

        activation = get_activation(activation)

        self.enable_env_encoder = enable_env_encoder

        if self.enable_env_encoder:
            mlp_input_dim_a = rnn_hidden_size  # num_actor_obs + num_latent_dim
            mlp_input_dim_c = rnn_hidden_size  # num_critic_obs + num_latent_dim
        else:
            mlp_input_dim_a = rnn_hidden_size  # num_actor_obs
            mlp_input_dim_c = rnn_hidden_size  # num_critic_obs

        # Value function
        self.create_critic(mlp_input_dim_c, critic_hidden_dims, activation)

        if enable_env_encoder:
            self.memory_a = Memory(num_actor_obs + num_latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_c1 = Memory(num_critic_obs + num_latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_c2 = Memory(num_critic_obs + num_latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        else:
            self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_c1 = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
            self.memory_c2 = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic 1 RNN: {self.memory_c1}")
        print(f"Critic 2 RNN: {self.memory_c2}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c1.reset(dones)
        self.memory_c2.reset(dones)

    def create_critic(self, mlp_input_dim_c, critic_hidden_dims, activation):
        # Value function 1
        critic_layers_1 = []
        critic_layers_1.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers_1.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers_1.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers_1.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers_1.append(activation)
        self.critic_1 = nn.Sequential(*critic_layers_1)

        # Value function 2
        critic_layers_2 = []
        critic_layers_2.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers_2.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers_2.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers_2.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers_2.append(activation)
        self.critic_2 = nn.Sequential(*critic_layers_2)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c1.hidden_states, self.memory_c2.hidden_states

    def evaluate1(self, critic_obs, masks=None, hidden_states=None):
        input_c = self.memory_c1(critic_obs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def evaluate2(self, critic_obs, masks=None, hidden_states=None):
        input_c = self.memory_c2(critic_obs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def evaluate1_with_env(self, critic_obs, privileged_obs, masks=None, hidden_states=None):
        latent = self.env_encoder(privileged_obs)
        obs = torch.cat((critic_obs, latent), dim=-1)
        input_c = self.memory_c1(obs, masks, hidden_states)
        value = self.critic_1(input_c.squeeze(0))
        return value

    def evaluate2_with_env(self, critic_obs, privileged_obs, masks=None, hidden_states=None):
        latent = self.env_encoder(privileged_obs)
        obs = torch.cat((critic_obs, latent), dim=-1)
        input_c = self.memory_c2(obs, masks, hidden_states)
        value = self.critic_2(input_c.squeeze(0))
        return value

    def evaluate1_with_adaptation(self, critic_obs, obs_history, masks=None, hidden_states=None):
        latent = self.adaptation_module((obs_history))
        obs = torch.cat((critic_obs, latent), dim=-1)
        input_c = self.memory_c1(obs, masks, hidden_states)
        value = self.critic_1(input_c.squeeze(0))
        return value

    def evaluate2_with_adaptation(self, critic_obs, obs_history, masks=None, hidden_states=None):
        latent = self.adaptation_module((obs_history))
        obs = torch.cat((critic_obs, latent), dim=-1)
        input_c = self.memory_c2(obs, masks, hidden_states)
        value = self.critic_2(input_c.squeeze(0))
        return value
