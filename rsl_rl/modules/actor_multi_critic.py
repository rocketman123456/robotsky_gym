# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.utils.torch_utils import get_activation
from rsl_rl.modules.actor_critic import ActorCritic


class ActorCriticMultiCritic(ActorCritic):
    is_recurrent = False

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
            init_noise_std,
            **kwargs,
        )

        self.critic = None

        activation = get_activation(activation)

        if self.enable_env_encoder:
            mlp_input_dim_a = num_actor_obs + num_latent_dim
            mlp_input_dim_c = num_critic_obs + num_latent_dim
        else:
            mlp_input_dim_a = num_actor_obs
            mlp_input_dim_c = num_critic_obs

        # Value function
        self.create_critic(mlp_input_dim_c, critic_hidden_dims, activation)

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

    def evaluate1(self, critic_obs, **kwargs):
        value1 = self.critic_1(critic_obs)
        return value1

    def evaluate2(self, critic_obs, **kwargs):
        value2 = self.critic_2(critic_obs)
        return value2

    def evaluate1_with_env(self, critic_obs, privileged_obs, **kwargs):
        latent = self.env_encoder(privileged_obs)
        value1 = self.critic_1(torch.cat((critic_obs, latent), dim=-1))
        return value1

    def evaluate2_with_env(self, critic_obs, privileged_obs, **kwargs):
        latent = self.env_encoder(privileged_obs)
        value2 = self.critic_2(torch.cat((critic_obs, latent), dim=-1))
        return value2

    def evaluate1_with_adaptation(self, critic_obs, obs_history, **kwargs):
        latent = self.adaptation_module((obs_history))
        value1 = self.critic_1(torch.cat((critic_obs, latent), dim=-1))
        return value1

    def evaluate2_with_adaptation(self, critic_obs, obs_history, **kwargs):
        latent = self.adaptation_module((obs_history))
        value2 = self.critic_2(torch.cat((critic_obs, latent), dim=-1))
        return value2
