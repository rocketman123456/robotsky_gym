# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.utils.torch_utils import get_activation


class ActorCritic(nn.Module):
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
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.enable_env_encoder = enable_env_encoder

        if self.enable_env_encoder:
            mlp_input_dim_a = num_actor_obs + num_latent_dim
            mlp_input_dim_c = num_critic_obs + num_latent_dim
        else:
            mlp_input_dim_a = num_actor_obs
            mlp_input_dim_c = num_critic_obs

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

        if self.enable_env_encoder:
            # create env encoder
            self.create_env_encoder(mlp_input_dim_encoder, env_encoder_hidden_dims, num_latent_dim, activation)

            # create adaptation module
            self.create_adaptation_module(mlp_input_dim_adap, adaptation_hidden_dims, num_latent_dim, activation)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False  # type: ignore

    def create_actor(self, mlp_input_dim_a, actor_hidden_dims, num_actions, activation):
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        print(f"Actor MLP: {self.actor}")

    def create_critic(self, mlp_input_dim_c, critic_hidden_dims, activation):
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        print(f"Critic MLP: {self.critic}")

    def create_env_encoder(self, mlp_input_dim_encoder, env_encoder_hidden_dims, num_latent_dim, activation):
        env_encoder_layers = []
        env_encoder_layers.append(nn.Linear(mlp_input_dim_encoder, env_encoder_hidden_dims[0]))
        env_encoder_layers.append(activation)
        for l in range(len(env_encoder_hidden_dims)):
            if l == len(env_encoder_hidden_dims) - 1:
                env_encoder_layers.append(nn.Linear(env_encoder_hidden_dims[l], num_latent_dim))
            else:
                env_encoder_layers.append(nn.Linear(env_encoder_hidden_dims[l], env_encoder_hidden_dims[l + 1]))
                env_encoder_layers.append(activation)
        self.env_encoder = nn.Sequential(*env_encoder_layers)
        print(f"Env Encoder MLP: {self.env_encoder}")

    def create_adaptation_module(self, mlp_input_dim_adap, adaptation_hidden_dims, num_latent_dim, activation):
        adaptation_layers = []
        adaptation_layers.append(nn.Linear(mlp_input_dim_adap, adaptation_hidden_dims[0]))
        adaptation_layers.append(activation)
        for l in range(len(adaptation_hidden_dims)):
            if l == len(adaptation_hidden_dims) - 1:
                adaptation_layers.append(nn.Linear(adaptation_hidden_dims[l], num_latent_dim))
            else:
                adaptation_layers.append(nn.Linear(adaptation_hidden_dims[l], adaptation_hidden_dims[l + 1]))
                adaptation_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_layers)
        print(f"Adaptation Module MLP: {self.adaptation_module}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.actor(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_with_env(self, actor_obs, privileged_obs, **kwargs):
        latent = self.env_encoder(privileged_obs)
        self.update_distribution(torch.cat((actor_obs, latent), dim=-1))
        return self.distribution.sample()

    def act_with_adaptation(self, actor_obs, obs_history, **kwargs):
        latent = self.adaptation_module((obs_history))
        self.update_distribution(torch.cat((actor_obs, latent), dim=-1))
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def encoder_inference(self, privileged_obs):
        latent = self.env_encoder(privileged_obs)
        return latent

    def adaptation_inference(self, obs_history):
        latent = self.adaptation_module(obs_history)
        return latent

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def act_with_env_inference(self, actor_obs, privileged_obs, **kwargs):
        latent = self.env_encoder(privileged_obs)
        actions_mean = self.actor(torch.cat((actor_obs, latent), dim=-1))
        return actions_mean

    def act_with_adaptation_inference(self, actor_obs, obs_history, **kwargs):
        latent = self.adaptation_module(obs_history)
        actions_mean = self.actor(torch.cat((actor_obs, latent), dim=-1))
        return actions_mean

    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value

    def evaluate_with_env(self, critic_obs, privileged_obs, **kwargs):
        latent = self.env_encoder(privileged_obs)
        value = self.critic(torch.cat((critic_obs, latent), dim=-1))
        return value

    def evaluate_with_adaptation(self, critic_obs, obs_history, **kwargs):
        latent = self.adaptation_module((obs_history))
        value = self.critic(torch.cat((critic_obs, latent), dim=-1))
        return value
