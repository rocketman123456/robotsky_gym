# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage, RolloutStorageMultiCritic


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_envs,
        enable_env_encoder,
        enable_adaptation_module,
        evaluate_teacher,
        enable_dagger=False,
        evaluate_expert_teacher=False,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        adaptation_module_learning_rate=1.0e-3,
        num_adaptation_module_substeps=2,
        enable_mirror_data=False,
        multi_critics=False,  # multi critic option
        grad_penalty_coef_schedule=[0, 0, 0],  # LCP option
        device="cpu",
    ):
        assert multi_critics == False

        self.device = device

        self.multi_critics = multi_critics
        self.enable_env_encoder = enable_env_encoder
        self.enable_adaptation_module = enable_adaptation_module
        self.evaluate_teacher = evaluate_teacher
        self.enable_mirror_data = enable_mirror_data

        self.num_envs = num_envs

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.vae_learning_rate = 1e-3  # TODO
        self.adaptation_module_learning_rate = adaptation_module_learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.transition = RolloutStorage.Transition()

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.adaptation_module_learning_rate)
        # self.vae_optimizer = optim.Adam(self.actor_critic.vae.parameters(), lr=self.vae_learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.num_adaptation_module_substeps = num_adaptation_module_substeps

        # PPO Extra
        self.grad_penalty_coef_schedule = grad_penalty_coef_schedule
        self.counter = 0

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            self.enable_env_encoder,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            None,
            None,
            None,
            self.enable_mirror_data,
            self.multi_critics,
            self.device,
        )

    def init_storage_with_env_encoder(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        privileged_obs_shape,
        obs_history_shape,
        action_history_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            self.enable_env_encoder,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            privileged_obs_shape,
            obs_history_shape,
            action_history_shape,
            self.enable_mirror_data,
            self.multi_critics,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, actor_obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(actor_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        # self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        return self.transition.actions

    def act_with_env_encoder(self, actor_obs, critic_obs, privileged_obs, obs_history, action_history):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        # add env encoder part
        self.transition.privileged_obs = privileged_obs
        self.transition.obs_history = obs_history
        self.transition.action_history = action_history
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act_with_env(actor_obs, privileged_obs).detach()
        self.transition.values = self.actor_critic.evaluate_with_env(critic_obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        return self.transition.actions

    def act_with_adaptation_module(self, actor_obs, critic_obs, privileged_obs, obs_history, action_history):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        # add env encoder part
        self.transition.privileged_obs = privileged_obs
        self.transition.obs_history = obs_history
        self.transition.action_history = action_history
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act_with_adaptation(actor_obs, obs_history).detach()
        self.transition.values = self.actor_critic.evaluate_with_adaptation(critic_obs, obs_history).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_returns_with_env(self, last_critic_obs, last_privileged_obs):
        last_values = self.actor_critic.evaluate_with_env(last_critic_obs, last_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_returns_with_adaptation(self, last_critic_obs, last_obs_history):
        last_values = self.actor_critic.evaluate_with_adaptation(last_critic_obs, last_obs_history).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_grad_penalty(self, obs_batch, actions_log_prob_batch):
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss

    def update_counter(self):
        self.counter += 1

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_symmetry_loss = 0

        # vae
        mean_recons_loss = 0
        mean_vel_loss = 0
        mean_kld_loss = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            actor_obs_batch,
            critic_obs_batch,
            actions_batch,
            privileged_obs_batch,
            obs_history_batch,
            action_history_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            # base_vel_batch, # TODO
            # dones_batch,
            # next_obs_batch,
        ) in generator:

            actor_obs_est_batch = actor_obs_batch.clone()
            actor_obs_est_batch.requires_grad_()

            # 用于更新normal distribution,然后用于计算log prob
            if self.enable_env_encoder:
                if self.evaluate_teacher:
                    self.actor_critic.act_with_env(actor_obs_est_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                else:
                    self.actor_critic.act_with_adaptation(actor_obs_est_batch, obs_history_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            else:
                self.actor_critic.act(actor_obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

            if self.enable_env_encoder:
                if self.evaluate_teacher:
                    value_batch = self.actor_critic.evaluate_with_env(
                        critic_obs_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                    )
                else:
                    value_batch = self.actor_critic.evaluate_with_adaptation(
                        critic_obs_batch, obs_history_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
                    )
            else:
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_1 = torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    kl_2 = (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch))
                    kl = torch.sum(kl_1 + kl_2 - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-4, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 镜像loss，镜像观测通过policy输出的动作 和 镜像动作之间的差
            # if self.enable_env_encoder:
            #     if self.evaluate_teacher:
            #         action_mirror_obs_batch = self.actor_critic.act_with_env(actor_obs_mirror_batch, privileged_obs_mirror_batch)
            # symmetry_loss = torch.sum((action_mirror_obs_batch - actions_mirror_batch).pow(2), dim=1).mean()

            # Calculate the gradient penalty loss
            gradient_penalty_loss = self.compute_grad_penalty(actor_obs_est_batch, actions_log_prob_batch)
            gradient_stage = min(max((self.counter - self.grad_penalty_coef_schedule[2]), 0) / self.grad_penalty_coef_schedule[3], 1.0)
            gradient_penalty_coef = (
                gradient_stage * (self.grad_penalty_coef_schedule[1] - self.grad_penalty_coef_schedule[0]) + self.grad_penalty_coef_schedule[0]
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + gradient_penalty_coef * gradient_penalty_loss
                # + symmetry_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            # mean_symmetry_loss += symmetry_loss

            #! Training VAE
            # if self.enable_env_encoder and self.enable_adaptation_module and self.evaluate_teacher:
            #     for epoch in range(self.cfg.num_adaptation_module_substeps):
            #         #! 需要 next obs
            #         #! 通过 dones 来隔断训练
            #         self.vae_optimizer.zero_grad()
            #         vae_loss_dict = self.actor_critic.vae.loss_fn(obs_history_batch, next_obs_batch, base_vel_batch, self.kl_weight)
            #         valid = (dones_batch == 0).squeeze()
            #         vae_loss = torch.mean(vae_loss_dict["loss"][valid])
            #         vae_loss.backward()
            #         nn.utils.clip_grad_norm_(self.actor_critic.vae.parameters(), self.cfg.max_grad_norm)
            #         self.vae_optimizer.step()
            #         with torch.no_grad():
            #             recons_loss = torch.mean(vae_loss_dict["recons_loss"][valid])
            #             vel_loss = torch.mean(vae_loss_dict["vel_loss"][valid])
            #             kld_loss = torch.mean(vae_loss_dict["kld_loss"][valid])
            #         mean_recons_loss += recons_loss.item()
            #         mean_vel_loss += vel_loss.item()
            #         mean_kld_loss += kld_loss.item()

            # add adaptation optimize step
            if self.enable_env_encoder and self.enable_adaptation_module and self.evaluate_teacher:
                for epoch in range(self.num_adaptation_module_substeps):
                    adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                    with torch.no_grad():
                        adaptation_target = self.actor_critic.env_encoder(privileged_obs_batch)
                        residual = (adaptation_target - adaptation_pred).norm(dim=1)

                    adaptation_loss = F.mse_loss(adaptation_pred, adaptation_target)

                    # # Adaptation module update
                    # priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                    # with torch.inference_mode():
                    #     hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                    # priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                    # priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                    # priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

                    # optimize adaptation module
                    self.adaptation_module_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_module_optimizer.step()

                    mean_adaptation_module_loss += adaptation_loss.item()
                # print(f"adaptation module residual : {residual}")

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= num_updates * self.num_adaptation_module_substeps
        mean_symmetry_loss /= num_updates
        mean_recons_loss /= num_updates * self.num_adaptation_module_substeps
        mean_vel_loss /= num_updates * self.num_adaptation_module_substeps
        mean_kld_loss /= num_updates * self.num_adaptation_module_substeps

        self.storage.clear()
        self.update_counter()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_symmetry_loss, mean_recons_loss, mean_vel_loss, mean_kld_loss
