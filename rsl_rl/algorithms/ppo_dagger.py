# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPODagger:
    actor_critic: ActorCritic
    actor_critic_expert: ActorCritic

    def __init__(
        self,
        actor_critic,
        actor_critic_expert,
        num_envs,
        enable_env_encoder,
        enable_adaptation_module,
        evaluate_teacher,
        enable_dagger,
        evaluate_expert_teacher,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        vae_learning_rate=1e-3,
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
        self.enable_dagger = enable_dagger
        self.evaluate_expert_teacher = evaluate_expert_teacher

        self.num_envs = num_envs

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.vae_learning_rate = vae_learning_rate
        self.adaptation_module_learning_rate = adaptation_module_learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.actor_critic_expert = actor_critic_expert
        self.actor_critic_expert.to(self.device)
        self.storage = None  # initialized later
        self.transition = RolloutStorage.Transition()

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.adaptation_module.parameters(), lr=self.adaptation_module_learning_rate)
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
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        return self.transition.actions.clone()

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
        return self.transition.actions.clone()

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
        return self.transition.actions.clone()

    def act_expert(self, actor_obs, critic_obs):
        if self.actor_critic_expert.is_recurrent:
            self.transition.hidden_states = self.actor_critic_expert.get_hidden_states()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        # Compute the actions and values
        self.transition.actions = self.actor_critic_expert.act(actor_obs).detach()
        self.transition.values = self.actor_critic_expert.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic_expert.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic_expert.action_mean.detach()
        self.transition.action_sigma = self.actor_critic_expert.action_std.detach()
        return self.transition.actions.clone()

    def act_with_env_encoder_expert(self, actor_obs, critic_obs, privileged_obs, obs_history, action_history):
        if self.actor_critic_expert.is_recurrent:
            self.transition.hidden_states = self.actor_critic_expert.get_hidden_states()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        # add env encoder part
        self.transition.privileged_obs = privileged_obs
        self.transition.obs_history = obs_history
        self.transition.action_history = action_history
        # Compute the actions and values
        self.transition.actions = self.actor_critic_expert.act_with_env(actor_obs, privileged_obs).detach()
        self.transition.values = self.actor_critic_expert.evaluate_with_env(critic_obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic_expert.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic_expert.action_mean.detach()
        self.transition.action_sigma = self.actor_critic_expert.action_std.detach()
        return self.transition.actions.clone()

    def act_with_adaptation_module_expert(self, actor_obs, critic_obs, privileged_obs, obs_history, action_history):
        if self.actor_critic_expert.is_recurrent:
            self.transition.hidden_states = self.actor_critic_expert.get_hidden_states()
        # need to record obs and critic_obs before env.step()
        self.transition.actor_obs = actor_obs
        self.transition.critic_obs = critic_obs
        # add env encoder part
        self.transition.privileged_obs = privileged_obs
        self.transition.obs_history = obs_history
        self.transition.action_history = action_history
        # Compute the actions and values
        self.transition.actions = self.actor_critic_expert.act_with_adaptation(actor_obs, obs_history).detach()
        self.transition.values = self.actor_critic_expert.evaluate_with_adaptation(critic_obs, obs_history).detach()
        self.transition.actions_log_prob = self.actor_critic_expert.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic_expert.action_mean.detach()
        self.transition.action_sigma = self.actor_critic_expert.action_std.detach()
        return self.transition.actions.clone()

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

    def compute_returns_expert(self, last_critic_obs):
        last_values = self.actor_critic_expert.evaluate(last_critic_obs).detach()
        self.storage.compute_returns_dagger(last_values, self.gamma, self.lam)

    def compute_returns_with_env_expert(self, last_critic_obs, last_privileged_obs):
        last_values = self.actor_critic_expert.evaluate_with_env(last_critic_obs, last_privileged_obs).detach()
        self.storage.compute_returns_dagger(last_values, self.gamma, self.lam)

    def compute_returns_with_adaptation_expert(self, last_critic_obs, last_obs_history):
        last_values = self.actor_critic_expert.evaluate_with_adaptation(last_critic_obs, last_obs_history).detach()
        self.storage.compute_returns_dagger(last_values, self.gamma, self.lam)

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

            if self.enable_dagger:
                if self.enable_env_encoder:
                    if self.evaluate_teacher:
                        action = self.actor_critic.act_with_env_inference(
                            actor_obs_est_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                        )
                    else:
                        action = self.actor_critic.act_with_adaptation_inference(
                            actor_obs_est_batch, obs_history_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                        )
                else:
                    action = self.actor_critic.act_inference(actor_obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

                if self.enable_env_encoder:
                    if self.evaluate_expert_teacher:
                        action_expert = self.actor_critic_expert.act_with_env_inference(
                            actor_obs_est_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                        )
                    else:
                        action_expert = self.actor_critic_expert.act_with_adaptation_inference(
                            actor_obs_est_batch, obs_history_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                        )
                else:
                    action_expert = self.actor_critic_expert.act_inference(actor_obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0])

                loss = F.mse_loss(action_expert, action)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
            # if self.enable_env_encoder and self.enable_adaptation_module and self.evaluate_teacher:
            #     for epoch in range(self.num_adaptation_module_substeps):
            #         with torch.inference_mode():
            #             adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
            #             with torch.no_grad():
            #                 if self.enable_dagger:
            #                     adaptation_target = self.actor_critic_expert.env_encoder(privileged_obs_batch)
            #                 else:
            #                     adaptation_target = self.actor_critic.env_encoder(privileged_obs_batch)
            #         residual = (adaptation_target - adaptation_pred).norm(dim=1)
            #
            #         # Adaptation module update
            #         adaptation_loss = F.mse_loss(adaptation_target, adaptation_pred)
            #         # adaptation_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
            #         # adaptation_coef = adaptation_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]
            #
            #         # optimize adaptation module
            #         self.adaptation_module_optimizer.zero_grad()
            #         adaptation_loss.backward()
            #         self.adaptation_module_optimizer.step()
            #
            #         mean_adaptation_module_loss += adaptation_loss.item()
            #     # print(f"adaptation module residual : {residual}")

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
