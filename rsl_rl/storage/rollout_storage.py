# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.actor_obs = None
            self.critic_obs = None
            self.actions = None
            self.privileged_obs = None
            self.obs_history = None
            self.action_history = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        enable_env_encoder,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        privileged_obs_shape,
        obs_history_shape,
        action_history_shape,
        enable_mirror_data,
        multi_critics,
        device="cpu",
    ):
        self.device = device

        self.enable_env_encoder = enable_env_encoder
        self.enable_mirror_data = enable_mirror_data
        self.multi_critics = multi_critics

        assert multi_critics == False

        self.actor_obs_shape = actor_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape
        if self.enable_env_encoder:
            self.privileged_obs_shape = privileged_obs_shape
            self.obs_history_shape = obs_history_shape
            self.action_history_shape = action_history_shape

        # Core
        self.actor_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device)
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)  # no mirror
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()  # no mirror

        if self.enable_env_encoder:
            self.privileged_obs = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
            self.obs_history = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
            self.action_history = torch.zeros(num_transitions_per_env, num_envs, *action_history_shape, device=self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)  # no mirror
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)  # no mirror
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)  # no mirror
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.reward_center = 0
        self.use_reward_center = False
        self.step = 0
        self.use_mirror = False

        # mirror data
        if self.use_mirror:
            self.actor_obs_mirror = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device)
            self.critic_obs_mirror = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
            self.actions_mirror = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            if self.enable_env_encoder:
                self.privileged_obs_mirror = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
                self.obs_history_mirror = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
                self.action_history_mirror = torch.zeros(num_transitions_per_env, num_envs, *action_history_shape, device=self.device)
            self.mu_mirror = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma_mirror = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.actor_obs[self.step].copy_(transition.actor_obs)
        self.critic_obs[self.step].copy_(transition.critic_obs)
        self.actions[self.step].copy_(transition.actions)

        if self.enable_env_encoder:
            self.privileged_obs[self.step].copy_(transition.privileged_obs)
            self.obs_history[self.step].copy_(transition.obs_history)
            self.action_history[self.step].copy_(transition.action_history)

        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.actor_obs.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.actor_obs.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):
            self.reward_center = self.reward_center + 0.005 * (self.rewards[step] - self.reward_center)

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            if self.use_reward_center:
                delta = (self.rewards[step] - self.reward_center) + next_is_not_terminal * gamma * next_values - self.values[step]
            else:
                delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_returns_dagger(self, last_values, gamma, lam):
        advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):
            self.reward_center = self.reward_center + 0.005 * (self.rewards[step] - self.reward_center)

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            if self.use_reward_center:
                delta = (self.rewards[step] - self.reward_center) + next_is_not_terminal * gamma * next_values - self.values[step]
            else:
                delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        actor_obs = self.actor_obs.flatten(0, 1)
        critic_obs = self.critic_obs.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        if self.enable_env_encoder:
            privileged_obs = self.privileged_obs.flatten(0, 1)
            obs_history = self.obs_history.flatten(0, 1)
            action_history = self.action_history.flatten(0, 1)
        else:
            privileged_obs = None
            obs_history = None
            action_history = None

        if self.use_mirror:
            actor_obs_mirror = self._mirror_actor_obs(self.actor_obs).flatten(0, 1)
            critic_obs_mirror = self._mirror_critic_obs(self.critic_obs).flatten(0, 1)
            actions_mirror = self._mirror_actions(self.actions).flatten(0, 1)
            old_mu_mirror = self._mirror_mu(self.mu).flatten(0, 1)
            old_sigma_mirror = self._mirror_sigma(self.sigma).flatten(0, 1)

            if self.enable_env_encoder:
                privileged_obs_mirror = self._mirror_privileged_obs(self.privileged_obs).flatten(0, 1)
                obs_history_mirror = self._mirror_obs_history(self.obs_history, self.actor_obs).flatten(0, 1)
                action_history_mirror = self._mirror_action_history(self.action_history, self.actions).flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                actor_obs_batch = actor_obs[batch_idx]
                critic_obs_batch = critic_obs[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                if self.enable_env_encoder:
                    privileged_obs_batch = privileged_obs[batch_idx]
                    obs_history_batch = obs_history[batch_idx]
                    action_history_batch = action_history[batch_idx]
                else:
                    privileged_obs_batch = None
                    obs_history_batch = None
                    action_history_batch = None

                if self.use_mirror:
                    actor_obs_mirror_batch = actor_obs_mirror[batch_idx]
                    critic_obs_mirror_batch = critic_obs_mirror[batch_idx]
                    actions_mirror_batch = actions_mirror[batch_idx]
                    old_mu_mirror_batch = old_mu_mirror[batch_idx]
                    old_sigma_mirror_batch = old_sigma_mirror[batch_idx]

                    actor_obs_batch = torch.cat([actor_obs_batch, actor_obs_mirror_batch], dim=0)
                    critic_obs_batch = torch.cat([critic_obs_batch, critic_obs_mirror_batch], dim=0)
                    actions_batch = torch.cat([actions_batch, actions_mirror_batch], dim=0)
                    old_mu_batch = torch.cat([old_mu_batch, old_mu_mirror_batch], dim=0)
                    old_sigma_batch = torch.cat([old_sigma_batch, old_sigma_mirror_batch], dim=0)
                    returns_batch = returns_batch.repeat(2, 1)
                    advantages_batch = advantages_batch.repeat(2, 1)
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(2, 1)
                    target_values_batch = target_values_batch.repeat(2, 1)

                    if self.enable_env_encoder:
                        privileged_obs_mirror_batch = privileged_obs_mirror[batch_idx]
                        obs_history_mirror_batch = obs_history_mirror[batch_idx]
                        action_history_mirror_batch = action_history_mirror[batch_idx]

                        privileged_obs_batch = torch.cat([privileged_obs_batch, privileged_obs_mirror_batch], dim=0)
                        obs_history_batch = torch.cat([obs_history_batch, obs_history_mirror_batch], dim=0)
                        action_history_batch = torch.cat([action_history_batch, action_history_mirror_batch], dim=0)
                yield (
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
                    (None, None),
                    None,
                )

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.actor_obs, self.dones)
        padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.critic_obs, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                actor_obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                if self.enable_env_encoder:
                    padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_obs, self.dones)
                    padded_obs_history_trajectories, _ = split_and_pad_trajectories(self.obs_history, self.dones)
                    padded_action_history_trajectories, _ = split_and_pad_trajectories(self.action_history, self.dones)

                    privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                    obs_history_batch = padded_obs_history_trajectories[:, first_traj:last_traj]
                    action_history_batch = padded_action_history_trajectories[:, first_traj:last_traj]
                else:
                    privileged_obs_batch = None
                    obs_history_batch = None
                    action_history_batch = None

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_a_batch

                yield (
                    actor_obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    privileged_obs_batch,
                    obs_history_batch,
                    action_history_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (hid_a_batch, hid_c_batch),
                    masks_batch,
                )

                first_traj = last_traj

    # 四足机器人 mirror data according to the XZ plan
    # 精简代码
    # def _mirror_actor_obs(self, actor_obs):
    #     actor_obs_mirror = actor_obs.clone()
    #     # 反转特定轴的数据 (x, z轴角速度，重力y轴，控制指令y轴和z轴角速度)
    #     actor_obs_mirror[:, :, [0, 2, 4, 7, 8]] *= -1
    #     # 交换并反转腿部传感器和动作数据 (FL <-> FR, HL <-> HR)
    #     actor_obs_mirror[:, :, 9:12], actor_obs_mirror[:, :, 12:15] = actor_obs[:, :, 12:15], actor_obs[:, :, 9:12]
    #     actor_obs_mirror[:, :, 15:18], actor_obs_mirror[:, :, 18:21] = actor_obs[:, :, 18:21], actor_obs[:, :, 15:18]
    #     actor_obs_mirror[:, :, 21:24], actor_obs_mirror[:, :, 24:27] = actor_obs[:, :, 24:27], actor_obs[:, :, 21:24]
    #     actor_obs_mirror[:, :, 27:30], actor_obs_mirror[:, :, 30:33] = actor_obs[:, :, 30:33], actor_obs[:, :, 27:30]
    #     actor_obs_mirror[:, :, 33:36], actor_obs_mirror[:, :, 36:39] = actor_obs[:, :, 36:39], actor_obs[:, :, 33:36]
    #     actor_obs_mirror[:, :, 39:42], actor_obs_mirror[:, :, 42:45] = actor_obs[:, :, 42:45], actor_obs[:, :, 39:42]
    #     # 反转x, z轴的电机控制
    #     actor_obs_mirror[:, :, [9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]] *= -1
    #     return actor_obs_mirror

    # mirror data according to the XZ plane
    def _mirror_actor_obs(self, actor_obs):
        actor_obs_mirror = actor_obs.clone()
        actor_obs_mirror[:, :, [0, 2]] = -actor_obs[:, :, [0, 2]]  # x,z轴的角速度是相反的
        actor_obs_mirror[:, :, 4] = -actor_obs[:, :, 4]  # 重力的y轴是相反的
        actor_obs_mirror[:, :, [7, 8]] = -actor_obs[:, :, [7, 8]]  # 控制指令的y轴是相反的，z的角速度是相反的
        # 交替一下左右腿
        actor_obs_mirror[:, :, 9:14] = actor_obs[:, :, 14:19]
        actor_obs_mirror[:, :, 14:19] = actor_obs[:, :, 9:14]
        actor_obs_mirror[:, :, 19:24] = actor_obs[:, :, 24:29]
        actor_obs_mirror[:, :, 24:29] = actor_obs[:, :, 19:24]
        actor_obs_mirror[:, :, 29:34] = actor_obs[:, :, 34:39]
        actor_obs_mirror[:, :, 34:39] = actor_obs[:, :, 29:34]
        # 轴的朝向是x和z轴的电机是相反的
        actor_obs_mirror[:, :, [9, 10, 14, 15]] = -actor_obs_mirror[:, :, [9, 10, 14, 15]]
        actor_obs_mirror[:, :, [19, 20, 24, 25]] = -actor_obs_mirror[:, :, [19, 20, 24, 25]]
        actor_obs_mirror[:, :, [29, 30, 34, 35]] = -actor_obs_mirror[:, :, [29, 30, 34, 35]]
        return actor_obs_mirror

    # def _mirror_critic_obs(self, critic_obs):
    #     critic_obs_mirror = critic_obs.clone()
    #     # 反转指定轴的数据
    #     critic_obs_mirror[:, :, [1, 3, 5, 7, 10, 11, 49, 50, 52, 54, 56, 58]] *= -1
    #     # 交替腿部传感器和动作数据 (FL <-> FR, HL <-> HR)
    #     swap_indices = [(12, 15), (18, 21), (24, 27), (30, 33), (36, 39), (42, 45), (62, 65), (68, 71), (74, 77), (80, 83), (86, 89), (92, 95)]
    #     for i, j in swap_indices:
    #         critic_obs_mirror[:, :, i:j], critic_obs_mirror[:, :, j : j + 3] = critic_obs[:, :, j : j + 3], critic_obs[:, :, i : i + 3]
    #     # 反转电机的x, y轴方向 (只在指定的电机部分)
    #     critic_obs_mirror[:, :, [12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]] *= -1
    #     return critic_obs_mirror

    def _mirror_critic_obs(self, critic_obs):
        critic_obs_mirror = critic_obs.clone()
        # 线速度 y轴是反的
        critic_obs_mirror[:, :, 1] = -critic_obs[:, :, 1]
        # 角速度 x,z轴是相反的
        critic_obs_mirror[:, :, [3, 5]] = -critic_obs[:, :, [3, 5]]
        # 重力向量 y轴是相反的
        critic_obs_mirror[:, :, 7] = -critic_obs[:, :, 7]
        # 控制指令的y轴是相反的，z的角速度是相反的
        critic_obs_mirror[:, :, [10, 11]] = -critic_obs[:, :, [10, 11]]
        # 交替一下左右腿
        critic_obs_mirror[:, :, 12:17] = critic_obs[:, :, 17:22]
        critic_obs_mirror[:, :, 17:22] = critic_obs[:, :, 12:17]
        critic_obs_mirror[:, :, 22:27] = critic_obs[:, :, 27:32]
        critic_obs_mirror[:, :, 27:32] = critic_obs[:, :, 22:27]
        critic_obs_mirror[:, :, 32:37] = critic_obs[:, :, 37:42]
        critic_obs_mirror[:, :, 37:42] = critic_obs[:, :, 32:37]
        # 轴的朝向是x和y轴的电机是相反的
        critic_obs_mirror[:, :, [12, 13, 17, 18]] = -critic_obs_mirror[:, :, [12, 13, 17, 18]]
        critic_obs_mirror[:, :, [22, 23, 27, 28]] = -critic_obs_mirror[:, :, [22, 23, 27, 28]]
        critic_obs_mirror[:, :, [32, 33, 37, 38]] = -critic_obs_mirror[:, :, [32, 33, 37, 38]]
        # 线速度扰动 y轴是反的
        critic_obs_mirror[:, :, 43] = -critic_obs[:, :, 43]
        # 角速度扰动 x,z轴是反的
        critic_obs_mirror[:, :, [44, 46]] = -critic_obs[:, :, [44, 46]]
        # 外力扰动 y轴是反的
        critic_obs_mirror[:, :, 48] = -critic_obs[:, :, 48]
        # 力矩扰动 x,z轴是反的
        critic_obs_mirror[:, :, [50, 52]] = -critic_obs[:, :, [50, 52]]
        # 交替左右腿
        # 电机力矩系数
        critic_obs_mirror[:, :, 56:61] = critic_obs[:, :, 61:66]
        critic_obs_mirror[:, :, 61:66] = critic_obs[:, :, 56:61]
        # 电机刚度系数
        critic_obs_mirror[:, :, 66:71] = critic_obs[:, :, 71:76]
        critic_obs_mirror[:, :, 71:76] = critic_obs[:, :, 66:71]
        # 电机阻尼系数
        critic_obs_mirror[:, :, 76:81] = critic_obs[:, :, 81:86]
        critic_obs_mirror[:, :, 81:86] = critic_obs[:, :, 76:81]

        return critic_obs_mirror

    def _mirror_actions(self, actions):
        actions_mirror = actions.clone()
        # 交替左右腿
        actions_mirror[:, :, 0:5] = actions[:, :, 5:10]
        actions_mirror[:, :, 5:10] = actions[:, :, 0:5]
        # 轴的朝向是x和y轴的电机是相反的
        actions_mirror[:, :, [0, 1, 5, 6]] = -actions_mirror[:, :, [0, 1, 5, 6]]
        return actions_mirror

    def _mirror_privileged_obs(self, privileged_obs):
        privileged_obs_mirror = privileged_obs.clone()
        # 和critic_obs_mirror的顺序一致
        # 线速度 y轴是反的
        privileged_obs_mirror[:, :, 1] = -privileged_obs[:, :, 1]
        # 角速度 x,z轴是相反的
        privileged_obs_mirror[:, :, [3, 5]] = -privileged_obs[:, :, [3, 5]]
        # 重力向量 y轴是相反的
        privileged_obs_mirror[:, :, 7] = -privileged_obs[:, :, 7]
        # 控制指令的y轴是相反的，z的角速度是相反的
        privileged_obs_mirror[:, :, [10, 11]] = -privileged_obs[:, :, [10, 11]]
        # 交替一下左右腿
        privileged_obs_mirror[:, :, 12:17] = privileged_obs[:, :, 17:22]
        privileged_obs_mirror[:, :, 17:22] = privileged_obs[:, :, 12:17]
        privileged_obs_mirror[:, :, 22:27] = privileged_obs[:, :, 27:32]
        privileged_obs_mirror[:, :, 27:32] = privileged_obs[:, :, 22:27]
        privileged_obs_mirror[:, :, 32:37] = privileged_obs[:, :, 37:42]
        privileged_obs_mirror[:, :, 37:42] = privileged_obs[:, :, 32:37]
        # 轴的朝向是x和y轴的电机是相反的
        privileged_obs_mirror[:, :, [12, 13, 17, 18]] = -privileged_obs_mirror[:, :, [12, 13, 17, 18]]
        privileged_obs_mirror[:, :, [22, 23, 27, 28]] = -privileged_obs_mirror[:, :, [22, 23, 27, 28]]
        privileged_obs_mirror[:, :, [32, 33, 37, 38]] = -privileged_obs_mirror[:, :, [32, 33, 37, 38]]
        # 线速度扰动 y轴是反的
        privileged_obs_mirror[:, :, 43] = -privileged_obs[:, :, 43]
        # 角速度扰动 x,z轴是反的
        privileged_obs_mirror[:, :, [44, 46]] = -privileged_obs[:, :, [44, 46]]
        # 外力扰动 y轴是反的
        privileged_obs_mirror[:, :, 48] = -privileged_obs[:, :, 48]
        # 力矩扰动 x,z轴是反的
        privileged_obs_mirror[:, :, [50, 52]] = -privileged_obs[:, :, [50, 52]]
        # 交替左右腿
        # 电机力矩系数
        privileged_obs_mirror[:, :, 56:61] = privileged_obs[:, :, 61:66]
        privileged_obs_mirror[:, :, 61:66] = privileged_obs[:, :, 56:61]
        # 电机刚度系数
        privileged_obs_mirror[:, :, 66:71] = privileged_obs[:, :, 71:76]
        privileged_obs_mirror[:, :, 71:76] = privileged_obs[:, :, 66:71]
        # 电机阻尼系数
        privileged_obs_mirror[:, :, 76:81] = privileged_obs[:, :, 81:86]
        privileged_obs_mirror[:, :, 81:86] = privileged_obs[:, :, 76:81]
        return privileged_obs_mirror

    def _mirror_obs_history(self, obs_history, actor_obs):
        obs_history_mirror = obs_history.clone()
        obs_history_length = obs_history.shape[-1]
        obs_length = actor_obs.shape[-1]
        history_nums = obs_history_length // obs_length
        for i in range(history_nums):
            obs_history_mirror[:, :, i * obs_length : (i + 1) * obs_length] = self._mirror_actor_obs(obs_history[:, :, i * obs_length : (i + 1) * obs_length])
        return obs_history_mirror

    def _mirror_action_history(self, action_history, actions):
        action_history_mirror = action_history.clone()
        action_history_length = action_history.shape[-1]
        action_length = actions.shape[-1]
        history_nums = action_history_length // action_length
        for i in range(history_nums):
            action_history_mirror[:, :, i * action_length : (i + 1) * action_length] = self._mirror_actions(
                action_history[:, :, i * action_length : (i + 1) * action_length]
            )
        return action_history_mirror

    def _mirror_mu(self, mu):
        mu_mirror = mu.clone()
        # 交替左右腿
        mu_mirror[:, :, 0:5] = mu[:, :, 5:10]
        mu_mirror[:, :, 5:10] = mu[:, :, 0:5]
        return mu_mirror

    def _mirror_sigma(self, sigma):
        sigma_mirror = sigma.clone()
        # 交替左右腿
        sigma_mirror[:, :, 0:5] = sigma[:, :, 5:10]
        sigma_mirror[:, :, 5:10] = sigma[:, :, 0:5]
        return sigma_mirror
