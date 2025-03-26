# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
import wandb

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import *
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
from legged_gym.utils.helpers import class_to_dict
from datetime import datetime


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.all_cfg["reward"] = class_to_dict(env.cfg)
        self.wandb_run_name = datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg["runner"]["experiment_name"] + "_" + train_cfg["runner"]["run_name"]
        self.device = device
        self.env = env
        self.multi_critics = self.alg_cfg["multi_critics"]
        self.calc_mirror_action = False

        if self.multi_critics:
            actor_critic_class = eval(self.cfg["policy_class_name"] + "MultiCritic")  # ActorCritic MultiCritic
        else:
            actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_actor_obs,
            self.env.num_critic_obs,
            self.env.enable_env_encoder,
            self.env.num_privileged_obs,
            self.env.num_obs_history,
            self.env.num_action_history,
            self.env.num_actions,
            self.env.num_latent_dim,
            **self.policy_cfg,
        ).to(self.device)

        if self.multi_critics:
            alg_class = eval(self.cfg["algorithm_class_name"] + "MultiCritic")  # PPO
        else:
            alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(
            actor_critic,
            self.env.num_envs,
            self.env.enable_env_encoder,
            self.env.enable_adaptation_module,
            self.env.evaluate_teacher,
            device=self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        if self.env.enable_env_encoder:
            self.alg.init_storage_with_env_encoder(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.env.num_actor_obs],
                [self.env.num_critic_obs],
                [self.env.num_actions],
                [self.env.num_privileged_obs],
                [self.env.num_obs_history],
                [self.env.num_action_history],
            )
        else:
            self.alg.init_storage(
                self.env.num_envs,
                self.num_steps_per_env,
                [self.env.num_actor_obs],
                [self.env.num_critic_obs],
                [self.env.num_actions],
            )

        if self.env.enable_env_encoder:
            if not self.env.evaluate_teacher:
                # 固定actor_critic中的env_encoder
                for param in self.alg.actor_critic.env_encoder.parameters():
                    param.requires_grad = False
            if not self.env.enable_adaptation_module:
                # 固定actor_critic中的adaptation_module
                for param in self.alg.actor_critic.adaptation_module.parameters():
                    param.requires_grad = False

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # self.run = wandb.init(
            #     group="srobot",
            #     project=self.cfg["experiment_name"],
            #     sync_tensorboard=True,
            #     name=self.wandb_run_name,
            #     config=self.all_cfg,
            #     reinit=True,
            # )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        actor_obs = self.env.get_actor_obs()
        critic_obs = self.env.get_critic_obs()
        actor_obs, critic_obs = actor_obs.to(self.device), critic_obs.to(self.device)

        if self.env.enable_env_encoder:
            privileged_obs = self.env.get_privileged_obs()
            obs_history = self.env.get_obs_history()
            action_history = self.env.get_action_history()

            privileged_obs = privileged_obs.to(self.device)
            obs_history = obs_history.to(self.device)
            action_history = action_history.to(self.device)
        else:
            privileged_obs = None
            obs_history = None
            action_history = None

        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self.env.enable_env_encoder:
                        if self.env.evaluate_teacher:
                            actions = self.alg.act_with_env_encoder(actor_obs, critic_obs, privileged_obs, obs_history, action_history)
                        else:
                            actions = self.alg.act_with_adaptation_module(actor_obs, critic_obs, privileged_obs, obs_history, action_history)
                    else:
                        actions = self.alg.act(actor_obs, critic_obs)

                    if self.multi_critics:
                        (
                            actor_obs,
                            critic_obs,
                            privileged_obs,
                            obs_history,
                            action_history,
                            rewards1,
                            rewards2,
                            dones,
                            infos,
                        ) = self.env.step(actions)

                        actor_obs, critic_obs, rewards1, rewards2, dones = (
                            actor_obs.to(self.device),
                            critic_obs.to(self.device),
                            rewards1.to(self.device),
                            rewards2.to(self.device),
                            dones.to(self.device),
                        )
                    else:
                        (
                            actor_obs,
                            critic_obs,
                            privileged_obs,
                            obs_history,
                            action_history,
                            rewards,
                            dones,
                            infos,
                        ) = self.env.step(actions)

                        actor_obs, critic_obs, rewards, dones = (
                            actor_obs.to(self.device),
                            critic_obs.to(self.device),
                            rewards.to(self.device),
                            dones.to(self.device),
                        )

                    if self.env.enable_env_encoder:
                        privileged_obs = privileged_obs.to(self.device)
                        obs_history = obs_history.to(self.device)
                        action_history = action_history.to(self.device)

                    if self.multi_critics:
                        self.alg.process_env_step(rewards1, rewards2, dones, infos)
                    else:
                        self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        if self.multi_critics:
                            cur_reward_sum += rewards1 + rewards2
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if self.env.enable_env_encoder:
                    if self.env.evaluate_teacher:
                        self.alg.compute_returns_with_env(critic_obs, privileged_obs)
                    else:
                        self.alg.compute_returns_with_env(critic_obs, privileged_obs)
                else:
                    self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_adaptation_module_loss,
                mean_symmetry_loss,
                mean_recons_loss,
                mean_vel_loss,
                mean_kld_loss,
            ) = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/adapt_module", locs["mean_adaptation_module_loss"], locs["it"])
        self.writer.add_scalar("Loss/symmetry", locs["mean_symmetry_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Adaptation module loss:':>{pad}} {locs['mean_adaptation_module_loss']:.4f}\n"""
                f"""{'Symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )
        # artifact = wandb.Artifact("model", type="model")
        # artifact.add_file(path)
        # self.run.log_artifact(artifact)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        # 固定env_encoder和adaptation_module的参数
        if self.env.enable_env_encoder:
            if not self.env.evaluate_teacher:
                # 固定actor_critic中的env_encoder
                for param in self.alg.actor_critic.env_encoder.parameters():
                    param.requires_grad = False
            if not self.env.enable_adaptation_module:
                # 固定actor_critic中的adaptation_module
                for param in self.alg.actor_critic.adaptation_module.parameters():
                    param.requires_grad = False

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_adaptation_module(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.adaptation_inference

    def get_env_encoder(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.encoder_inference
