from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import torch.nn.functional as F


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_robots_force = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.asset.fix_base_link = True
    env_cfg.domain_rand.add_random = False
    env_cfg.init_state.pos = [0.0, 0.0, 0.8]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_actor_obs()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    if env.enable_env_encoder:
        module = ppo_runner.get_adaptation_module(device=env.device)
        encoder = ppo_runner.get_env_encoder(device=env.device)
        privileged_obs = env.get_privileged_obs()
        obs_history = env.get_obs_history()
        action_history = env.get_action_history()

    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array([0.0, -3.0, 2.0], dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array([0.0, 0.0, 0.0]) - np.array(env_cfg.viewer.pos)
    env.set_camera(camera_position, camera_position + camera_direction)
    img_idx = 0
    actions = torch.zeros([env_cfg.env.num_envs, 10])
    actions[:, 0] = 1.0
    actions[:, 1] = 1.0
    actions[:, 2] = -1.5

    for i in range(10 * int(env.max_episode_length)):
        actions[1, 0:5] = actions[0, 5:10]
        actions[1, 5:10] = actions[0, 0:5]
        actions[1, [0, 1, 5, 6]] = -actions[1, [0, 1, 5, 6]]

        (
            obs,
            critic_obs,
            privileged_obs,
            obs_history,
            action_history,
            rews,
            dones,
            infos,
        ) = env.step(actions.detach())

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)


if __name__ == "__main__":

    MOVE_CAMERA = False
    args = get_args()
    play(args)
