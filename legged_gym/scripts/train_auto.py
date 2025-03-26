# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from datetime import datetime

import json
import torch
import wandb


def train(args, env_cfg_extra, train_cfg_extra):
    env, env_cfg = task_registry.make_env_extra(name=args.task, args=args, extra_cfg=env_cfg_extra)
    ppo_runner, train_cfg = task_registry.make_alg_runner_extra(env=env, name=args.task, args=args, extra_cfg=train_cfg_extra)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

    env.destroy()
    del env
    del ppo_runner

    torch.cuda.empty_cache()

    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    print(f"Max allocated memory: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
    # print("")


def prepare_cfgs():
    configs = []
    configs.append(
        {
            "env-cfg": "legged_gym/config/env-cfg-01.json",
            "train-cfg": "legged_gym/config/train-cfg-01.json",
        }
    )
    # configs.append(
    #     {
    #         "env-cfg": "legged_gym/config/env-cfg-02.json",
    #         "train-cfg": "legged_gym/config/train-cfg-02.json",
    #     }
    # )
    # configs.append(
    #     {
    #         "env-cfg": "legged_gym/config/env-cfg-03.json",
    #         "train-cfg": "legged_gym/config/train-cfg-03.json",
    #     }
    # )
    # configs.append(
    #     {
    #         "env-cfg": "legged_gym/config/env-cfg-04.json",
    #         "train-cfg": "legged_gym/config/train-cfg-04.json",
    #     }
    # )
    # configs.append(
    #     {
    #         "env-cfg": "legged_gym/config/env-cfg-05.json",
    #         "train-cfg": "legged_gym/config/train-cfg-05.json",
    #     }
    # )

    env_cfgs = []
    train_cfgs = []

    for config in configs:
        with open(config["env-cfg"], "r") as file:
            env_cfg = json.load(file)

        with open(config["train-cfg"], "r") as file:
            train_cfg = json.load(file)

        env_cfgs.append(env_cfg)
        train_cfgs.append(train_cfg)

    return configs, env_cfgs, train_cfgs


if __name__ == "__main__":
    args = get_args()

    config_files, env_cfgs, train_cfgs = prepare_cfgs()

    for i in range(len(config_files)):
        train(args, env_cfgs[i], train_cfgs[i])
