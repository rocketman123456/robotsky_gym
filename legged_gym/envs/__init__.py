# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .robotsky.robotsky_wq_config import RobotSkyWQCfg, RobotSkyWQCfgPPO
from .robotsky.robotsky_wq_env import RobotSkyWQ

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register("anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
task_registry.register("anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO())
task_registry.register("robotsky_wq", RobotSkyWQ, RobotSkyWQCfg(), RobotSkyWQCfgPPO())
