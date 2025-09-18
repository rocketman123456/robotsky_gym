# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import *
from .base.legged_robot import LeggedRobot

# go2
from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO

# go2_rough
# from legged_gym.envs.go2.go2_rough.go2_rough_config import GO2RoughCfg, GO2RoughCfgPPO

# go2 walk these ways
from legged_gym.envs.go2.go2_wtw.go2_wtw import GO2WTW
from legged_gym.envs.go2.go2_wtw.go2_wtw_config import GO2WTWCfg, GO2WTWCfgPPO

# bipedal_walker
from legged_gym.envs.bipedal_walker.bipedal_walker_config import BipedalWalkerCfg, BipedalWalkerCfgPPO
from legged_gym.envs.bipedal_walker.bipedal_walker import BipedalWalker

# # go2_sysid
# from legged_gym.envs.go2.go2_sysid.go2_sysid import GO2SysID
# from legged_gym.envs.go2.go2_sysid.go2_sysid_config import GO2SysIDCfg

# go2_ts(teacher-student)
from legged_gym.envs.go2.go2_ts.go2_ts import Go2TS
from legged_gym.envs.go2.go2_ts.go2_ts_config import Go2TSCfg, Go2TSCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", GO2, GO2Cfg(), GO2CfgPPO())
# task_registry.register( "go2_rough", GO2, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register("go2_wtw", GO2WTW, GO2WTWCfg(), GO2WTWCfgPPO())
# task_registry.register( "go2_sysid", GO2SysID, GO2SysIDCfg(), GO2CfgPPO())
task_registry.register("go2_ts", Go2TS, Go2TSCfg(), Go2TSCfgPPO())
task_registry.register("bipedal_walker", BipedalWalker, BipedalWalkerCfg(), BipedalWalkerCfgPPO())
