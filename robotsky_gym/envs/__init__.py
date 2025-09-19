# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import os

from robotsky_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .base.legged_robot import LeggedRobot

# from .anymal_c.anymal import Anymal
# from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
# from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
# from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
# from .cassie.cassie import Cassie
# from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
# from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
# from .go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO
# from .pointfoot.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO

from .unitree_g1.g1_config import G1Cfg, G1CfgPPO
from .unitree_g1.g1_rough.g1_rough_config import G1RoughCfg, G1RoughCfgPPO
from .unitree_g1.g1 import G1

from robotsky_gym.utils.task_registry import task_registry

# # task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
# task_registry.register("anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO())
# # task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
# task_registry.register("a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
# task_registry.register("go2", LeggedRobot, Go2RoughCfg(), Go2RoughCfgPPO())
# task_registry.register("cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO())
# # task_registry.register( "g1", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO() )
# task_registry.register("pointfoot", LeggedRobot, PointFootRoughCfg(), PointFootRoughCfgPPO())

task_registry.register("g1", G1, G1Cfg(), G1CfgPPO())
task_registry.register("g1_rough", G1, G1RoughCfg(), G1RoughCfgPPO())
