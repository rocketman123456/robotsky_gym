# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from robotsky_lab.envs.base.base_env import BaseEnv
from robotsky_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from robotsky_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from robotsky_lab.envs.gr2.gr2_config import (
    GR2FlatAgentCfg,
    GR2FlatEnvCfg,
    GR2RoughAgentCfg,
    GR2RoughEnvCfg,
)
from robotsky_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)
from robotsky_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from robotsky_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from robotsky_lab.envs.tienkung.tienkung_env import TienKungEnv
from robotsky_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from robotsky_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
from robotsky_lab.envs.robotsky.robotsky_wq_config import (
    RobotSkyWQFlatAgentCfg,
    RobotSkyWQFlatEnvCfg,
    RobotSkyWQRoughAgentCfg,
    RobotSkyWQRoughEnvCfg,
)
from robotsky_lab.envs.robotsky.robotsky_wq_env import RobotSkyWQEnv
from robotsky_lab.utils.task_registry import task_registry

task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())
task_registry.register("tienkung_walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("tienkung_run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register("tienkung_walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg())
task_registry.register("tienkung_run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg())
task_registry.register("robotsky_wq_flat", RobotSkyWQEnv, RobotSkyWQFlatEnvCfg(), RobotSkyWQFlatAgentCfg())
task_registry.register("robotsky_wq_rough", RobotSkyWQEnv, RobotSkyWQRoughEnvCfg(), RobotSkyWQRoughAgentCfg())
