# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR

SROBOT_WHEEL_LEGGED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/srobot_wheel_legged/usd/srobot-wheel-legged.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*F_Roll_Joint.*": -0.1,
            ".*F_Hip_Joint.*": 0.6,
            ".*F_Knee_Joint.*": -1.2,
            ".*B_Roll_Joint.*": -0.1,
            ".*B_Hip_Joint.*": -0.6,
            ".*B_Knee_Joint.*": 1.2,
            ".*Wheel_Joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Roll_Joint.*",
                ".*Hip_Joint.*",
                ".*Knee_Joint.*",
                # ".*Wheel_Joint.*",
            ],
            effort_limit_sim=24.0,
            velocity_limit_sim=30.0,
            stiffness={
                ".*Roll_Joint.*": 40.0,
                ".*Hip_Joint.*": 40.0,
                ".*Knee_Joint.*": 80.0,
                # ".*Wheel_Joint.*": 5.0,
            },
            damping={
                ".*Roll_Joint.*": 5.0,
                ".*Hip_Joint.*": 5.0,
                ".*Knee_Joint.*": 5.0,
                # ".*Wheel_Joint.*": 1.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*Wheel_Joint.*"],
            effort_limit_sim=3.0,
            velocity_limit_sim=100.0,
            stiffness={".*Wheel_Joint.*": 5.0},
            damping={".*Wheel_Joint.*": 1.0},
        ),
    },
)
