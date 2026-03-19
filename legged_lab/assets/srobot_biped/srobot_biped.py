# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR

SROBOT_BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/srobot_biped/usd/srobot-biped.usd",
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
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*HipYaw_Joint.*": 0.0,
            ".*HipRoll_Joint.*": 0.0,
            ".*HipPitch_Joint.*": -0.3,
            ".*Knee_Joint.*": 0.6,
            ".*Ankle_Joint.*": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*HipYaw_Joint.*",
                ".*HipRoll_Joint.*",
                ".*HipPitch_Joint.*",
                ".*Knee_Joint.*",
            ],
            effort_limit_sim=12.0,
            velocity_limit_sim=30.0,
            stiffness={
                ".*HipYaw_Joint.*": 15.0,
                ".*HipRoll_Joint.*": 15.0,
                ".*HipPitch_Joint.*": 15.0,
                ".*Knee_Joint.*": 15.0,
            },
            damping={
                ".*HipYaw_Joint.*": 1.0,
                ".*HipRoll_Joint.*": 1.0,
                ".*HipPitch_Joint.*": 1.0,
                ".*Knee_Joint.*": 1.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*Ankle_Joint.*"],
            effort_limit_sim=12.0,
            velocity_limit_sim=30.0,
            stiffness={".*Ankle_Joint.*": 15.0},
            damping={".*Ankle_Joint.*": 1.0},
        ),
    },
)
