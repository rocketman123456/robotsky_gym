# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.assets import ISAAC_ASSET_DIR

ROBOTSKY_WQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/robotsky_wq/usd/robotsky_wq.usd",
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
            ".*RF_Roll_Joint.*": 0.1,
            ".*RF_Hip_Joint.*": -0.5,
            ".*RF_Knee_Joint.*": 1.0,
            ".*RF_Wheel_Joint.*": 0.0,
            ".*LF_Roll_Joint.*": -0.1,
            ".*LF_Hip_Joint.*": -0.5,
            ".*LF_Knee_Joint.*": 1.0,
            ".*LF_Wheel_Joint.*": 0.0,
            ".*RB_Roll_Joint.*": 0.1,
            ".*RB_Hip_Joint.*": 0.5,
            ".*RB_Knee_Joint.*": -1.0,
            ".*RB_Wheel_Joint.*": 0.0,
            ".*LB_Roll_Joint.*": -0.1,
            ".*LB_Hip_Joint.*": 0.5,
            ".*LB_Knee_Joint.*": -1.0,
            ".*LB_Wheel_Joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": ImplicitActuatorCfg(
            joint_names_expr=[".*Roll_Joint.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=12.0,
            stiffness={".*Roll_Joint.*": 20.0},
            damping={".*Roll_Joint.*": 1.0},
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Hip_Joint.*",
                ".*Knee_Joint.*",
            ],
            effort_limit_sim=14.0,
            velocity_limit_sim=20.0,
            stiffness={
                ".*Hip_Joint.*": 20.0,
                ".*Knee_Joint.*": 40.0,
            },
            damping={
                ".*Hip_Joint.*": 1.0,
                ".*Knee_Joint.*": 1.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*Wheel_Joint.*"],
            effort_limit_sim=7.0,
            velocity_limit_sim=12.0,
            stiffness={".*Wheel_Joint.*": 0.0},
            damping={".*Wheel_Joint.*": 1.0},
        ),
    },
)
