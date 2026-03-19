# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.robotsky_wq.robotsky_wq import ROBOTSKY_WQ_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    HeightScannerCfg,
    NormalizationCfg,
    ObsScalesCfg,
    RewardCfg,
    RobotCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class WheelLeggedRobotCfg(RobotCfg):
    """Extends RobotCfg with wheel-specific fields for mixed position/velocity control."""

    wheel_joint_names: list = [".*Wheel_Joint.*"]
    # Velocity command scale for wheel joints (rad/s per unit action)
    wheel_action_scale: float = 10.0


@configclass
class RobotSkyWQRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*Knee_Link.*", ".*Hip_Link.*"]),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Link.*"),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.15,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Link.*"),
    #         "threshold": 0.4,
    #     },
    # )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Link.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Wheel_Link.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Link.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*Wheel_Link.*"])},
    )
    # Penalise roll joints straying from default (lateral stability)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Roll_Joint.*"])},
    )
    # Penalise leg joints straying from default (energy efficiency)
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    joint_velocity_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    wheel_velocity_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Wheel_Joint.*"])},
    )


@configclass
class RobotSkyWQFlatEnvCfg(BaseEnvCfg):

    reward: RobotSkyWQRewardCfg = RobotSkyWQRewardCfg()
    robot: WheelLeggedRobotCfg = WheelLeggedRobotCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ROBOTSKY_WQ_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene.height_scanner.prim_body_name = "base_link"
        self.robot.terminate_contacts_body_names = [".*base_link.*"]
        self.robot.feet_body_names = [".*Wheel_Link.*"]
        self.robot.wheel_joint_names = [".*Wheel_Joint.*"]
        self.robot.wheel_action_scale = 10.0
        self.robot.action_scale = 0.25
        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*base_link.*"]
        # Wheel joints can spin up to ~12 rad/s; scale to ~±1.2 range in obs
        self.normalization.obs_scales.wheel_vel = 0.1


@configclass
class RobotSkyWQFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "robotsky_wq_flat"
    wandb_project: str = "robotsky_wq_flat"


@configclass
class RobotSkyWQRoughEnvCfg(RobotSkyWQFlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = False
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 5
        self.robot.critic_obs_history_length = 5
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class RobotSkyWQRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "robotsky_wq_rough"
    wandb_project: str = "robotsky_wq_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
