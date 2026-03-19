from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseEnvCfg,
    BaseAgentCfg,
    BaseSceneCfg,
    RobotCfg,
    DomainRandCfg,
    RewardCfg,
    HeightScannerCfg,
    AddRigidBodyMassCfg,
    MLPPolicyCfg,
    RNNPolicyCfg,
)

# from legged_lab.assets.unitree import G1_CFG
from legged_lab.assets.srobot_wheel_legged import SROBOT_WHEEL_LEGGED_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import legged_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class SrobotWheelLeggedRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    energy = RewTerm(
        func=mdp.energy,
        weight=-1e-3,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*Knee_Joint.*"]), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Joint.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*")},
        weight=-2.0,
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Joint.*"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Joint.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Wheel_Joint.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Joint.*"), "threshold": 500, "max_reward": 400},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*Wheel_Joint.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*Wheel_Joint.*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Roll_Joint.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*"])},
    )


@configclass
class SrobotWheelLeggedFlatEnvCfg(BaseEnvCfg):
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(enable_height_scan=False, prim_body_name="base_link"),
        robot=SROBOT_WHEEL_LEGGED_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
    )
    robot = RobotCfg(
        terminate_contacts_body_names=[".*base_link.*"],
        feet_body_names=[".*Wheel_Joint.*"],
    )
    domain_rand = DomainRandCfg(
        add_rigid_body_mass=AddRigidBodyMassCfg(
            enable=True, params={"body_names": [".*base_link.*"], "mass_distribution_params": (-5.0, 5.0), "operation": "add"}
        )
    )
    reward = SrobotWheelLeggedRewardCfg()


@configclass
class SrobotWheelLeggedFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "srobot_wheel_legged_flat"
    wandb_project: str = "srobot_wheel_legged_flat"


@configclass
class SrobotWheelLeggedRoughEnvCfg(SrobotWheelLeggedFlatEnvCfg):
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(enable_height_scan=True, prim_body_name="base_link"),
        robot=SROBOT_WHEEL_LEGGED_CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
    )
    robot = RobotCfg(
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        terminate_contacts_body_names=[".*base_link.*"],
        feet_body_names=[".*Wheel_Joint.*"],
    )
    reward = SrobotWheelLeggedRewardCfg(
        track_lin_vel_xy_exp=RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5}),
        track_ang_vel_z_exp=RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5}),
        lin_vel_z_l2=RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25),
    )


@configclass
class SrobotWheelLeggedRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "srobot_wheel_legged_rough"
    wandb_project: str = "srobot_wheel_legged_rough"
    policy = RNNPolicyCfg()
