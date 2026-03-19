import math
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from robotsky_lab.envs.base.base_env_config import (  # noqa:F401
    BaseEnvCfg,
    BaseAgentCfg,
    BaseSceneCfg,
    RobotCfg,
    DomainRandCfg,
    RewardCfg,
    HeightScannerCfg,
    PushRobotCfg,
    ActionDelayCfg,
    AddRigidBodyMassCfg,
    CommandsCfg,
    NormalizationCfg,
    ObsScalesCfg,
    CommandRangesCfg,
    MLPPolicyCfg,
    RNNPolicyCfg,
)

# from robotsky_lab.assets.unitree import G1_CFG
from robotsky_lab.assets.srobot_biped import SROBOT_BIPED_CFG
from robotsky_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import robotsky_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class SrobotBipedRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
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
        weight=-1e-5,  # -1e-3
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,  # -2.5e-7
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,  # -1.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*base_link.*", ".*HipPitch.*", ".*Knee.*"]),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Ankle.*"),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-5.0,  # -2.0
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*")},
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
        weight=5.0,  # 0.15
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Ankle.*"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Ankle.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Ankle.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-1e-3,  # -3e-3
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Ankle.*"), "threshold": 500, "max_reward": 400},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*Ankle.*"]),
            "threshold": 0.2,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*Ankle.*"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,  # -2.0
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,  # -0.15
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipYaw_Joint.*", ".*HipRoll_Joint.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,  # -0.02
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HipPitch_Joint.*", ".*Knee_Joint.*", ".*Ankle_Joint.*"])},
    )


@configclass
class SrobotBipedFlatEnvCfg(BaseEnvCfg):
    terrain_cfg = TerrainGeneratorCfg(
        curriculum=False,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={"random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.2, noise_range=(-0.01, 0.01), noise_step=0.02, border_width=0.25)},
    )
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(enable_height_scan=False, prim_body_name="base_link"),
        robot=SROBOT_BIPED_CFG,
        terrain_type="generator",
        terrain_generator=terrain_cfg,
    )
    robot = RobotCfg(
        terminate_contacts_body_names=[".*base_link.*"],
        feet_body_names=[".*Ankle.*"],
    )
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=2.0,
            ang_vel=0.25,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=0.05,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    domain_rand = DomainRandCfg(
        add_rigid_body_mass=AddRigidBodyMassCfg(
            enable=True,
            params={
                "body_names": [".*base_link.*"],
                "mass_distribution_params": (-0.5, 0.5),
                "operation": "add",
            },
        ),
        push_robot=PushRobotCfg(
            enable=True,
            push_interval_s=15.0,
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        ),
        action_delay=ActionDelayCfg(
            enable=True,
            params={"max_delay": 5, "min_delay": 0},
        ),
    )
    reward = SrobotBipedRewardCfg()


@configclass
class SrobotBipedFlatAgentCfg(BaseAgentCfg):
    max_iterations = 20000
    experiment_name: str = "srobot_biped_flat"
    wandb_project: str = "srobot_biped_flat"
    policy = RNNPolicyCfg(init_noise_std=1.5)
    # policy = MLPPolicyCfg(init_noise_std=1.5)


@configclass
class SrobotBipedRoughEnvCfg(SrobotBipedFlatEnvCfg):
    scene = BaseSceneCfg(
        height_scanner=HeightScannerCfg(enable_height_scan=True, prim_body_name="base_link"),
        robot=SROBOT_BIPED_CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
    )
    robot = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        terminate_contacts_body_names=[".*base_link.*"],
        feet_body_names=[".*Ankle.*"],
    )
    reward = SrobotBipedRewardCfg(
        track_lin_vel_xy_exp=RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5}),
        track_ang_vel_z_exp=RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.5}),
        lin_vel_z_l2=RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5),  # -0.25
    )


@configclass
class SrobotBipedRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "srobot_biped_rough"
    wandb_project: str = "srobot_biped_rough"
    policy = RNNPolicyCfg(init_noise_std=1.5)
