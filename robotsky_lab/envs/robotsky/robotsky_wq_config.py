# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from dataclasses import MISSING

import robotsky_lab.mdp as mdp
from robotsky_lab.assets.robotsky_wq.robotsky_wq import ROBOTSKY_WQ_CFG
from robotsky_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    HeightScannerCfg,
    NormalizationCfg,
    ObsScalesCfg,
    RewardCfg,
    RobotCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    BaseSceneCfg,
    CommandsCfg,
    NoiseCfg,
    CommandRangesCfg,
    NoiseScalesCfg,
    DomainRandCfg,
    EventCfg,
    EventTerm,
    ActionDelayCfg,
    SimCfg,
    PhysxCfg,
)
from robotsky_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG

import math


@configclass
class DiagonalTrotCfg:
    """开环对角 trot：在策略输出的腿关节 action 上叠加正弦，辅助抬脚。

    对角组：RF 与 LB 同相，LF 与 RB 反相（相差 π），典型对角小跑相位。
    步态周期随速度指令增大而缩短（更快指令 → 更高步频）。
    适用于前/侧移、转弯、侧移+转、原地转等任何非静止指令（由 gate 关闭静止段）。
    """

    enable: bool = False
    # 合成驱动量低于此值时不叠加（避免指令噪声引起微颤）
    cmd_deadband: float = 0.05
    # 超过死区后幅值从 0→1 的过渡宽度，越大开启越平滑
    cmd_blend: float = 0.10
    # 线速度归一化尺度（m/s）：用于把 ||v_xy|| 映射到周期混合项
    speed_ref: float = 2.0
    # 角速度归一化尺度（rad/s）：|wz| 参与周期与步频
    ang_ref: float = 2.0
    # 门控里 |wz| 的权重：偏大则原地自转更易触发抬脚辅助
    wz_cmd_weight: float = 0.2
    # 指令很大时逼近的最短步态周期（s）
    period_min_s: float = 0.30
    # 指令很小时逼近的最长步态周期（s）
    period_max_s: float = 0.60
    # 以下幅度为「归一化 action」空间，再经 clip_actions 约束；轮关节不参与
    roll_amp: float = 0.0
    hip_amp: float = 0.15
    knee_amp: float = 0.30


@configclass
class WheelLeggedRobotCfg(RobotCfg):
    """Extends RobotCfg with wheel-specific fields for mixed position/velocity control."""

    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = [".*base_link.*"]
    feet_body_names: list = [".*Wheel_Link.*"]

    wheel_joint_names: list = [".*Wheel_Joint.*"]
    # Velocity command scale for wheel joints (rad/s per unit action)
    wheel_action_scale: float = 4.0

    # 对角 trot 正弦叠加（见 DiagonalTrotCfg）
    diagonal_trot: DiagonalTrotCfg = DiagonalTrotCfg()


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
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.1,  # -0.02 # -0.02 # -0.01
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*"),
            "sensor_cfg": None,  # SceneEntityCfg("height_scanner_base"),
            "target_height": 0.3,
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,  # -1.0
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
    # upward = RewTerm(func=mdp.upward, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-1.5e-4,  # -3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Link.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Link.*"),
        },
    )
    # feet_height = RewTerm(
    #     func=mdp.feet_height,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*Wheel_Link.*"),
    #         "tanh_mult": 2.0,
    #         "target_height": 0.05,
    #         "command_name": "base_velocity",
    #     },
    # )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.3, # 0.15
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*Wheel_Link.*"), "threshold": 0.4},
    # )
    # Penalise leg joints straying from default (energy efficiency)
    joint_torque_l2 = RewTerm(
        func=mdp.joint_torque_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,  # -0.05 # -0.02 # -0.5 # -0.2 # -0.02
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    # joint_velocity_l2 = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1.0e-3,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    # )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,  # -2.0
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_Joint.*", ".*Knee_Joint.*", ".*Roll_Joint.*"])},
    )
    stand_still_wheel = RewTerm(
        func=mdp.stand_still_vel,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Wheel_Joint.*"])},
    )
    # wheel_torque_l2 = RewTerm(
    #     func=mdp.joint_torque_l2,
    #     weight=-2.5e-7,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Wheel_Joint.*"])},
    # )
    wheel_velocity_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,  # 1.0e-6 # 1.0e-5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Wheel_Joint.*"])},
    )
    wheel_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-9,  # -1.0e-7 # -1.0e-6
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Wheel_Joint.*"])},
    )
    # wheel_power = RewTerm(
    #     func=mdp.joint_power,
    #     weight=-2.5e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Wheel_Joint.*"])},
    # )


@configclass
class RobotSkyWQFlatEnvCfg(BaseEnvCfg):
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=ROBOTSKY_WQ_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="base_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )

    reward: RobotSkyWQRewardCfg = RobotSkyWQRewardCfg()
    robot: WheelLeggedRobotCfg = WheelLeggedRobotCfg()

    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=2.0,
            ang_vel=0.5,  # 1.0
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=0.05,  # 1.0
            actions=1.0,
            height_scan=1.0,
            wheel_vel=0.1,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(lin_vel_x=(-2.0, 2.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-2.0, 2.0), heading=(-math.pi, math.pi)),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=0.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.4, 1.6),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*"),
                    "mass_distribution_params": (-2.0, 2.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(6.0, 8.0),
                params={"velocity_range": {"x": (-1.2, 1.2), "y": (-1.2, 1.2), "yaw": (-1.0, 1.0)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=True, params={"max_delay": 2, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(
        dt=0.005,
        decimation=4,
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )

    def __post_init__(self):
        super().__post_init__()


@configclass
class RobotSkyWQFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "robotsky_wq_flat"
    wandb_project: str = "robotsky_wq_flat"

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = 24
        self.max_iterations = 40001
        self.empirical_normalization = False
        self.save_interval = 1000
        self.logger = "tensorboard"


@configclass
class RobotSkyWQRoughEnvCfg(RobotSkyWQFlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = False
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10
        # self.reward.track_lin_vel_xy_exp.weight = 1.5
        # self.reward.track_ang_vel_z_exp.weight = 1.5
        # self.reward.lin_vel_z_l2.weight = -0.25


@configclass
class RobotSkyWQRoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "robotsky_wq_rough"
    wandb_project: str = "robotsky_wq_rough"

    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = 24
        self.max_iterations = 60001
        self.empirical_normalization = False

        # self.policy.class_name = "ActorCriticRecurrent"
        # self.policy.actor_hidden_dims = [256, 256, 128]
        # self.policy.critic_hidden_dims = [256, 256, 128]
        # self.policy.rnn_hidden_size = 256
        # self.policy.rnn_num_layers = 1
        # self.policy.rnn_type = "lstm"
