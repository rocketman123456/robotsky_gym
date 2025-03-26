# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_actor_obs = 235
        num_critic_obs = 235
        num_actions = 12

        env_spacing = 1.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

        # for env encoder
        num_privileged_obs = 235
        num_single_obs = 39
        num_obs_length = 10
        num_action_length = 10
        num_obs_history = num_single_obs * num_obs_length
        num_action_history = num_actions * num_action_length
        num_latent_dim = 18

        enable_env_encoder = False
        enable_adaptation_module = False
        evaluate_teacher = False

        multi_critics = False
        dagger = False

        debug_viz = False

    class terrain:
        mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # rough terrain only:
        measure_heights = False
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        max_lin_x_curriculum = 1.0
        max_lin_y_curriculum = 1.0
        max_ang_curriculum = 1.0
        min_lin_x_curriculum = 0.2
        min_lin_y_curriculum = 0.2
        min_ang_curriculum = 0.2

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_a": 0.0,
            "joint_b": 0.0,
        }

    class control:
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {"joint_a": 10.0, "joint_b": 15.0}  # [N*m/rad]
        damping = {"joint_a": 1.0, "joint_b": 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        knee_name = "None"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        add_random = False
        randomize_friction = False
        friction_range = [0.5, 1.25]
        rolling_friction_range = [0.0, 0.01]
        torsion_friction_range = [0.0, 0.02]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        randomize_body_inertia = False
        scaled_body_inertia_range = [0.90, 1.1]  # %5 error
        push_robots = False
        push_interval_s = 1  # 15
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2
        push_robots_force = False
        push_robots_force_interval = 20
        push_robots_force_duration = 5
        max_push_force = 20.0
        max_push_torque = 0.5
        # additional domain randomization
        randomize_leg_mass = False
        added_leg_mass_range = [-0.1, 0.1]  # randomize on each link
        randomize_com_displacement = False
        com_displacement_range = [-0.05, 0.05]
        randomize_restitution = False
        restitution_range = [0.0, 0.1]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_kp_factor = False
        kp_factor_range = [0.8, 1.2]
        randomize_kd_factor = False
        kd_factor_range = [0.5, 1.5]
        # latent simulation
        enable_latent = False
        queue_latent_obs = 8
        queue_latent_act = 8
        obs_latency = [5, 20]  # ms
        act_latency = [5, 20]  # ms
        # add init joint pos rand
        randomize_init_joint_pos = False
        init_joint_pos_range = [-0.1, 0.1]

    class rewards:
        regularization_scale_curriculum = True
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_rewards = False
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.0  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 1.0
        max_contact_force = 100.0  # forces above this value are penalized

        class curriculum:
            regularization_names = []
            regularization_scale = 1.0
            regularization_scale_range = [1.0, 1.0]
            regularization_scale_gamma = 0.0001

        class pbrs:
            pbrs_prev_names = []

        class barrier_scales:
            barrier_tracking_linx_vel = 0.0
            barrier_tracking_liny_vel = 0.0
            barrier_tracking_ang_vel = 0.0
            barrier_gait_fl = 0.0
            barrier_gait_fr = 0.0
            barrier_gait_hl = 0.0
            barrier_gait_hr = 0.0
            barrier_joint_limit = 0.0

        class scales:
            termination = -0.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0000
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = 0.0
            stand_still = 0.0

    class normalization:
        clip_observations = 100.0
        clip_actions = 100.0

        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise:
        add_noise = False
        noise_level = 2.0  # scales other values

        class noise_scales:
            cmd = 0.03
            dof_pos = 0.02
            dof_vel = 0.15
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11.0, 5, 3.0]  # [m]

    class sim:
        dt = 0.002
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

        env_encoder_hidden_dims = [512, 256, 128]
        adaptation_hidden_dims = [512, 256, 128]

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 32  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        # -- extra --
        multi_critics = False  # multi critics
        enable_dagger = False
        grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]  # LCP

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 3000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "test"
        run_name = ""
        # load and resume
        resume = False
        resume_dagger = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
