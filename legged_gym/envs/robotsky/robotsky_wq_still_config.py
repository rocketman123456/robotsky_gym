from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class RobotSkyWQStillCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4000
        num_actor_obs = 57 + 0  # 57
        num_critic_obs = 123 + 0  # 123
        num_actions = 16
        episode_length_s = 10  # 20

        num_privileged_obs = num_critic_obs
        num_single_obs = num_actor_obs
        num_obs_length = 50
        num_action_length = 50
        num_latent_dim = 18
        num_obs_history = num_single_obs * num_obs_length
        num_action_history = num_actions * num_action_length

        enable_env_encoder = True
        enable_adaptation_module = True
        evaluate_teacher = True

        multi_critics = False
        dagger = True

        joint_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        wheel_indices = [3, 7, 11, 15]

    class expert:
        num_envs = 4000
        num_actor_obs = 57 + 2  # 57
        num_critic_obs = 123 + 2  # 123
        num_actions = 16

        num_privileged_obs = num_critic_obs
        num_single_obs = num_actor_obs
        num_obs_length = 50
        num_action_length = 50
        num_latent_dim = 18
        num_obs_history = num_single_obs * num_obs_length
        num_action_history = num_actions * num_action_length

        enable_env_encoder = True
        enable_adaptation_module = True
        evaluate_teacher = True

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/robotsky_wq/urdf/robotsky_wq.urdf"

        name = "robotsky_wq"
        foot_name = "Wheel"
        knee_name = "Knee"

        penalize_contacts_on = ["base_link", "Knee", "Hip"]
        terminate_after_contacts_on = ["base_link", "Knee"]  # "Hip", "Knee"
        self_collisions = 0  # 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False  # fixe the base of the robot

        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        default_dof_drive_mode = 3

        # TODO
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = "plane"
        mesh_type = "trimesh"
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # 0.005  # [m]
        border_size = 15  # 15  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        terrain_noise_magnitude = 0.1
        # rough terrain only:
        measure_heights = False
        # 1mx1.6m rectangle (without center line)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        min_init_terrain_level = 0
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 10.0
        terrain_width = 10.0
        num_rows = 5  # 10 # number of terrain rows (levels)
        num_cols = 10  # 10  # number of terrain cols (types)
        max_init_terrain_level = 2  # starting curriculum state
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.0, 0.4, 0.3, 0.3, 0.0, 0.0, 0.0]
        terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # trimesh only:
        # slopes above this threshold will be corrected to vertical surfaces
        slope_treshold = 0.75

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.6  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1  # 0.4
            lin_vel = 0.05  # 0.2
            quat = 0.03  # 0.2
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.60]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_Roll_Joint": 0.1,
            "RF_Hip_Joint": -0.5,
            "RF_Knee_Joint": 1.0,
            "RF_Wheel_Joint": 0.0,
            "LF_Roll_Joint": -0.1,
            "LF_Hip_Joint": -0.5,
            "LF_Knee_Joint": 1.0,
            "LF_Wheel_Joint": 0.0,
            "RB_Roll_Joint": 0.1,
            "RB_Hip_Joint": 0.5,
            "RB_Knee_Joint": -1.0,
            "RB_Wheel_Joint": 0.0,
            "LB_Roll_Joint": -0.1,
            "LB_Hip_Joint": 0.5,
            "LB_Knee_Joint": -1.0,
            "LB_Wheel_Joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # P: position, V: velocity, T: torques
        control_type = {
            "RF_Roll_Joint": "P",
            "RF_Hip_Joint": "P",
            "RF_Knee_Joint": "P",
            "RF_Wheel_Joint": "V",
            "LF_Roll_Joint": "P",
            "LF_Hip_Joint": "P",
            "LF_Knee_Joint": "P",
            "LF_Wheel_Joint": "V",
            "RB_Roll_Joint": "P",
            "RB_Hip_Joint": "P",
            "RB_Knee_Joint": "P",
            "RB_Wheel_Joint": "V",
            "LB_Roll_Joint": "P",
            "LB_Hip_Joint": "P",
            "LB_Knee_Joint": "P",
            "LB_Wheel_Joint": "V",
        }
        # PD Drive parameters:
        stiffness = {
            "Roll_Joint": 20.0,
            "Hip_Joint": 20.0,
            "Knee_Joint": 40.0,
            "Wheel_Joint": 0.5,  # 1.0,
        }
        damping = {
            "Roll_Joint": 0.5,
            "Hip_Joint": 0.5,
            "Knee_Joint": 0.5,
            "Wheel_Joint": 0.5,
        }

        vel_limits = {
            "Roll_Joint": 12.0,
            "Hip_Joint": 30.0,
            "Knee_Joint": 30.0,
            "Wheel_Joint": 12.0,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_joint_scale = 0.25
        action_wheel_scale = 1.0  # 2.0
        # hip_scale_reduction = 4.0  # 1.0  # 2.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.002  # 1000 Hz

    class domain_rand(LeggedRobotCfg.domain_rand):
        add_random = True
        randomize_friction = add_random
        friction_range = [0.5, 2.0]  # privilege
        rolling_friction_range = [0.0, 0.01]  # privilege
        torsion_friction_range = [0.0, 0.2]  # privilege
        randomize_base_mass = add_random
        added_mass_range = [-2.0, 2.0]  # privilege
        randomize_body_inertia = add_random
        scaled_body_inertia_range = [0.90, 1.1]  # %5 error
        push_robots = add_random
        push_interval_s = 10  # 15
        max_push_vel_xy = 1.0  # 0.5
        max_push_ang_vel = 1.0  # 0.5
        push_robots_force = False
        push_robots_force_interval = 20
        push_robots_force_duration = 5
        max_push_force = 50.0  # privilege
        max_push_torque = 5.5  # privilege
        # push_curriculum_start_step = 10000 * 24
        # push_curriculum_common_step = 30000 * 24
        # additional domain randomization
        randomize_leg_mass = add_random
        added_leg_mass_range = [-0.1, 0.1]  # randomize on each link
        randomize_com_displacement = add_random
        com_displacement_range = [-0.05, 0.05]
        randomize_restitution = add_random
        restitution_range = [0.0, 1.0]
        randomize_motor_strength = add_random
        motor_strength_range = [0.9, 1.1]  # privilege
        randomize_kp_factor = add_random
        kp_factor_range = [0.8, 1.2]  # privilege
        randomize_kd_factor = add_random
        kd_factor_range = [0.7, 1.3]  # privilege
        # -- latent simulation
        enable_latent = True
        queue_latent_obs = 12
        queue_latent_act = 12
        obs_latency = [0, 6]
        act_latency = [0, 6]
        # -- add init joint pos rand
        randomize_init_joint_pos = add_random
        init_joint_pos_range = [-0.05, 0.05]

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        curriculum = False
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 6.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        max_lin_x_curriculum = 3.0
        max_lin_y_curriculum = 0.4
        max_ang_curriculum = 1.5
        min_lin_x_curriculum = 0.1
        min_lin_y_curriculum = 0.1
        min_ang_curriculum = 0.2

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0
        clip_actions = 100.0

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25  # 1.0
            dof_pos = 1.0
            dof_vel = 0.05  # [0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 1.0]
            quat = 1.0
            height_measurements = 5.0
            body_height_cmd = 2.0

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.25  # 0.35  # 0.38
        tracking_sigma = 0.2
        target_feet_height = 0.15
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.99
        max_contact_force = 150.0
        target_joint_pos_scale = 0.2
        only_positive_rewards = False
        regularization_scale_curriculum = True
        clip_rewards = True
        cycle_time = 0.5  # sec
        phase_offset_fr = 0.5
        phase_offset_fl = 0.0
        phase_offset_hr = 0.0
        phase_offset_hl = 0.5
        target_joint_pos_scale = 0.3
        min_feet_air_time = 0.25
        max_feet_air_time = 0.65
        foot_min_dist = 0.32
        foot_max_dist = 0.36
        foot_min_dist_y = 0.10
        foot_max_dist_y = 0.13
        foot_dist_sigma = 0.25

        class curriculum(LeggedRobotCfg.rewards.curriculum):
            regularization_names = [
                # gait
                "feet_stumble",
                "feet_contact_forces",
                # body
                "lin_vel_z",
                "ang_vel_xy",
                "orientation_v1",
                "base_height",
                # collision
                "collision",
                # robot limits
                "dof_pos_limits",
                "dof_torque_limits",
                "dof_acc",
                "dof_vel",
                "torques",
                "action_rate",
                "energy_expenditure",
                # extra
                "hip_default_joint_pos",
                "ref_joint_regularization",
            ]
            regularization_scale = 1.0
            regularization_scale_range = [0.8, 1.2]
            regularization_scale_gamma = 0.0001

        class pbrs(LeggedRobotCfg.rewards.pbrs):
            pbrs_prev_names = [
                "orientation",
                "base_height",
                "default_joint_regularization",
                "ref_joint_regularization",
                "tracking_lin_vel",
                "tracking_ang_vel",
                "ang_vel_xy_regularization",
                "lin_vel_z_regularization",
                "feet_air_time",
                "feet_land_time",
                "dof_acc_reg",
                "dof_vel_reg",
                "torques_reg",
                "torques_wheel_reg",
                "action_rate_reg",
                # "feet_air_time_reg",
            ]

        class pbrs_scale:  # Not in use
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0

            tracking_lin_vel_pbrs = 5.0
            tracking_ang_vel_pbrs = 1.0

            ang_vel_xy_reg_pbrs = 1.0
            lin_vel_z_reg_pbrs = 0.1

            feet_air_time_pbrs = 1.0

            default_joint_reg_pbrs = 1.0  # 0.1  # 1.0
            ref_joint_reg_pbrs = 1.0
            orientation_pbrs = 1.0
            base_height_pbrs = 1.0

            dof_acc_pbrs = 0.1
            dof_vel_pbrs = 0.1
            torques_reg_pbrs = 0.1
            torques_wheel_reg_pbrs = 0.1
            action_rate_pbrs = 0.1

        class barrier_scales(LeggedRobotCfg.rewards.barrier_scales):
            # -- constraints --
            barrier_tracking_linx_vel = 0.2
            barrier_tracking_liny_vel = 0.2
            barrier_tracking_ang_vel = 0.2
            barrier_gait_fl = 0.1
            barrier_gait_fr = 0.1
            barrier_gait_hl = 0.1
            barrier_gait_hr = 0.1
            # barrier_joint_limit = 0.1
            # barrier_ref_joint_regularization = 0.2
            # barrier_base_height = 1.0e-4
            # barrier_orientation = 0.1

        class scales(LeggedRobotCfg.rewards.scales):
            # termination = -500.0  # -200.0  # -100.0
            # alive = 0.1  # 0.15
            survival = 0.1

            # -- vel tracking --
            # tracking_lin_vel_eth = 1.0
            tracking_lin_vel = 1.0  # 1.0  # 2.0
            tracking_ang_vel = 1.0  # 0.5  # 1.0
            # tracking_lin_vel_v4 = -1.0
            # tracking_ang_vel_v4 = -1.0
            # low_speed = 0.5

            # -- gait --
            feet_air_time = 2.0  # 5.0  # 10.0  # 4.0
            # feet_land_time = 0.01  # 0.01  # 0.02  # 0.05
            # feet_air_time_v1 = 2.0
            # feet_air_time_v2 = 2.0  # 5.0
            # feet_air_time_v3 = 5.0

            # feet_contact_number = 0.1
            # barrier_gait_fl = 0.2
            # barrier_gait_fr = 0.2
            # barrier_gait_hl = 0.2
            # barrier_gait_hr = 0.2

            no_fly = 0.1  # 0.2  # 0.05
            # no_fly_v1 = -0.1  # -0.2
            # no_fly_v2 = 0.1  # 0.2  # 0.1
            # no_fly_v3 = 0.1
            # feet_stumble = -0.02  # 0.5
            # feet_clearance_v1 = -0.1  # -0.05  # -0.2
            # foot_slip = -0.1

            # feet_distance_x = 0.2  # 0.2
            # feet_distance_y = 0.2  # 0.2
            # knee_distance = 0.2  # 0.2
            # feet_stumble_v1 = -1.25
            # stand_still = -0.5  # -0.02  # -0.05
            stand_still_v1 = -1.0  # -0.5
            # feet_clearance = -0.001
            # feet_clearance_v2 = -0.01
            # feet_clearance_v2 = -0.001

            # -- contact --
            collision = -10.0  # -5.0  # -2.0
            feet_contact_forces = -0.001  # -0.0001

            # -- base pos --
            orientation = -10.0  # -5.0  # -2.0  # -1.0
            base_height = -5.0  # -2.0
            # orientation_v1 = -1.0  # -0.5
            # base_height_v1 = -5.0
            ang_vel_xy = -0.05  # -0.1  # -0.05
            lin_vel_z = -2.0  # -10.0  # -4.0  # -2.0
            base_acc = 0.5  # 2.0  # 0.5  # 0.2

            # -- joint limit --
            dof_pos_limits = -5.0  # -1.0
            dof_torque_limits = -0.5  # -0.1  # -1.0  # -0.002

            # -- energy --
            dof_acc = -1.0e-8  # -2.0e-8  # -2.0e-7
            dof_vel = -1.0e-4  # -2.0e-3  # -2.0e-4
            torques = -5.0e-5  # -2.0e-5  # -1.0e-5
            torques_wheel = -2.0e-1  # -2.0e-2  # -2.0e-3
            action_rate = -2.0e-4  # -5.0e-4  # -2.0e-4  # -5.0e-6
            action_acc = -1.0e-3  # -1.0e-5  # -1.0e-4
            energy_expenditure = -2.0e-4  # -1.0e-4

            # -- extra --
            # default_joint_pos = -4.0  # -2.0  # -1.0  # 0.5
            # hip_default_joint_pos = -2.0  # -10.0  # -5.0
            # default_joint_pos_v2 = -1.0  # -0.5  # -1.0  # -5.0
            # ref_joint_pos = -1.0
            # default_joint_regularization = 1.0

            # -- PBRS rewards --
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0

            # tracking_lin_vel_pbrs = 5.0
            # tracking_ang_vel_pbrs = 1.0

            # ang_vel_xy_reg_pbrs = 1.0
            # lin_vel_z_reg_pbrs = 0.1

            # feet_air_time_pbrs = 1.0

            # default_joint_reg_pbrs = 1.0  # 0.1  # 1.0
            # ref_joint_reg_pbrs = 1.0
            # orientation_pbrs = 1.0
            # base_height_pbrs = 1.0

            # dof_acc_pbrs = 0.1
            # dof_vel_pbrs = 0.1
            # torques_reg_pbrs = 0.1
            # torques_wheel_reg_pbrs = 0.1
            # action_rate_pbrs = 0.1


class RobotSkyWQCfgPPO(LeggedRobotCfgPPO):
    seed = 1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        # only for 'ActorCriticRecurrent':
        rnn_type = "lstm"
        rnn_hidden_size = 512
        rnn_num_layers = 1

        # env encoder
        adaptation_hidden_dims = [512, 256, 128]
        env_encoder_hidden_dims = [512, 256, 128]
        env_encoder_latent_dims = [18]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        entropy_coef = 0.001  # 0.01  # 0.005
        schedule = "adaptive"  # could be adaptive, fixed
        multi_critics = RobotSkyWQCfg.env.multi_critics
        enable_dagger = RobotSkyWQCfg.env.dagger
        evaluate_expert_teacher = RobotSkyWQCfg.expert.evaluate_teacher
        grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"  # ActorCritic, ActorCriticRecurrent
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # 48  # 24  # per iteration
        max_iterations = 10000  # 2001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "robotsky_wq"
        run_name = ""
