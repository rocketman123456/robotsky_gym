from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from legged_gym.envs import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.robotsky.robotsky_wq_config import RobotSkyWQCfg, RobotSkyWQCfgPPO
from legged_gym.utils.terrain import WheeledQuadTerrain
from legged_gym.utils.math import *


class RobotSkyWQ(LeggedRobot):
    cfg: RobotSkyWQCfg

    def __init__(self, cfg: RobotSkyWQCfg, sim_params, physics_engine, sim_device, graphics_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, graphics_device, headless)
        self.last_feet_z = 0.05  #
        self.feet_height = torch.zeros((self.num_envs, 4), device=self.device)
        self.rwd_prev = {}
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                name = self.dof_names[i]
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                # self.dof_vel_limits[i] = props["velocity"][i].item()
                for dof_name in self.cfg.control.stiffness.keys():
                    if dof_name in name:
                        self.dof_vel_limits[i] = self.cfg.control.vel_limits[dof_name]
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        # to match sim2sim
        noise_vec[0:5] = 0.0  # commands
        noise_vec[5:21] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[21:37] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[37:53] = 0.0  # previous actions
        noise_vec[53:56] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[56:59] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        return noise_vec

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase_fl = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fl
        fl = torch.sin(2 * np.pi * phase_fl)
        phase_fr = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fr
        fr = torch.sin(2 * np.pi * phase_fr)
        phase_hl = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hl
        hl = torch.sin(2 * np.pi * phase_hl)
        phase_hr = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hr
        hr = torch.sin(2 * np.pi * phase_hr)

        return fl, fr, hl, hr

    def _get_gait_stance_mask(self):
        # return float mask 1 is stance, 0 is swing
        # phase = self._get_phase()
        # sin_pos = torch.sin(2 * torch.pi * phase)

        # cycle_time = self.cfg.rewards.cycle_time
        # phase_fl = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fl
        # fl = torch.sin(2 * np.pi * phase_fl)
        # phase_fr = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fr
        # fr = torch.sin(2 * np.pi * phase_fr)
        # phase_hl = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hl
        # hl = torch.sin(2 * np.pi * phase_hl)
        # phase_hr = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hr
        # hr = torch.sin(2 * np.pi * phase_hr)

        fl, fr, hl, hr = self._get_gait_phase()

        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 4), device=self.device)
        # left foot stance
        stance_mask[:, 1] = fl < 0
        stance_mask[:, 3] = hl < 0
        # right foot stance
        stance_mask[:, 0] = fr < 0
        stance_mask[:, 2] = hr < 0
        # Double support phase
        # stance_mask[torch.abs(sin_pos) < 0.05] = 1

        return stance_mask

    def compute_ref_state(self):
        # calculate reference motion
        # phase = self._get_phase()
        # sin_pos = torch.sin(2 * torch.pi * phase)
        # sin_pos_l = sin_pos.clone()
        # sin_pos_r = sin_pos.clone()
        # reset dof pos
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)

        # cycle_time = self.cfg.rewards.cycle_time
        # phase_fl = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fl
        # fl = torch.sin(2 * np.pi * phase_fl)
        # phase_fr = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fr
        # fr = torch.sin(2 * np.pi * phase_fr)
        # phase_hl = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hl
        # hl = torch.sin(2 * np.pi * phase_hl)
        # phase_hr = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hr
        # hr = torch.sin(2 * np.pi * phase_hr)

        fl, fr, hl, hr = self._get_gait_phase()

        # set default joint pos
        self.ref_dof_pos[:] = self.default_dof_pos
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2.0 * scale_1
        # right foot stance phase set to default joint pos
        # sin_pos_l[sin_pos_l > 0] = 0.0
        self.ref_dof_pos[:, 1] += fr * scale_1 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (fr >= 0.0)
        self.ref_dof_pos[:, 2] += -fr * scale_2 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (fr >= 0.0)
        self.ref_dof_pos[:, 9] += -hr * scale_1 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (hr >= 0.0)
        self.ref_dof_pos[:, 10] += hr * scale_1 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (hr >= 0.0)
        # left foot stance phase set to default joint pos
        # sin_pos_r[sin_pos_r < 0] = 0.0
        self.ref_dof_pos[:, 5] += fl * scale_1 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (fl >= 0.0)
        self.ref_dof_pos[:, 6] += -fl * scale_2 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (fl >= 0.0)
        self.ref_dof_pos[:, 13] += -hl * scale_1 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (hl >= 0.0)
        self.ref_dof_pos[:, 14] += hl * scale_1 * (torch.norm(self.commands[:, :3], dim=1) < 0.05) * (hl >= 0.0)
        # Double support phase
        # self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        # self.ref_dof_pos = self.ref_dof_pos * (torch.norm(self.commands[:, :3], dim=1) < 0.05)

    def pre_physics_step(self):
        for name in self.cfg.rewards.pbrs.pbrs_prev_names:
            func = getattr(self, "_reward_" + name)
            self.rwd_prev[name] = func()

    def compute_observations(self):
        """Computes observations"""

        # special handle for wheels
        # self.dof_pos[:, self.cfg.env.wheel_indices] = self.dof_vel[:, self.cfg.env.wheel_indices]
        # self.dof_vel[:, self.cfg.env.wheel_indices] = 0.0
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        # sin_pos = self.smooth_sqr_wave(2 * torch.pi * phase).unsqueeze(1)
        # cos_pos = self.smooth_sqr_wave(2 * torch.pi * phase).unsqueeze(1)

        # sin_pos = sin_pos * (torch.norm(self.commands[:, :3]) < 0.05)
        # cos_pos = cos_pos * (torch.norm(self.commands[:, :3]) < 0.05)

        self.compute_ref_state()

        dof_pos = self.dof_pos.clone()
        dof_vel = self.dof_vel.clone()
        dof_pos[:, self.cfg.env.wheel_indices] = dof_vel[:, self.cfg.env.wheel_indices]
        dof_vel[:, self.cfg.env.wheel_indices] = 0.0

        self.actor_obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel,  # 3, need state estimate
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                dof_vel * self.obs_scales.dof_vel,
                self.actions,
                # self.ref_dof_pos,
                # sin_pos,
                # cos_pos,
            ),
            dim=-1,
        )
        self.critic_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                dof_vel * self.obs_scales.dof_vel,
                self.actions,
                # self.ref_dof_pos,
                # sin_pos,
                # cos_pos,
                # additional
                self.rand_push_lin_vel[:, :2],  # 2
                self.rand_push_ang_vel,  # 3
                self.rand_push_force[:, 0, :],  # 3
                self.rand_push_torque[:, 0, :],  # 3
                self.env_frictions,  # 1
                self.env_rolling_frictions,  # 1
                self.env_torsion_frictions,  # 1
                self.motor_strengths,  # 10
                self.kd_factors,  # 10
                self.kp_factors,  # 10
                self.body_mass / 30.0,  # 1
            ),
            dim=-1,
        )
        if self.enable_env_encoder:
            self.privileged_obs_buf = self.critic_obs_buf.clone()
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
            self.actor_obs_buf = torch.cat((self.actor_obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            noise_vec = torch.randn_like(self.actor_obs_buf) * self.noise_scale_vec
            self.actor_obs_buf += noise_vec

        if self.cfg.domain_rand.enable_latent:
            self._process_latent_obs()

        # build history
        self.actor_obs_history.append(self.actor_obs_buf)
        self.critic_obs_history.append(self.critic_obs_buf)
        self.action_history.append(self.actions)
        self.obs_history_buf = torch.cat([self.actor_obs_history[i] for i in range(self.cfg.env.num_obs_length)], dim=1)
        self.action_history_buf = torch.cat([self.action_history[i] for i in range(self.cfg.env.num_action_length)], dim=1)
        # print(self.actor_obs_history)
        # print(self.action_history)
        if self.enable_env_encoder:
            self.privileged_obs_history.append(self.privileged_obs_buf)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # actions = torch.zeros_like(actions, device=self.device)
        # actions[:, 3] = 1.0
        # actions[:, 7] = 1.0
        # actions[:, 11] = 1.0
        # actions[:, 15] = 1.0
        # pd controller
        actions_scaled = actions.clone()
        actions_scaled[:, self.cfg.env.joint_indices] = actions[:, self.cfg.env.joint_indices] * self.cfg.control.action_joint_scale
        actions_scaled[:, self.cfg.env.wheel_indices] = actions[:, self.cfg.env.wheel_indices] * self.cfg.control.action_wheel_scale

        p_gains = self.p_gains * self.kp_factors
        d_gains = self.d_gains * self.kd_factors
        torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # add seperate control type for each joint
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            control_type = self.cfg.control.control_type[name]
            if control_type == "P":
                torques[:, i] = p_gains[:, i] * (actions_scaled[:, i] + self.default_dof_pos[:, i] - self.dof_pos[:, i]) - d_gains[:, i] * self.dof_vel[:, i]
            elif control_type == "V":
                torques[:, i] = p_gains[:, i] * (actions_scaled[:, i] - self.dof_vel[:, i])
                # - d_gains[:, i] * (self.dof_vel[:, i] - self.last_dof_vel[:, i]) / self.sim_params.dt
            elif control_type == "T":
                torques[:, i] = actions_scaled[:, i]
            else:
                raise NameError(f"Unknown controller type: {control_type}")
        # raise NameError(f"Unknown controller type: {control_type}")
        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # 由于方向的问题，所以这里需要curriculum的是y轴的速度
        # if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.9 * self.reward_scales["tracking_lin_vel"]:
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            max_lin_x = self.cfg.commands.max_lin_x_curriculum
            max_lin_y = self.cfg.commands.max_lin_y_curriculum
            max_ang = self.cfg.commands.max_ang_curriculum
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] * 1.2, -max_lin_x, 0.0)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] * 1.2, 0.0, max_lin_x)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] * 1.2, -max_lin_y, 0.0)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] * 1.2, 0.0, max_lin_y)
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] * 1.1, -max_ang, 0.0)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] * 1.1, 0.0, max_ang)

        # if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length < 0.1 * self.reward_scales["tracking_lin_vel"]:
        #     max_lin_x = self.cfg.commands.max_lin_x_curriculum
        #     max_lin_y = self.cfg.commands.max_lin_y_curriculum
        #     self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] * 0.8, -max_lin_x, -self.cfg.commands.min_lin_x_curriculum)
        #     self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] * 0.8, self.cfg.commands.min_lin_x_curriculum, max_lin_x)
        #     self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] * 0.8, -max_lin_y, -self.cfg.commands.min_lin_x_curriculum)
        #     self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] * 0.8, self.cfg.commands.min_lin_x_curriculum, max_lin_y)

    # ================================================ Rewards ================================================== #

    def _reward_alive(self):
        return 1.0

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_survival(self):
        # Survival reward / penalty
        return ~(self.reset_buf * ~self.time_out_buf)

    # -- gait -- #

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance_x(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist_r = torch.norm(foot_pos[:, 0, :2] - foot_pos[:, 2, :2], dim=1)
        foot_dist_l = torch.norm(foot_pos[:, 1, :2] - foot_pos[:, 3, :2], dim=1)
        min_fd = self.cfg.rewards.foot_min_dist
        max_df = self.cfg.rewards.foot_max_dist
        d_min_r = torch.clamp(foot_dist_r - min_fd, -0.5, 0.0)
        d_max_r = torch.clamp(foot_dist_r - max_df, 0, 0.5)
        d_min_l = torch.clamp(foot_dist_l - min_fd, -0.5, 0.0)
        d_max_l = torch.clamp(foot_dist_l - max_df, 0, 0.5)
        rew_r = (torch.exp(-torch.abs(d_min_r) * self.cfg.rewards.foot_dist_sigma) + torch.exp(-torch.abs(d_max_r) * self.cfg.rewards.foot_dist_sigma)) / 2.0
        rew_l = (torch.exp(-torch.abs(d_min_l) * self.cfg.rewards.foot_dist_sigma) + torch.exp(-torch.abs(d_max_l) * self.cfg.rewards.foot_dist_sigma)) / 2.0
        return rew_r + rew_l

    def _reward_feet_distance_y(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist_f = torch.norm(foot_pos[:, 0, :2] - foot_pos[:, 1, :2], dim=1)
        foot_dist_h = torch.norm(foot_pos[:, 2, :2] - foot_pos[:, 3, :2], dim=1)
        min_fd = self.cfg.rewards.foot_min_dist_y
        max_df = self.cfg.rewards.foot_max_dist_y
        d_min_r = torch.clamp(foot_dist_f - min_fd, -0.5, 0.0)
        d_max_r = torch.clamp(foot_dist_f - max_df, 0, 0.5)
        d_min_l = torch.clamp(foot_dist_h - min_fd, -0.5, 0.0)
        d_max_l = torch.clamp(foot_dist_h - max_df, 0, 0.5)
        rew_r = (torch.exp(-torch.abs(d_min_r) * self.cfg.rewards.foot_dist_sigma) + torch.exp(-torch.abs(d_max_r) * self.cfg.rewards.foot_dist_sigma)) / 2.0
        rew_l = (torch.exp(-torch.abs(d_min_l) * self.cfg.rewards.foot_dist_sigma) + torch.exp(-torch.abs(d_max_l) * self.cfg.rewards.foot_dist_sigma)) / 2.0
        return rew_r + rew_l

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        min_fd = self.cfg.rewards.knee_min_dist
        max_df = self.cfg.rewards.knee_max_dist / 2
        d_min = torch.clamp(foot_dist - min_fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        rew = (torch.exp(-torch.abs(d_min) * self.cfg.rewards.knee_dist_sigma) + torch.exp(-torch.abs(d_max) * self.cfg.rewards.knee_dist_sigma)) / 2.0
        return rew

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = self.get_first_contact()
        rew_air_time = torch.sum((self.feet_air_time) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_air_time *= torch.norm(self.commands[:, :3], dim=1) > 0.1  # no reward for zero command
        # self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_feet_land_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_air = self.get_first_air()
        rew_land_time = torch.sum((self.feet_land_time) * first_air, dim=1)  # reward only on first contact with the ground
        rew_land_time *= torch.norm(self.commands[:, :3], dim=1) > 0.1  # no reward for zero command
        # self.feet_land_time *= ~no_contact_filt
        return rew_land_time

    def _reward_feet_air_time_v1(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airtime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airtime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airtime

    def _reward_feet_air_time_v2(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_stance_mask()
        contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        # self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_air_time_v3(self):
        stance_mask = self._get_gait_stance_mask()
        stance_mask = stance_mask > 0
        first_contact = self.get_first_contact()
        # rew_air_time = torch.where(stance_mask, self.feet_air_time, -self.feet_air_time)
        rew_air_time = torch.sum((self.feet_air_time) * first_contact * stance_mask, dim=1)  # reward only on first contact with the ground
        rew_air_time *= torch.norm(self.commands[:, :3], dim=1) > 0.1  # no reward for zero command
        # self.feet_air_time *= ~contact_filt
        return rew_air_time

    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_stance_mask()
        reward = torch.where(contact == stance_mask, 1.0, -1.0)
        return torch.sum(reward, dim=-1)  # torch.mean(reward, dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        contact_xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.0
        contact_z = self.contact_forces[:, self.feet_indices, 2] > 5.0
        return torch.any(contact_xy * contact_z, dim=1)

    def _reward_feet_stumble_v1(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        # penalize high contact forces
        foot_contact_force = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        return torch.sum((foot_contact_force - self.cfg.rewards.max_contact_force).clip(min=0.0, max=800.0), dim=1)

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.1
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        feet_z[:, 0] += base_height
        feet_z[:, 1] += base_height
        feet_z[:, 2] += base_height
        feet_z[:, 3] += base_height

        # feet height should be closed to target feet height at the peak
        pos_error = torch.abs(torch.clamp(feet_z - self.cfg.rewards.target_feet_height, min=-1.0, max=1.0))
        rew_pos = torch.sum(pos_error * contact)
        return rew_pos

    def _reward_feet_clearance_v1(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # feet_z[:, 0] += base_height
        # feet_z[:, 1] += base_height
        # feet_z[:, 2] += base_height
        # feet_z[:, 3] += base_height

        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        swing_mask = 1 - self._get_gait_stance_mask()
        fl, fr, hl, hr = self._get_gait_phase()

        delta_fr = self.feet_height[:, 0] - self.cfg.rewards.target_feet_height * fr * (fr > 0)
        delta_fl = self.feet_height[:, 1] - self.cfg.rewards.target_feet_height * fl * (fl > 0)
        delta_hr = self.feet_height[:, 2] - self.cfg.rewards.target_feet_height * hr * (hr > 0)
        delta_hl = self.feet_height[:, 3] - self.cfg.rewards.target_feet_height * hl * (hl > 0)

        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height)  #  < 0.01
        rew_pos = rew_pos * swing_mask

        delta_fr = rew_pos[:, 0] - self.cfg.rewards.target_feet_height * fr * (fr > 0)
        delta_fl = rew_pos[:, 1] - self.cfg.rewards.target_feet_height * fl * (fl > 0)
        delta_hr = rew_pos[:, 2] - self.cfg.rewards.target_feet_height * hr * (hr > 0)
        delta_hl = rew_pos[:, 3] - self.cfg.rewards.target_feet_height * hl * (hl > 0)

        rew_pos = torch.abs(delta_fr) + torch.abs(delta_fl) + torch.abs(delta_hr) + torch.abs(delta_hl)
        # rew_pos = torch.sum(rew_pos * swing_mask, dim=1)

        self.feet_height *= ~contact
        return rew_pos

    def _reward_feet_clearance_v2(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_stance_mask()

        fl, fr, hl, hr = self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_foot_slip(self):
        """
        penalize foot slip, including x,y linear velocity and yaw angular velocity, when contacting ground
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    # -- base --

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_v1(self):
        # Penalize non flat base orientation
        target_projected_gravity = torch.zeros_like(self.projected_gravity)
        target_projected_gravity[:, 2] = -1.0
        return torch.sum(torch.square(target_projected_gravity - self.projected_gravity), dim=1)

    def _reward_upward(self):
        # Penalize non flat base orientation
        return torch.square(1.0 - self.projected_gravity[:, 2])

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        height_diff = base_height - self.cfg.rewards.base_height_target
        return torch.square(height_diff)

    def _reward_base_height_v1(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        height_diff = base_height - self.cfg.rewards.base_height_target
        height_diff_clamped = torch.clamp(height_diff, max=0.0, min=-4.0)
        return torch.square(height_diff_clamped)

    def _reward_base_pos(self):
        """
        Computes the reward based on the base's position. Penalizes position-xy error of the robot's base,
        encouraging smoother motion.
        From Desiney
        """
        xy_pos = torch.zeros_like(self.root_states[:, 0:1])
        root_pos = xy_pos - self.root_states[:, 0:1]
        pos_mismatch = torch.exp(-torch.sum(torch.abs(root_pos), dim=1) * self.cfg.rewards.pos_mismatch_sigma)
        # rew = torch.exp(-torch.norm(root_pos, dim=1) * 3)
        return pos_mismatch

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3.0)
        return rew

    # -- task --

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)  # * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_body(self):
        # Tracking of linear velocity commands (xy axes)
        vel_yaw = quat_rotate_inverse(yaw_quat(self.base_quat), self.base_lin_vel[:, :3])
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - vel_yaw[:, :2]), dim=1)  # * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])  # * (torch.norm(self.commands[:, 2]) > 0.1)
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_v4(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)  # * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        return lin_vel_error

    def _reward_tracking_ang_vel_v4(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])  # * (torch.norm(self.commands[:, 2]) > 0.1)
        return ang_vel_error

    def _reward_tracking_lin_vel_v1(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_sum_x = torch.dot(self.commands[:, 0], self.base_lin_vel[:, 0])
        lin_vel_sum_y = torch.dot(self.commands[:, 1], self.base_lin_vel[:, 1])
        lin_vel = torch.sum(torch.square(self.commands[:, :2]), dim=1)
        return (lin_vel_sum_x + lin_vel_sum_y) / (lin_vel + 0.1)

    def _reward_tracking_lin_vel_v2(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_err_1 = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.05)
        lin_vel_err_2 = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.05)
        return torch.exp(-(lin_vel_err_1 + lin_vel_err_2) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_v3(self):
        norm = torch.norm(self.commands[:, :2], dim=-1, keepdim=True)
        target_vec_norm = self.commands[:, :2] / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        return rew

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    # -- energy --

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.dof_pos[:, self.cfg.env.joint_indices] - self.default_dof_pos[:, self.cfg.env.joint_indices]), dim=1)
        return dof_error * (torch.norm(self.commands[:, 2]) < 0.1)

    def _reward_stand_still_v1(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.dof_pos[:, self.cfg.env.joint_indices] - self.default_dof_pos[:, self.cfg.env.joint_indices]), dim=1)
        return dof_error * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_stand_still_wheel(self):
        # Penalize motion at zero commands
        vel_error = torch.sum(torch.abs(self.dof_vel[:, self.cfg.env.wheel_indices]), dim=1)
        return vel_error * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques[:, self.cfg.env.joint_indices]), dim=1)

    def _reward_torques_wheel(self):
        return self.sqrdexp(torch.norm(self.torques[:, self.cfg.env.wheel_indices], dim=1))

    def _reward_energy(self):
        # Penalize torques
        return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)

    def _reward_energy_expenditure(self):
        # Penalize torques
        return torch.sum(torch.clip(torch.multiply(self.torques, self.dof_vel), 0, 1e10), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel[:, self.cfg.env.joint_indices]), dim=1)

    def _reward_dof_vel_wheel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel[:, self.cfg.env.wheel_indices]), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel[:, self.cfg.env.joint_indices] - self.dof_vel[:, self.cfg.env.joint_indices]) / self.dt), dim=1)

    def _reward_dof_acc_wheel(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel[:, self.cfg.env.wheel_indices] - self.dof_vel[:, self.cfg.env.wheel_indices]) / self.dt), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(
            self.dof_pos[:, self.cfg.env.joint_indices] - self.dof_pos_limits[self.cfg.env.joint_indices, 0] * self.cfg.rewards.soft_dof_pos_limit
        ).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (
            self.dof_pos[:, self.cfg.env.joint_indices] - self.dof_pos_limits[self.cfg.env.joint_indices, 1] * self.cfg.rewards.soft_dof_pos_limit
        ).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.dof_vel[:, self.cfg.env.joint_indices]) - self.dof_vel_limits[self.cfg.env.joint_indices] * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_dof_vel_wheel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.dof_vel[:, self.cfg.env.wheel_indices]) - self.dof_vel_limits[self.cfg.env.wheel_indices] * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_dof_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques[:, self.cfg.env.joint_indices]) - self.torque_limits[self.cfg.env.joint_indices] * self.cfg.rewards.soft_torque_limit).clip(
                min=0.0
            ),
            dim=1,
        )

    def _reward_dof_torque_wheel_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques[:, self.cfg.env.wheel_indices]) - self.torque_limits[self.cfg.env.wheel_indices] * self.cfg.rewards.soft_torque_limit).clip(
                min=0.0
            ),
            dim=1,
        )

    def _reward_action_rate(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions[:, self.cfg.env.joint_indices] - self.actions[:, self.cfg.env.joint_indices]), dim=1)
        term_2 = torch.sum(
            torch.square(
                self.actions[:, self.cfg.env.joint_indices]
                + self.last_last_actions[:, self.cfg.env.joint_indices]
                - 2.0 * self.last_actions[:, self.cfg.env.joint_indices]
            ),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions[:, self.cfg.env.joint_indices]), dim=1)
        return term_1 + term_2 + term_3

    def _reward_action_rate_wheel(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions[:, self.cfg.env.wheel_indices] - self.actions[:, self.cfg.env.wheel_indices]), dim=1)
        term_2 = torch.sum(
            torch.square(
                self.actions[:, self.cfg.env.wheel_indices]
                + self.last_last_actions[:, self.cfg.env.wheel_indices]
                - 2.0 * self.last_actions[:, self.cfg.env.wheel_indices]
            ),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions[:, self.cfg.env.wheel_indices]), dim=1)
        return term_1 + term_2 + term_3

    def _reward_action_rate_v1(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_scale(self):
        # Penalize actions scale
        rew = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return rew

    def _reward_action_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(
            torch.square(
                (
                    self.actions[:, self.cfg.env.joint_indices]
                    - 2.0 * self.last_actions[:, self.cfg.env.joint_indices]
                    + self.last_last_actions[:, self.cfg.env.joint_indices]
                )
            ),
            dim=1,
        )

    def _reward_action_acc_wheel(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(
            torch.square(
                (
                    self.actions[:, self.cfg.env.wheel_indices]
                    - 2.0 * self.last_actions[:, self.cfg.env.wheel_indices]
                    + self.last_last_actions[:, self.cfg.env.wheel_indices]
                )
            ),
            dim=1,
        )

    # -- extra --

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        contact_sum = torch.sum(contacts, dim=1)
        # rotate_cmd = self.commands[:, 2] > 0.05
        # contact_sum *= rotate_cmd
        return contact_sum > 1.0  #  / 4.0

    def _reward_no_fly_v1(self):
        no_contacts = self.contact_forces[:, self.feet_indices, 2] < 0.5
        no_contact_sum = torch.sum(no_contacts, dim=1)
        return no_contact_sum < 4.0  # / 4.0

    def _reward_no_fly_v2(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        stance_mask = self._get_gait_stance_mask()
        no_contacts = ~contacts
        swing_mask = 1.0 - stance_mask
        contact_sum = torch.sum(contacts * stance_mask, dim=1)
        swing_sum = torch.sum(no_contacts * swing_mask, dim=1)
        return contact_sum >= 1.0  # + swing_sum >= 1.0  # contact_sum > 1.0

    def _reward_no_fly_v3(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        # stance_mask = self._get_gait_stance_mask()
        contact_sum = torch.sum(contacts, dim=1)
        return contact_sum

    def _reward_joint_deviation_legs(self):
        joint_diff = self.dof_pos[:, self.cfg.env.joint_indices] - self.default_dof_pos[:, self.cfg.env.joint_indices]
        # return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
        # return 0.01 * torch.sum(torch.abs(joint_diff), dim=1)
        return 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos[:, self.cfg.env.joint_indices] - self.default_dof_pos[:, self.cfg.env.joint_indices]
        # return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
        return 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_default_joint_pos_v2(self):
        roll_l2_joint_diff = self.dof_pos[:, [0, 4, 8, 12]] - self.default_dof_pos[:, [0, 4, 8, 12]]
        hip_l2_joint_diff = self.dof_pos[:, [1, 5, 9, 13]] - self.default_dof_pos[:, [1, 5, 9, 13]]
        knee_l1_joint_diff = self.dof_pos[:, [2, 6, 10, 14]] - self.default_dof_pos[:, [2, 6, 10, 14]]
        # return torch.sum(torch.abs(hip_l2_joint_diff))
        rew_roll = 0.01 * torch.norm(roll_l2_joint_diff, dim=1)  # -0.01
        rew_hip = 0.01 * torch.norm(hip_l2_joint_diff, dim=1)  # -0.01
        rew_knee = 0.02 * torch.norm(knee_l1_joint_diff, dim=1)  # -0.001
        return rew_roll + rew_hip + rew_knee

    def _reward_ref_joint_pos(self):
        joint_indices = [1, 2, 5, 6, 9, 10, 13, 14]  # hip + knee simple ref track
        diff_joint = self.dof_pos[:, joint_indices] - self.ref_dof_pos[:, joint_indices]
        hip_indices = [0, 4, 8, 12]
        diff_hip = self.dof_pos[:, hip_indices] - self.ref_dof_pos[:, hip_indices]
        return 0.01 * torch.norm(diff_joint, dim=1) + 0.0 * torch.norm(diff_hip, dim=1)

    def _reward_joint_mirror(self):
        group_rf = [0, 1, 2]  # RF
        group_lf = [3, 4, 5]  # LF
        group_rb = [6, 7, 8]  # RB
        group_lb = [9, 10, 11]  # LB
        diff1 = torch.norm(self.dof_pos[:, self.cfg.env.joint_indices[group_rf]] + self.dof_pos[:, self.cfg.env.joint_indices[group_lb]], dim=1)
        diff2 = torch.norm(self.dof_pos[:, self.cfg.env.joint_indices[group_lf]] + self.dof_pos[:, self.cfg.env.joint_indices[group_rb]], dim=1)
        reward = 0.5 * (diff1 + diff2)
        return reward

    # -- eth --

    def _reward_tracking_lin_vel_eth(self):
        stand_still = torch.norm(self.commands[:, :2], dim=1) < 0.05
        rew_stand_still = 2.0 * torch.exp(-2.0 * torch.square(torch.norm(self.base_lin_vel[:, :2], dim=1)))
        rew_vel_tracking = torch.exp(-2.0 * torch.square(torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)))
        rew_vel_tracking += torch.sum(self.commands[:, :2] * self.base_lin_vel[:, :2], dim=1) / (torch.norm(self.commands[:, :2], dim=1) + 0.02)
        # rew_vel_tracking += torch.multiply(self.commands[:, :2], self.base_lin_vel[:, :2])
        return rew_stand_still * stand_still + rew_vel_tracking * ~stand_still

    # --------- PBRS ---------------------------------------

    def _reward_default_joint_regularization(self):
        roll_l2_joint_diff = self.dof_pos[:, [0, 4, 8, 12]] - self.default_dof_pos[:, [0, 4, 8, 12]]
        hip_l2_joint_diff = self.dof_pos[:, [1, 5, 9, 13]] - self.default_dof_pos[:, [1, 5, 9, 13]]
        knee_l1_joint_diff = self.dof_pos[:, [2, 6, 10, 14]] - self.default_dof_pos[:, [2, 6, 10, 14]]
        # return torch.sum(torch.abs(hip_l2_joint_diff))
        rew_roll = self.sqrdexp(torch.norm(roll_l2_joint_diff, dim=1))
        rew_hip = self.sqrdexp(torch.norm(hip_l2_joint_diff, dim=1))
        rew_knee = self.sqrdexp(0.2 * torch.norm(knee_l1_joint_diff, dim=1))
        return rew_roll + rew_hip + rew_knee

    def _reward_ref_joint_regularization(self):
        joint_indices = [1, 2, 5, 6, 9, 10, 13, 14]
        diff_joint = self.dof_pos[:, joint_indices] - self.ref_dof_pos[:, joint_indices]
        hip_indices = [0, 4, 8, 12]
        diff_hip = self.dof_pos[:, hip_indices] - self.ref_dof_pos[:, hip_indices]
        rew_diff = self.sqrdexp(0.1 * torch.norm(diff_joint, dim=1) + 1.0 * torch.norm(diff_hip, dim=1))
        return rew_diff

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.0
        # Yaw joints regularization around 0
        error += self.sqrdexp((self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp((self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint symmetry
        error += self.sqrdexp((self.dof_pos[:, 1] - self.dof_pos[:, 6]) / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        error += self.sqrdexp((self.dof_pos[:, 2] + self.dof_pos[:, 7]) / self.cfg.normalization.obs_scales.dof_pos)
        return error / 4

    def _reward_ang_vel_xy_regularization(self):
        # Penalize xy axes base angular velocity
        return self.sqrdexp(torch.norm(self.base_ang_vel[:, :2]))

    def _reward_lin_vel_z_regularization(self):
        # Penalize z axis base linear velocity
        return self.sqrdexp(self.base_lin_vel[:, 2])

    def _reward_dof_acc_reg(self):
        return self.sqrdexp(torch.norm((self.last_dof_vel - self.dof_vel) / self.dt, dim=1))

    def _reward_dof_vel_reg(self):
        return self.sqrdexp(torch.norm(self.dof_vel[:, self.cfg.env.joint_indices], dim=1))

    def _reward_torques_reg(self):
        return self.sqrdexp(torch.norm(self.torques[:, self.cfg.env.joint_indices], dim=1))

    def _reward_torques_wheel_reg(self):
        return self.sqrdexp(torch.norm(self.torques[:, self.cfg.env.wheel_indices], dim=1))

    def _reward_action_rate_reg(self):
        term_1 = self.sqrdexp(torch.norm(self.last_actions - self.actions, dim=1))
        term_2 = self.sqrdexp(torch.norm(self.actions + self.last_last_actions - 2.0 * self.last_actions, dim=1))
        term_3 = 0.05 * self.sqrdexp(torch.norm(self.actions, dim=1))
        return term_1 + term_2 + term_3

    # -- Potential-based rewards --

    def _reward_orientation_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_orientation() - self.rwd_prev["orientation"])
        return delta_phi / self.dt

    def _reward_tracking_lin_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_prev["tracking_lin_vel"])
        return delta_phi / self.dt

    def _reward_tracking_ang_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_prev["tracking_ang_vel"])
        return delta_phi / self.dt

    def _reward_ang_vel_xy_reg_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_ang_vel_xy_regularization() - self.rwd_prev["ang_vel_xy_regularization"])
        return delta_phi / self.dt

    def _reward_lin_vel_z_reg_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_lin_vel_z_regularization() - self.rwd_prev["lin_vel_z_regularization"])
        return delta_phi / self.dt

    def _reward_feet_air_time_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_feet_air_time() - self.rwd_prev["feet_air_time"])
        return delta_phi / self.dt

    def _reward_feet_land_time_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_feet_land_time() - self.rwd_prev["feet_land_time"])
        return delta_phi / self.dt

    def _reward_joint_reg_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_joint_regularization() - self.rwd_prev["joint_regularization"])
        return delta_phi / self.dt

    def _reward_default_joint_reg_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_default_joint_regularization() - self.rwd_prev["default_joint_regularization"])
        return delta_phi / self.dt

    def _reward_ref_joint_reg_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_ref_joint_regularization() - self.rwd_prev["ref_joint_regularization"])
        return delta_phi / self.dt

    def _reward_base_height_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_base_height() - self.rwd_prev["base_height"])
        return delta_phi / self.dt

    def _reward_dof_acc_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_dof_acc_reg() - self.rwd_prev["dof_acc_reg"])
        return delta_phi / self.dt

    def _reward_dof_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_dof_vel_reg() - self.rwd_prev["dof_vel_reg"])
        return delta_phi / self.dt

    def _reward_torques_reg_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_torques_reg() - self.rwd_prev["torques_reg"])
        return delta_phi / self.dt

    def _reward_torques_wheel_reg_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_torques_reg() - self.rwd_prev["torques_wheel_reg"])
        return delta_phi / self.dt

    def _reward_action_rate_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_action_rate_reg() - self.rwd_prev["action_rate_reg"])
        return delta_phi / self.dt

    ## ------ barrier_function -------

    def _reward_barrier_tracking_linx_vel(self):
        error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        return self._relaxed_barrier_function(error, -0.4, 0.4, 0.2)

    def _reward_barrier_tracking_liny_vel(self):
        error = self.base_lin_vel[:, 1] - self.commands[:, 1]
        return self._relaxed_barrier_function(error, -0.4, 0.4, 0.2)

    def _reward_barrier_tracking_ang_vel(self):
        error = self.base_ang_vel[:, 2] - self.commands[:, 2]
        return self._relaxed_barrier_function(error, -0.4, 0.4, 0.2)

    def _reward_barrier_ref_joint_regularization(self):
        error = self.dof_pos[:, self.cfg.env.joint_indices] - self.ref_dof_pos[:, self.cfg.env.joint_indices]
        torch.sum(torch.abs(error), dim=-1)
        return self._relaxed_barrier_function(error, -0.4, 0.4, 0.2)

    def _reward_barrier_gait_fl(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fl
        g = torch.sin(2 * np.pi * phase)
        f = torch.where(self.contact_filt[:, 1], g, -g)
        r = self._relaxed_barrier_function(f, -0.6, 2.0, 0.1)
        return r

    def _reward_barrier_gait_fr(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_fr
        g = torch.sin(2 * np.pi * phase)
        f = torch.where(self.contact_filt[:, 0], g, -g)
        r = self._relaxed_barrier_function(f, -0.6, 2.0, 0.1)
        return r

    def _reward_barrier_gait_hl(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hl
        g = torch.sin(2 * np.pi * phase)
        f = torch.where(self.contact_filt[:, 3], g, -g)
        r = self._relaxed_barrier_function(f, -0.6, 2.0, 0.1)
        return r

    def _reward_barrier_gait_hr(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time + self.cfg.rewards.phase_offset_hr
        g = torch.sin(2 * np.pi * phase)
        f = torch.where(self.contact_filt[:, 2], g, -g)
        r = self._relaxed_barrier_function(f, -0.6, 2.0, 0.1)
        return r

    def _reward_barrier_joint_limit(self):
        joint_pos = self.dof_pos[:, 6:]
        joint_limits_lower = self.dof_pos_limits[6:, 0]
        joint_limits_upper = self.dof_pos_limits[6:, 1]
        r = torch.sum(self._relaxed_barrier_function(joint_pos, joint_limits_lower, joint_limits_upper, 0.08), dim=-1)
        return r

    def _reward_barrier_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        error = base_height - self.cfg.rewards.base_height_target
        return torch.sum(self._relaxed_barrier_function(error, -0.1, 0.1, 0.2), dim=-1)

    def _reward_barrier_orientation(self):
        error = self.projected_gravity[:, :2]
        return torch.sum(self._relaxed_barrier_function(error, -0.1, 0.1, 0.2), dim=-1)

    # -- HELPER FUNCTIONS --

    def _relaxed_barrier_function(self, c: torch.Tensor, d_lower, d_upper, delta):
        def B(z: torch.Tensor, delta: torch.Tensor):
            return torch.where(z > delta, torch.log(z), np.log(delta) - 0.5 * (torch.square((z - 2.0 * delta) / delta) - 1.0))

        return B(-d_lower + c, delta) + B(d_upper - c, delta)

    def sqrdexp(self, x):
        """shorthand helper for squared exponential"""
        return torch.exp(-torch.square(x) / self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        self.eps = 0.2
        p = 2.0 * torch.pi * phase * self.cfg.rewards.cycle_time  #  * self.phase_freq
        return torch.sin(p) / (2 * torch.sqrt(torch.sin(p) ** 2.0 + (self.eps) ** 2.0)) + 1.0 / 2.0
