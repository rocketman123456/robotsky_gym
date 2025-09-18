from legged_gym import *

if SIMULATOR == "genesis":
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
    from genesis.engine.solvers.avatar_solver import AvatarSolver
    from genesis.utils.geom import transform_by_quat, inv_quat
elif SIMULATOR == "isaacgym":
    from isaacgym import gymtorch, gymapi, gymutil

    # from isaacgym.torch_utils import *
import torch
import numpy as np
import os

from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
from legged_gym.utils.gs_utils import *
from legged_gym.simulator.simulator import Simulator

"""********** Genesis Simulator **********"""


class GenesisSimulator(Simulator):
    """Simulator class for Genesis"""

    def __init__(self, cfg, sim_params: dict, device, headless):
        self.sim_params = sim_params
        super().__init__(cfg, sim_params, device, headless)

    def _parse_cfg(self):
        self.debug = self.cfg.env.debug
        self.control_dt = self.cfg.sim.dt * self.cfg.control.decimation

    def _create_sim(self):
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.sim_params["dt"], substeps=self.sim_params["substeps"]),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.control_dt * self.cfg.control.decimation),
                camera_pos=np.array(self.cfg.viewer.pos),
                camera_lookat=np.array(self.cfg.viewer.lookat),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=self.cfg.viewer.rendered_envs_idx),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_params["dt"],
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions_gs,
                max_collision_pairs=self.cfg.sim.max_collision_pairs,
                IK_max_targets=self.cfg.sim.IK_max_targets,
            ),
            show_viewer=not self.headless,
        )
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            elif isinstance(solver, AvatarSolver):
                continue
            self.rigid_solver = solver

        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()

        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == "plane":
            self.gs_terrain = self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type == "heightfield":
            self.terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self.gs_terrain.set_friction(self.cfg.terrain.static_friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type == "heightfield":
            # give a small margin(1.0m)
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_x_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type == "plane":  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length / 2 + 1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length / 2 - 1
            # the plane is a square
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length / 2 + 1
            self.terrain_y_range[1] = self.cfg.terrain.plane_length / 2 - 1

    def _create_envs(self):
        # Create envs
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                merge_fixed_links=True,
                links_to_keep=self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot_gs),
                fixed=self.cfg.asset.fix_base_link,
            ),
            # visualize_contact=self.debug,
        )

        # build
        self.scene.build(n_envs=self.num_envs)

        self._get_env_origins()

        self.num_dof = len(self.cfg.asset.dof_names)
        self._init_domain_params()

        # name to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.cfg.asset.dof_names]

        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self.robot.links]
        print(f"all link names: {all_link_names}")
        print("termination link indices:", self.termination_contact_indices)
        self.penalized_contact_indices = find_link_indices(self.cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self.penalized_contact_indices}")
        self.feet_names = [link.name for link in self.robot.links if self.cfg.asset.foot_name in link.name]
        self.feet_indices = find_link_indices(self.feet_names)
        print(f"feet names: {self.feet_names}, feet link indices: {self.feet_indices}")
        assert len(self.feet_indices) > 0

        # dof position limits
        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motors_dof_idx), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motors_dof_idx)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))
        # randomize joint armature
        if self.cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(np.arange(self.num_envs))
        # randomize joint friction
        if self.cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(np.arange(self.num_envs))
        # randomize joint damping
        if self.cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(np.arange(self.num_envs))
        # randomize pd gain
        if self.cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(np.arange(self.num_envs))

    def _init_buffers(self):
        self.base_init_pos = torch.tensor(self.cfg.init_state.pos, device=self.device)
        self.base_init_quat = torch.tensor(self.cfg.init_state.rot_gs, device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.dof_pos = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)
        self.base_quat_gs = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)  # quaternion in genesis definition, wxyz
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.link_contact_forces = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=torch.float)
        self.feet_pos = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=torch.float)
        self.feet_vel = torch.zeros((self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=torch.float)

        # Terrain information around feet
        self.normal_vector_around_feet = torch.zeros(self.num_envs, len(self.feet_indices) * 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.height_around_feet = torch.zeros(self.num_envs, len(self.feet_indices), 9, dtype=torch.float, device=self.device, requires_grad=False)

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name] for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=torch.float,
        )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        # PD control
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping

        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.d_gains, self.motors_dof_idx)

    def step(self, actions):
        """Simulator steps, receiving actions from the agent"""
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(actions)
            self.robot.control_dofs_force(self.torques, self.motors_dof_idx)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
            self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

    def post_physics_step(self):
        # prepare quantities
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat_gs[:] = self.robot.get_quat()
        self.base_quat[:, -1] = self.robot.get_quat()[:, 0]  # wxyz to xyzw
        self.base_quat[:, :3] = self.robot.get_quat()[:, 1:4]  # wxyz to xyzw
        self.base_euler[:] = get_euler_xyz(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot.get_vel())
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot.get_ang())
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.global_gravity)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force()
        self.feet_pos[:] = self.robot.get_links_pos()[:, self.feet_indices, :]
        self.feet_vel[:] = self.robot.get_links_vel()[:, self.feet_indices, :]

        self._check_base_pos_out_of_bound()

    def push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        # in Genesis, base link also has DOF, it's 6DOF if not fixed.
        dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel = torch_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        dofs_vel[:, :2] += push_vel
        self.robot.set_dofs_velocity(dofs_vel)

    def reset_idx(self, env_ids):
        self._reset_root_states(env_ids)
        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)
        if self.cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        if self.cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(env_ids)
        if self.cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        if self.cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)

        self.last_dof_vel[env_ids] = 0.0

    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.dof_pos[env_ids] = dof_pos[:]
        self.dof_vel[env_ids] = dof_vel[:]

        self.robot.set_dofs_position(
            position=self.dof_pos[env_ids],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(env_ids)

    def get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (
                self.base_pos[env_ids, :3]
            ).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.base_pos[:, :3]).unsqueeze(1)

        # When acquiring heights, the points need to add border_size
        # because in the height_samples, the origin of the terrain is at (border_size, border_size)
        points += self.cfg.terrain.border_size
        points = (points / self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        self.measured_heights = heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    def calc_terrain_info_around_feet(self):
        """Finds neighboring points around each foot for terrain height measurement."""
        foot_points = self.feet_pos + self.cfg.terrain.border_size
        foot_points = (foot_points / self.cfg.terrain.horizontal_scale).long()
        # px and py for 4 feet, num_envs*len(feet_indices)
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        # clip to the range of height samples
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        # get heights around the feet, 9 points for each foot
        heights1 = self.height_samples[px - 1, py]  # [x-0.1, y]
        heights2 = self.height_samples[px + 1, py]  # [x+0.1, y]
        heights3 = self.height_samples[px, py - 1]  # [x, y-0.1]
        heights4 = self.height_samples[px, py + 1]  # [x, y+0.1]
        heights5 = self.height_samples[px, py]  # [x, y]
        heights6 = self.height_samples[px - 1, py - 1]  # [x-0.1, y-0.1]
        heights7 = self.height_samples[px + 1, py + 1]  # [x+0.1, y+0.1]
        heights8 = self.height_samples[px - 1, py + 1]  # [x-0.1, y+0.1]
        heights9 = self.height_samples[px + 1, py - 1]  # [x+0.1, y-0.1]
        # Calculate normal vectors around feet
        dx = ((heights2 - heights1) / (self.cfg.terrain.horizontal_scale * 2)).view(self.num_envs, -1)
        dy = ((heights4 - heights3) / (self.cfg.terrain.horizontal_scale * 2)).view(self.num_envs, -1)
        for i in range(len(self.feet_indices)):
            normal_vector = torch.cat((dx[:, i].unsqueeze(1), dy[:, i].unsqueeze(1), -1 * torch.ones_like(dx[:, i].unsqueeze(1))), dim=-1).to(self.device)
            normal_vector /= torch.norm(normal_vector, dim=-1, keepdim=True)
            self.normal_vector_around_feet[:, i * 3 : i * 3 + 3] = normal_vector[:]
        # Calculate height around feet
        for i in range(9):
            self.height_around_feet[:, :, i] = eval(f"heights{i+1}").view(self.num_envs, -1)[:] * self.cfg.terrain.vertical_scale

    def draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        self.scene.clear_debug_objects()

        # When visualizing the height points, the points donot need to add border_size
        # height_points = quat_apply_yaw(self.base_quat.repeat(
        #     1, self.num_height_points), self.height_points)
        # height_points[0, :, 0] += self.base_pos[0, 0]
        # height_points[0, :, 1] += self.base_pos[0, 1]
        # height_points[0, :, 2] = self.measured_heights[0, :]

        height_points = torch.zeros(self.num_envs, 9 * len(self.feet_indices), 3, device=self.device)
        foot_points = self.feet_pos + self.cfg.terrain.border_size
        foot_points = (foot_points / self.cfg.terrain.horizontal_scale).long()
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        heights1 = self.height_samples[px - 1, py]  # [x-0.1, y]
        heights2 = self.height_samples[px + 1, py]  # [x+0.1, y]
        heights3 = self.height_samples[px, py - 1]  # [x, y-0.1]
        heights4 = self.height_samples[px, py + 1]  # [x, y+0.1]
        heights5 = self.height_samples[px, py]  # [x, y]
        heights6 = self.height_samples[px - 1, py - 1]  # [x-0.1, y-0.1]
        heights7 = self.height_samples[px + 1, py + 1]  # [x+0.1, y+0.1]
        heights8 = self.height_samples[px - 1, py + 1]  # [x-0.1, y+0.1]
        heights9 = self.height_samples[px + 1, py - 1]  # [x+0.1, y-0.1]
        for i in range(len(self.feet_indices)):
            height_points[0, i * 9 + 0, 0] = (px - 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 0, 1] = (py - 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 0, 2] = heights6.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 1, 0] = (px - 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 1, 1] = py.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 1, 2] = heights1.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 2, 0] = px.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 2, 1] = (py - 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 2, 2] = heights3.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 3, 0] = px.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 3, 1] = (py + 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 3, 2] = heights4.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 4, 0] = px.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 4, 1] = py.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 4, 2] = heights5.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 5, 0] = (px + 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 5, 1] = py.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 5, 2] = heights2.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 6, 0] = (px + 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 6, 1] = (py + 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 6, 2] = heights7.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 7, 0] = (px - 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 7, 1] = (py + 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 7, 2] = heights8.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i * 9 + 8, 0] = (px + 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 8, 1] = (py - 1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i * 9 + 8, 2] = heights9.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale

        # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        self.scene.draw_debug_spheres(height_points[0, :], radius=0.02, color=(1, 0, 0, 0.7))  # only draw for the first env

    # ------------- Callbacks --------------

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        self.num_height_points = grid_x.numel()
        self.height_points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        self.height_points[:, :, 0] = grid_x.flatten()
        self.height_points[:, :, 1] = grid_y.flatten()

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base pos: xy [-1, 1]
        if self.custom_origins:
            self.base_pos[env_ids] = self.base_init_pos
            self.base_pos[env_ids] += self.env_origins[env_ids]
            self.base_pos[env_ids, :2] += torch_rand_float(-1.0, 1.0, (len(env_ids), 2), self.device)
        else:
            self.base_pos[env_ids] = self.base_init_pos
            self.base_pos[env_ids] += self.env_origins[env_ids]
        self.robot.set_pos(self.base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)

        # base quat
        self.base_quat[env_ids, :] = self.base_init_quat.reshape(1, -1)
        base_orien_scale = self.cfg.init_state.base_ang_random_scale
        self.base_quat[env_ids, :] = quat_from_euler_xyz(
            torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), self.device).view(-1),
            torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), self.device).view(-1),
            torch_rand_float(-base_orien_scale, base_orien_scale, (len(env_ids), 1), self.device).view(-1),
        )
        self.base_quat_gs[env_ids, 0] = self.base_quat[env_ids, 3]  # xyzw to wxyz
        self.base_quat_gs[env_ids, 1:4] = self.base_quat[env_ids, 0:3]  # xyzw to wxyz
        self.robot.set_quat(self.base_quat_gs[env_ids], zero_velocity=False, envs_idx=env_ids)
        self.robot.zero_all_dofs_velocity(env_ids)

        # update projected gravity
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.global_gravity)

        # reset root states - velocity
        self.base_lin_vel[env_ids] = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        self.base_ang_vel[env_ids] = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        base_vel = torch.concat([self.base_lin_vel[env_ids], self.base_ang_vel[env_ids]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=env_ids)

    def _check_base_pos_out_of_bound(self):
        """Check if the base position is out of the terrain bounds"""
        x_out_of_bound = (self.base_pos[:, 0] >= self.terrain_x_range[1]) | (self.base_pos[:, 0] <= self.terrain_x_range[0])
        y_out_of_bound = (self.base_pos[:, 1] >= self.terrain_y_range[1]) | (self.base_pos[:, 1] <= self.terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        env_ids = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        # reset base position to initial position
        self.base_pos[env_ids] = self.base_init_pos
        self.base_pos[env_ids] += self.env_origins[env_ids]
        self.robot.set_pos(self.base_pos[env_ids], zero_velocity=False, envs_idx=env_ids)

    def _create_heightfield(self):
        """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
        self.gs_terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self.cfg.terrain.border_size, -self.cfg.terrain.border_size, 0.0),
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                height_field=self.terrain.height_field_raw,
            ),
        )
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        torques = (
            self._kp_scale * self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
            - self._kd_scale * self.batched_d_gains * self.dof_vel
        )
        return torques

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device), (self.num_envs / self.cfg.terrain.num_cols), rounding_mode="floor"
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self.cfg.env.env_spacing
            if (
                num_rows * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2
                or num_cols * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2
            ):
                spacing = min((self.cfg.terrain.plane_length / 2) / (num_rows - 1), (self.cfg.terrain.plane_length / 2) / (num_cols - 1))
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
            self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4

    def _init_domain_params(self):
        """Initializes domain randomization parameters, which are used to randomize the environment."""
        self._friction_values = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._added_base_mass = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._rand_push_vels = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_armature = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_stiffness = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_damping = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

    def _randomize_friction(self, env_ids=None):
        """Randomize friction of all links"""
        min_friction, max_friction = self.cfg.domain_rand.friction_range

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) * (max_friction - min_friction) + min_friction
        self._friction_values[env_ids] = ratios[:, 0].unsqueeze(1).detach().clone()

        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids=None):
        """Randomize base mass"""
        min_mass, max_mass = self.cfg.domain_rand.added_mass_range
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self.rigid_solver.set_links_mass_shift(
            added_mass,
            [
                base_link_id,
            ],
            env_ids,
        )

    def _randomize_com_displacement(self, env_ids):
        """Randomize center of mass displacement of the robot"""
        min_displacement_x, max_displacement_x = self.cfg.domain_rand.com_pos_x_range
        min_displacement_y, max_displacement_y = self.cfg.domain_rand.com_pos_y_range
        min_displacement_z, max_displacement_z = self.cfg.domain_rand.com_pos_z_range
        base_link_id = 1
        com_displacement = torch.zeros((len(env_ids), 1, 3), dtype=torch.float, device=self.device)

        com_displacement[:, 0, 0] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) * (max_displacement_x - min_displacement_x) + min_displacement_x
        com_displacement[:, 0, 1] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) * (max_displacement_y - min_displacement_y) + min_displacement_y
        com_displacement[:, 0, 2] = gs.rand((len(env_ids), 1), dtype=float).squeeze(1) * (max_displacement_z - min_displacement_z) + min_displacement_z
        self._base_com_bias[env_ids] = com_displacement[:, 0, :].detach().clone()

        self.rigid_solver.set_links_COM_shift(
            com_displacement,
            [
                base_link_id,
            ],
            env_ids,
        )

    def _randomize_joint_armature(self, env_ids):
        """Randomize joint armature of the robot"""
        min_armature, max_armature = self.cfg.domain_rand.joint_armature_range
        armature = torch.rand((1,), dtype=torch.float, device=self.device) * (max_armature - min_armature) + min_armature  # scalar
        self._joint_armature[env_ids, 0] = armature[0].detach().clone()
        armature = armature.repeat(self.num_actions)  # repeat for all motors
        self.robot.set_dofs_armature(armature, self.motors_dof_idx, envs_idx=env_ids)  # all environments share the same armature
        # This armature will be Refreshed when envs are reset

    def _randomize_joint_friction(self, env_ids):
        """Randomize joint friction of the robot"""
        min_friction, max_friction = self.cfg.domain_rand.joint_friction_range
        friction = torch.rand((1,), dtype=torch.float, device=self.device) * (max_friction - min_friction) + min_friction
        self._joint_friction[env_ids, 0] = friction[0].detach().clone()
        friction = friction.repeat(self.num_actions)
        self.robot.set_dofs_stiffness(friction, self.motors_dof_idx, envs_idx=env_ids)

    def _randomize_joint_damping(self, env_ids):
        """Randomize joint damping of the robot"""
        min_damping, max_damping = self.cfg.domain_rand.joint_damping_range
        damping = torch.rand((1,), dtype=torch.float, device=self.device) * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids, 0] = damping[0].detach().clone()
        damping = damping.repeat(self.num_actions)
        self.robot.set_dofs_damping(damping, self.motors_dof_idx, envs_idx=env_ids)

    def _randomize_pd_gain(self, env_ids):
        self._kp_scale[env_ids] = torch_rand_float(
            self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device
        )
        self._kd_scale[env_ids] = torch_rand_float(
            self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device
        )

    def _setup_camera(self):
        """Set camera position and direction"""
        self.floating_camera = self.scene.add_camera(
            res=(1280, 960),
            pos=np.array(self.cfg.viewer.pos),
            lookat=np.array(self.cfg.viewer.lookat),
            fov=40,
            GUI=True,
        )

        self._recording = False
        self._recorded_frames = []
