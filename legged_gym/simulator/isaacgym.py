import os
import sys

import numpy as np

from envs.sim.joint_model import JointModel
from .isaacgym_terrain import IsaacGymTerrain
from isaacgym import gymtorch, gymapi, gymutil
import torch
from torch import Tensor
from envs.core import ConfigDict, DriveMode, SimBuffer, Simulator, randomize, type_and_device
from utils.utils import quat_from_euler_xyz, set_quat_convention

set_quat_convention("wxyz")


class IsaacGymSim(Simulator):

    @property
    def default_dof_props(self):
        return {"kp": self._default_dof_kp, "kd": self._default_dof_kd, "effort_limit": self.dof_torque_limits}

    @property
    def dof_kp(self):
        return self._dof_kp

    @property
    def dof_kd(self):
        return self._dof_kd

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.asset_cfg = self.cfg["asset"]

        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim()
        self.terrain = IsaacGymTerrain(self.gym, self.sim, self.device, self.cfg["terrain"])
        self._create_viewer()
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self.enable_torque_est = self.cfg.sim.get("enable_torque_est", False)
        self.enable_torque_est |= self.drive_mode == DriveMode.EFFORT

        joint_q_dot_max_at_tau_max = torch.zeros(1, self.num_dofs, **type_and_device()) + 10.0

        # joint_q_dot_max_at_tau_max[:, [2,3,4,5,6,7,8,9]] = 5.2
        # joint_q_dot_max_at_tau_max[:, 10] = 12.5
        # joint_q_dot_max_at_tau_max[:, [11, 17]] = 16.
        # joint_q_dot_max_at_tau_max[:, [12, 18]] = 12.5
        # joint_q_dot_max_at_tau_max[:, [13, 19]] = 12.5
        # joint_q_dot_max_at_tau_max[:, [14, 20]] = 13.6
        # joint_q_dot_max_at_tau_max[:, [15, 21]] = 10.4
        # joint_q_dot_max_at_tau_max[:, [16, 22]] = 10.4

        # if self.enable_torque_est:
        #     self.joint_model = JointModel(
        #         self.num_envs, self.num_dofs, self.dof_torque_limits, self.dof_vel_limits,
        #         q_dot_max_at_tau_max=joint_q_dot_max_at_tau_max,
        #         max_q_offset=0.1,
        #         min_backlash=0.003,
        #         max_backlash=0.1,
        #         min_friction_dynamic=0.0,
        #         max_friction_dynamic=0.3,
        #         min_friction_static=0.0,
        #         max_friction_static=1.,
        #         device=self.device)

        #     self.joint_model.sample_params(slice(None))

        # init config
        init_base_state_list = (
            self.asset_cfg["init_state"]["pos"]
            + self.asset_cfg["init_state"]["rot"]
            + self.asset_cfg["init_state"]["lin_vel"]
            + self.asset_cfg["init_state"]["ang_vel"]
        )
        self.init_root_state = torch.tensor(init_base_state_list, device=self.device)

        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.asset_cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.asset_cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.asset_cfg["init_state"]["default_joint_angles"]["default"]

        # init buf
        self.sim_buf = self.get_buffer()

    def draw_axes(self, pos: Tensor, quat: Tensor, scale: float = 0.1):
        if self.viewer is None:
            return
        axes_geom = gymutil.AxesGeometry(scale=scale)
        quat = quat[:, [1, 2, 3, 0]]  # convert wxyz to xyzw
        for i in range(self.num_envs):
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, None, gymapi.Transform(gymapi.Vec3(*pos[i]), gymapi.Quat(*quat[i])))

    def render(self, render_mode: str = None):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
            self.gym.clear_lines(self.viewer)

        if render_mode == "rgb_array":
            viewer_cfg = self.cfg["viewer"]
            env_idx = viewer_cfg.env_index
            if self.viewer is None:
                if self.device != "cpu":
                    self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
            if self.camera is None:
                camera_props = gymapi.CameraProperties()
                camera_props.width = viewer_cfg.resolution[0]
                camera_props.height = viewer_cfg.resolution[1]
                camera_props.use_collision_geometry = False
                self.camera = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)
            root_pos = self.sim_buf.root_states[env_idx, 0:3]
            cam_pos = gymapi.Vec3(*(x + y for x, y in zip(root_pos.tolist(), self.cfg["viewer"]["eye"])))
            cam_target = gymapi.Vec3(*self.sim_buf.root_states[env_idx, 0:3].tolist())
            self.gym.set_camera_location(self.camera, self.envs[env_idx], cam_pos, cam_target)
            self.gym.render_all_camera_sensors(self.sim)
            img = self.gym.get_camera_image(self.sim, self.envs[env_idx], self.camera, gymapi.IMAGE_COLOR)
            return img.reshape(img.shape[0], -1, 4)[:, :, :3]

    def get_buffer(self) -> SimBuffer:
        if hasattr(self, "sim_buf") and self.sim_buf is not None:
            return self.sim_buf

        # get gym state tensors
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(root_state)
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        if self.enable_torque_est:
            torques = torch.zeros(self.num_envs, self.num_dofs, **type_and_device())
            self._step_acc_torque = torch.zeros_like(torques)
        else:
            torques = self.gym.acquire_dof_force_tensor(self.sim)
            torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dofs)

        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._body_state = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state).view(self.num_envs, self.num_dofs, 2)
        self.external_force = torch.zeros(self.num_envs, self.num_bodies, 3, **type_and_device())
        self.external_torque = torch.zeros(self.num_envs, self.num_bodies, 3, **type_and_device())

        sim_buf = SimBuffer(
            root_state=self._root_state[:, [0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12]],  # convert xyzw to wxyz
            joint_pos=self.dof_state[..., 0],
            joint_vel=self.dof_state[..., 1],
            joint_torque=torques,
            body_state=self._body_state[:, :, [0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12]],  # convert xyzw to wxyz
            contact_force=gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3),  # shape: num_envs, num_bodies, xyz axis
            external_force=self.external_force,
            external_torque=self.external_torque,
        )

        self.sim_buf = sim_buf
        return sim_buf

    def _refresh_buffer(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # estimate joint torques
        if self.enable_torque_est:
            self.sim_buf.joint_torque[:] = self._step_acc_torque / self.decimation

        self.sim_buf.root_states[:] = self._root_state[:, [0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
        self.sim_buf.body_states[:] = self._body_state[:, :, [0, 1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12]]

    def reset(self, env_ids: Tensor) -> SimBuffer:
        # if env_ids.size(0) > 0:
        # self._reset_dofs(env_ids)
        # self._reset_root_states(env_ids)
        # self.sim_buf.extras['noised_dof_pos'] = self.sim_buf.dof_pos
        # self.joint_model.sample_params(env_ids)
        return super().reset(env_ids)

    def _start_simulate(self) -> None:
        if self.enable_torque_est:
            self._step_acc_torque.zero_()

    def _simulate_step(self, ctrls: Tensor = None):
        if ctrls is not None:
            self.gym.refresh_dof_state_tensor(self.sim)

            if self.enable_torque_est:
                # dof_torques, q_h = self.joint_model.estimate(
                #     ctrls, self.sim_buf.dof_pos, self.sim_buf.dof_vel,
                #     self.dof_stiffness, self.dof_damping,
                #     self.sim_buf.joint_torque)
                # self.sim_buf.extras['noised_dof_pos'] = q_h

                dof_torques = self._dof_kp * (ctrls - self.sim_buf.dof_pos) - self._dof_kd * self.sim_buf.dof_vel
                self._step_acc_torque[:] += dof_torques

            if self.drive_mode == DriveMode.POSITION:
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(ctrls))
            elif self.drive_mode == DriveMode.EFFORT:
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))

        self.gym.simulate(self.sim)

    def set_push_force(self) -> None:
        self.gym.apply_rigid_body_force_tensors(
            self.sim, gymtorch.unwrap_tensor(self.sim_buf.push_force), gymtorch.unwrap_tensor(self.sim_buf.push_torque), gymapi.LOCAL_SPACE
        )

    def set_dof_state(self, env_ids: Tensor = None) -> None:
        if env_ids is None:
            assert self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        else:
            if len(env_ids) == 0:
                return
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            assert self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
            )

    def set_root_state(self, env_ids: Tensor = None) -> None:
        self._root_state[:] = self.sim_buf.root_states[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12]]
        if env_ids is None:
            assert self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
        else:
            if len(env_ids) == 0:
                return
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            assert self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
            )

    def _reset_dofs(self, env_ids):
        self.sim_buf.dof_pos[env_ids] = randomize(self.default_dof_pos.repeat(len(env_ids), 1), self.cfg.randomization.init_dof_pos)
        self.sim_buf.dof_vel[env_ids] = 0.0
        self.set_dof_state(env_ids)

    def _reset_root_states(self, env_ids):
        self.sim_buf.root_states[env_ids, :] = self.init_root_state
        self.sim_buf.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.sim_buf.root_states[env_ids, :2] = randomize(self.sim_buf.root_states[env_ids, :2], self.cfg.randomization.init_base_pos_xy)
        self.sim_buf.root_states[env_ids, 2] += self.terrain.height(self.sim_buf.root_states[env_ids, :2])
        self.sim_buf.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), **type_and_device()),
            torch.zeros(len(env_ids), **type_and_device()),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
        self.sim_buf.root_states[env_ids, 7:9] = randomize(torch.zeros(len(env_ids), 2, **type_and_device()), self.cfg.randomization.init_base_lin_vel_xy)
        self.set_root_state(env_ids)

    def _create_sim(self):
        sim_cfg = self.cfg["sim"]
        sim_device = self.cfg["basic"]["sim_device"]
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)

        # graphics device for rendering, -1 for no rendering
        self.headless = self.cfg["basic"]["headless"]
        self.graphics_device_id = self.sim_device_id
        if self.cfg.sim.graphics_device_id is not None:
            self.graphics_device_id = self.cfg.sim.graphics_device_id
        if self.headless and not self.cfg["runner"]["record_video"]:
            self.graphics_device_id = -1

        self.sim_params = gymapi.SimParams()

        # assign general sim parameters
        self.sim_params.dt = sim_cfg["dt"]
        self.sim_params.num_client_threads = sim_cfg.get("num_client_threads", 0)
        self.sim_params.use_gpu_pipeline = sim_device_type == "cuda"
        self.sim_params.substeps = sim_cfg.get("substeps", 2)

        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        # assign gravity
        self.sim_params.gravity = gymapi.Vec3(*sim_cfg["gravity"])

        # configure physics parameters
        if sim_cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            # set the parameters
            if "physx" in sim_cfg:
                for opt in sim_cfg["physx"].keys():
                    if opt == "contact_collection":
                        setattr(self.sim_params.physx, opt, gymapi.ContactCollection(sim_cfg["physx"][opt]))
                    else:
                        setattr(self.sim_params.physx, opt, sim_cfg["physx"][opt])
                setattr(self.sim_params.physx, "use_gpu", sim_device_type == "cuda")
        elif sim_cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
            # set the parameters
            if "flex" in sim_cfg:
                for opt in sim_cfg["flex"].keys():
                    setattr(self.sim_params.flex, opt, sim_cfg["flex"][opt])
        else:
            raise ValueError(f"Invalid physics engine backend: {sim_cfg['physics_engine']}")

        return self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

    def _create_viewer(self):
        self.viewer = None
        self.camera = None
        if not self.headless:
            # if running with a viewer, set up keyboard shortcuts and camera
            self.enable_viewer_sync = True
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            position = self.cfg["viewer"]["eye"]
            lookat = self.cfg["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(position[0], position[1], position[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]

        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.use_physx_armature = asset_cfg["use_physx_armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        self._get_joint_limit(robot_asset)

        self._get_body_indices(robot_asset, asset_cfg)

        start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.torso_noise = torch.zeros(self.num_envs, 4, **type_and_device())
        self._dof_kp = torch.zeros(self.num_envs, self.num_dofs, **type_and_device())
        self._dof_kd = torch.zeros(self.num_envs, self.num_dofs, **type_and_device())
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, **type_and_device())

        self._default_dof_kp = torch.zeros(self.num_dofs, **type_and_device())
        self._default_dof_kd = torch.zeros(self.num_dofs, **type_and_device())
        for i in range(self.num_dofs):
            found = False
            for name in self.asset_cfg["joints"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self._default_dof_kp[i] = self.asset_cfg["joints"]["stiffness"][name]
                    self._default_dof_kd[i] = self.asset_cfg["joints"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")

        if self.drive_mode == DriveMode.EFFORT:
            for i in range(self.num_dofs):
                found = False
                for name in self.asset_cfg["joints"]["stiffness"].keys():
                    if name in self.dof_names[i]:
                        self._dof_kp[:, i] = self.asset_cfg["joints"]["stiffness"][name]
                        self._dof_kd[:, i] = self.asset_cfg["joints"]["damping"][name]
                        found = True
                if not found:
                    raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
            self._dof_kp = randomize(self._dof_kp, self.cfg["randomization"].get("dof_stiffness"))
            self._dof_kd = randomize(self._dof_kd, self.cfg["randomization"].get("dof_damping"))

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)

            if self.drive_mode == DriveMode.POSITION:
                dof_props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
                dof_props = self._process_dof_props(dof_props)
                self._dof_kp[i] = torch.tensor(dof_props["stiffness"], **type_and_device())
                self._dof_kd[i] = torch.tensor(dof_props["damping"], **type_and_device())
                self.dof_friction[i] = torch.tensor(dof_props["friction"], **type_and_device())
                self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            elif self.drive_mode == DriveMode.EFFORT:
                dof_props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
                dof_props["friction"] = randomize(dof_props["friction"], self.cfg["randomization"].get("dof_friction"))
                self.dof_friction[i] = torch.tensor(dof_props["friction"], **type_and_device())
                self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)

            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _get_joint_limit(self, robot_asset):
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, **type_and_device())
        self.dof_vel_limits = torch.zeros(self.num_dofs, **type_and_device())
        self.dof_torque_limits = torch.zeros(self.num_dofs, **type_and_device())
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.dof_torque_limits[i] = dof_props_asset["effort"][i].item()

    def _get_body_indices(self, robot_asset, asset_cfg: ConfigDict):
        self.body_indices = self.gym.get_asset_rigid_body_dict(robot_asset)

        self.base_index = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.foot_shape_indices = []
        for name in asset_cfg["foot_names"]:
            index = self.body_indices[name]
            self.foot_shape_indices += list(range(rbs_list[index].start, rbs_list[index].start + rbs_list[index].count))

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # TODO refactor terrain
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.height(self.env_origins)

    def _process_dof_props(self, props):
        for i in range(self.num_dofs):
            props["driveMode"][i] = self.cfg["asset"]["drive_mode"]
            dof_name = self.dof_names[i]
            found = False
            for name in self.asset_cfg["joints"]["stiffness"].keys():
                if name in dof_name:
                    props["stiffness"][i] = self.asset_cfg["joints"]["stiffness"][name]
                    props["damping"][i] = self.asset_cfg["joints"]["damping"][name]
                    props["armature"][i] = self.asset_cfg["joints"]["armature"][name]
                    found = True
            if not found:
                raise ValueError(f"Props of joint {dof_name} were not defined")
        props["stiffness"] = randomize(props["stiffness"], self.cfg["randomization"].get("dof_stiffness"))
        props["damping"] = randomize(props["damping"], self.cfg["randomization"].get("dof_damping"))
        props["friction"] = randomize(props["friction"], self.cfg["randomization"].get("dof_friction"))
        return props

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_index:
                props[j].com.x, self.torso_noise[i, 0] = randomize(props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].com.y, self.torso_noise[i, 1] = randomize(props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].com.z, self.torso_noise[i, 2] = randomize(props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True)
                props[j].mass, self.torso_noise[i, 3] = randomize(props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True)
            else:
                props[j].com.x = randomize(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = randomize(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = randomize(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = randomize(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = randomize(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = randomize(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = randomize(0.0, self.cfg["randomization"].get("restitution"))
        return props
