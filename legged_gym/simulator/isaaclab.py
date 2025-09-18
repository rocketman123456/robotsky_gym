from collections.abc import Sequence
import os
import sys
from dataclasses import MISSING, fields
import inspect

import numpy as np
import torch
from torch import Tensor

import trimesh
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg, ArticulationData
from isaaclab.managers import EventManager
from isaaclab.sensors import ContactSensorCfg, ContactSensor
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
import isaaclab.sim as sim_utils
from isaaclab.utils import math
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim.spawners import (
    GroundPlaneCfg,
    spawn_ground_plane,
    RigidBodyMaterialCfg,
)
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp, ViewerCfg
from isaaclab.envs.ui import ViewportCameraController

from isaaclab.managers import EventTermCfg, SceneEntityCfg

# from envs.sim.isaaclab_randomization import randomize_rigid_body_material

from envs.core import ConfigDict, DriveMode, SimBuffer, Simulator, randomize, type_and_device
from utils.utils import quat_from_euler_xyz, set_quat_convention, instantiate_cfg
from .terrain import Terrain


set_quat_convention("wxyz")

T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"resources/T1/T1_locomotion.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            ".*Hip_Pitch.*": -0.20,
            ".*Knee.*": 0.40,
            ".*Ankle_Pitch.*": -0.25,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*Hip.*",
                ".*Knee.*",
            ],
            # effort_limit={
            #     ".*Hip_Pitch.*": 45,
            #     ".*Hip_Roll.*": 30,
            #     ".*Hip_Yaw.*": 30,
            #     ".*Knee.*": 60
            # },
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*Hip_Pitch.*": 200.0,
                ".*Hip_Roll.*": 150.0,
                ".*Hip_Yaw.*": 150.0,
                ".*Knee.*": 200.0,
            },
            damping={
                ".*Hip.*": 5.0,
                ".*Knee.*": 5.0,
            },
            armature={
                ".*Hip.*": 0.01,
                ".*Knee.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*Ankle.*"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)


@configclass
class LocomotionSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING

    contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=1.0, debug_vis=True
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=150.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            enable_color_temperature=True,
            color_temperature=8000,
        ),
    )


PLAIN_TERRAIN_SCENE_CFG = LocomotionSceneCfg(
    env_spacing=1.5,
    replicate_physics=True,
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     debug_vis=True,
    #     physics_material = RigidBodyMaterialCfg(
    #         static_friction = 1.0,
    #         dynamic_friction = 1.0,
    #         restitution = 0.,
    #     ),
    # ),
    # ground terrain
    terrain=TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    ),
    contact=ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        # filter_prim_paths_expr = ['/World/ground/GroundPlane/CollisionPlane']
    ),
)

ROUGH_TERRAIN_SCENE_CFG = LocomotionSceneCfg(
    env_spacing=4.0,
    replicate_physics=True,
    # robot = T1_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    terrain=TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        debug_vis=True,
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=5,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                #     proportion=0.2,
                #     step_height_range=(0.05, 0.23),
                #     step_width=0.3,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),
                # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                #     proportion=0.2,
                #     step_height_range=(0.05, 0.23),
                #     step_width=0.3,
                #     platform_width=3.0,
                #     border_width=1.0,
                #     holes=False,
                # ),
                "boxes": terrain_gen.MeshRandomGridTerrainCfg(proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.05), platform_width=1.0),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.3, noise_range=(-0.02, 0.02), noise_step=0.02, border_width=0.25),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.3, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.2, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
                ),
            },
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    ),
    contact=ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=6, debug_vis=True, filter_prim_paths_expr=["/World/ground/terrain/mesh"]
    ),
)

for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAIN_SCENE_CFG.terrain.terrain_generator.sub_terrains.items():
    sub_terrain_cfg.flat_patch_sampling = {"flat": FlatPatchSamplingCfg(num_patches=10, patch_radius=0.5, max_height_diff=0.05)}


@configclass
class DomainRandomizeCfg:
    robot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "static_friction_range": (0.1, 2.0),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.1, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.95, 1.05],
            "damping_distribution_params": [0.95, 1.05],
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_parameters = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 2.0),
            "armature_distribution_params": (0.0, 0.02),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    # robot_root_mass = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*Trunk"),
    #         "mass_distribution_params": (-0.1, 0.1),
    #         "operation": 'add',
    #         "distribution": 'uniform',
    #         "recompute_inertia": True
    #     }
    # )
    reset_gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.7),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # joint_gains = EventTermCfg(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": [0.95, 1.05],
    #         "damping_distribution_params": [0.95, 1.05],
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # add_joint_default_pos = EventTermCfg(
    #     func=mdp.randomize_joint_default_pos,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "pos_distribution_params": (-0.01, 0.01),
    #         "operation": "add",
    #     },
    # )

    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            # "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # # interval
    # push_robot = EventTermCfg(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )


class _IsaacLabSimBuffer(SimBuffer):
    def __init__(self, sim: "IsaacLabSim"):
        self._sim = sim
        self._data = sim.robot.data
        self._dof_indexes = sim._dof_indexes
        self._body_contact_indexes = [self._sim.contact.body_names.index(name) for name in self._sim.robot.body_names]
        self._external_force = torch.zeros(sim.num_envs, sim.num_bodies, 3, **type_and_device())
        self._external_torque = torch.zeros(sim.num_envs, sim.num_bodies, 3, **type_and_device())
        self._extras: dict[str, Tensor] = {}
        if sim.contact.data.force_matrix_w is not None:
            self._extras["filtered_contact"] = torch.zeros_like(sim.contact.data.force_matrix_w, dtype=torch.float)

    @property
    def root_states(self) -> torch.Tensor:
        """
        Returns:
            (num_envs, 13)    in world frame
        """
        return self._data.root_link_state_w

    @property
    def dof_pos(self) -> Tensor:
        """
        Returns:
            (num_envs, num_dofs)
        """
        return self._data.joint_pos[:, self._dof_indexes]

    @property
    def dof_vel(self) -> Tensor:
        """
        Returns:
            (num_envs, num_dofs)
        """
        return self._data.joint_vel[:, self._dof_indexes]

    @property
    def body_states(self) -> torch.Tensor:
        """
        Returns:
            (num_envs, num_bodies, 13)
        """
        return self._data.body_link_state_w

    @property
    def joint_torque(self) -> torch.Tensor:
        """
        Returns:
            (num_envs, num_dofs)
        """
        return self._data.applied_torque[:, self._dof_indexes]

    @property
    def contact_forces(self) -> torch.Tensor:
        """
        Returns:
            (num_envs, num_bodies, 3)
        """
        if self._sim.contact.data.net_forces_w is not None:
            return self._sim.contact.data.net_forces_w[:, self._body_contact_indexes]
        raise NotImplementedError(f"{self.__class__}.contact_forces")

    @property
    def push_force(self) -> Tensor:
        if self._external_force is None:
            raise NotImplementedError(f"{self.__class__}.external_force")
        return self._external_force

    @property
    def push_torque(self) -> Tensor:
        if self._external_torque is None:
            raise NotImplementedError(f"{self.__class__}.external_torque")
        return self._external_torque

    @property
    def extras(self) -> dict[str, Tensor]:
        if self._sim.contact.data.force_matrix_w is not None:
            self._extras["filtered_contact"][:] = self._sim.contact.data.force_matrix_w[:, self._body_contact_indexes]
        return self._extras


class IsaacLabSim(Simulator):
    scene: InteractiveScene
    terrain: Terrain

    @property
    def default_dof_props(self):
        return {
            "kp": self.robot.data.default_joint_stiffness[0, self._dof_indexes],
            "kd": self.robot.data.default_joint_damping[0, self._dof_indexes],
            "effort_limit": self._default_effort_limits[0, self._dof_indexes],
        }

    @property
    def dof_kp(self):
        return self.robot.data.joint_stiffness[:, self._dof_indexes]

    @property
    def dof_kd(self):
        return self.robot.data.joint_damping[:, self._dof_indexes]

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)

        self.headless = self.cfg["basic"]["headless"]
        sim_cfg = sim_utils.SimulationCfg(
            device=self.device,
            dt=self.physics_dt,
            physx=sim_utils.PhysxCfg(
                solver_type=1, max_position_iteration_count=4, max_velocity_iteration_count=0, bounce_threshold_velocity=0.2, gpu_max_rigid_contact_count=2**23
            ),
            render=sim_utils.RenderCfg(
                enable_reflections=True,
                enable_global_illumination=True,
                antialiasing_mode="DLAA",
                enable_dl_denoiser=True,
                samples_per_pixel=64,
                enable_ambient_occlusion=True,
            ),
        )

        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

        print("setting up scene")
        self._setup_scence()
        print("setup_scene finished")
        # start the simulator
        self.sim.reset()

        self.env_origins = self.scene.env_origins

        self.num_dofs = self.robot.num_joints
        self.num_bodies = self.robot.num_bodies
        self.dof_names = self.robot.joint_names
        if self.cfg.asset.reorder_joints is not None:
            joints_order = instantiate_cfg(self.cfg.asset.reorder_joints)
            self._dof_indexes = [self.robot.joint_names.index(name) for name in joints_order]
            self.dof_names = joints_order
        else:
            self._dof_indexes = slice(None)

        self._default_effort_limits = self.robot.data.joint_effort_limits.clone()

        self._get_terrain_info()

        self._get_joint_limit()

        self._get_body_indices(self.cfg.asset)

        self.torso_noise = torch.zeros(self.num_envs, 4, **type_and_device())

        # self._create_viewer()
        self.cfg.viewer = ViewerCfg(**self.cfg.viewer)
        self.viewer = ViewportCameraController(self, self.cfg.viewer)

        # init configs
        self.default_dof_pos = self.robot.data.default_joint_pos[:1, self._dof_indexes]

        # init buf
        self.scene.update(self.physics_dt)
        self.sim_buf = self.get_buffer()

        self._setup_domain_randomization()

    def _setup_domain_randomization(self):
        self._domain_randomizers = {}
        if self.cfg.sim.randomization is None:
            return
        self.cfg.sim.randomization = instantiate_cfg(self.cfg.sim.randomization)
        print(self.cfg.sim.randomization)
        for field in fields(self.cfg.sim.randomization):
            name = field.name
            event_cfg = getattr(self.cfg.sim.randomization, name)
            print(f"{event_cfg=}")

            for key, value in event_cfg.params.items():
                if isinstance(value, SceneEntityCfg):
                    # load the entity
                    try:
                        value.resolve(self.scene)
                    except ValueError as e:
                        raise ValueError(f"Error while parsing '{name}:{key}'. {e}")

            if inspect.isclass(event_cfg.func):
                func = event_cfg.func(event_cfg, self)
            else:
                func = event_cfg.func

            if event_cfg.mode == "reset":
                self._domain_randomizers[name] = (func, event_cfg.params)
            elif event_cfg.mode == "startup":
                func(self, None, **event_cfg.params)

    @staticmethod
    def mesh_to_heightfield(mesh: trimesh.Trimesh, step=0.1, *, bary_thresh=-0.01, eps=0.0001):
        # 获取网格的边界框
        bounds = mesh.bounds
        x_min, y_min = bounds[0, :2]
        x_max, y_max = bounds[1, :2]

        grid_size = (
            int((x_max - x_min) / step) + 1,
            int((y_max - y_min) / step) + 1,
        )

        # 初始化高度矩阵
        heightfield = np.full(grid_size, -np.inf)

        # 遍历每个三角形
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            # skip vertical triangles
            edge = vertices[1:, :2] - vertices[:1, :2]
            if np.abs(np.linalg.norm(edge[0]) * np.linalg.norm(edge[1]) - np.abs(np.dot(edge[0], edge[1]))) < 1e-9:
                continue
            # 获取三角形顶点的 XY 范围
            tri_x_min = vertices[:, 0].min()
            tri_x_max = vertices[:, 0].max()
            tri_y_min = vertices[:, 1].min()
            tri_y_max = vertices[:, 1].max()

            # 确定三角形在高度场网格中的范围
            x_start = int((tri_x_min - x_min) / step)
            x_end = int((tri_x_max - x_min + eps) / step) + 1
            y_start = int((tri_y_min - y_min) / step)
            y_end = int((tri_y_max - y_min + eps) / step) + 1

            xx, yy = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end), indexing="ij")

            bary = trimesh.triangles.points_to_barycentric(
                np.repeat(vertices[np.newaxis, :, :2], xx.size, axis=0), np.c_[xx.ravel() * step + x_min, yy.ravel() * step + y_min]
            )

            pz = np.dot(bary, vertices[:, 2]).reshape(x_end - x_start, y_end - y_start)
            inside = (bary >= bary_thresh).all(axis=-1).reshape(x_end - x_start, y_end - y_start)
            heightfield[xx[inside], yy[inside]] = np.maximum(pz[inside], heightfield[xx[inside], yy[inside]])

        # from PIL import Image
        # heightfield[heightfield == -np.inf] = -2
        # min_val = np.min(heightfield)
        # max_val = np.max(heightfield)
        # print(f'{heightfield.shape=}')
        # print(f'{min_val=}, {max_val=}')
        # normalized_height_field = 255 * (heightfield - min_val) / (max_val - min_val)
        # image_data = normalized_height_field.astype(np.uint8)
        # image = Image.fromarray(image_data, mode='L')
        # image.save('heightfield.png')

        return torch.tensor(heightfield, **type_and_device())

    def _get_terrain_info(self):
        match self.scene.terrain.cfg.terrain_type:
            case "plane":
                self.terrain = Terrain("plane")
            case "generator":
                mesh = self.scene.terrain.meshes["terrain"]
                bounds = torch.tensor(mesh.bounds[:, :2], **type_and_device())
                scale = self.scene.terrain.cfg.terrain_generator.horizontal_scale
                # height_field = self.mesh_to_heightfield(mesh, scale)
                # self.terrain = Terrain('trimesh', bounds, scale, height_field)
                self.terrain = Terrain("trimesh", bounds, scale, None)
            case _ as _type:
                raise NotImplementedError(_type)

    def _setup_scence(self):
        self.cfg.sim.scene = instantiate_cfg(self.cfg.sim.scene).replace(num_envs=self.num_envs)

        self.cfg.sim.scene.robot = instantiate_cfg(self.cfg.asset.robot_cfg).replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene = InteractiveScene(self.cfg.sim.scene)

        self.robot: Articulation = self.scene.articulations["robot"]

        self.contact: ContactSensor = self.scene.sensors["contact"]  # type: ignore

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=['World/ground'])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def render(self, render_mode: str | None = None) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors():
            self.sim.render()
        # decide the rendering mode
        if render_mode == "human" or render_mode is None:
            return None
        elif render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution)
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(f"Render mode '{render_mode}' is not supported.")

    def get_buffer(self) -> SimBuffer:
        if hasattr(self, "sim_buf") and self.sim_buf is not None:
            return self.sim_buf
        external_force = torch.zeros(self.num_envs, self.num_bodies, 3, **type_and_device())
        external_torque = torch.zeros(self.num_envs, self.num_bodies, 3, **type_and_device())
        if self.contact.data.net_forces_w is not None:
            contact_force = torch.zeros_like(self.contact.data.net_forces_w, dtype=torch.float)
        else:
            contact_force = None
        self.sim_buf = SimBuffer(
            root_state=torch.zeros_like(self.robot.data.root_link_state_w, dtype=torch.float),
            joint_pos=torch.zeros_like(self.robot.data.joint_pos, dtype=torch.float),
            joint_vel=torch.zeros_like(self.robot.data.joint_vel, dtype=torch.float),
            joint_torque=torch.zeros_like(self.robot.data.applied_torque, dtype=torch.float),
            body_state=torch.zeros_like(self.robot.data.body_link_state_w, dtype=torch.float),
            contact_force=contact_force,
            external_force=external_force,
            external_torque=external_torque,
        )

        if self.contact.data.force_matrix_w is not None:
            self.sim_buf.extras["filtered_contact"] = torch.zeros_like(self.contact.data.force_matrix_w, dtype=torch.float)
        self._refresh_buffer()
        # self.sim_buf = _IsaacLabSimBuffer(self)
        return self.sim_buf

    def _refresh_buffer(self):
        # self.scene.update(self.physics_dt)

        self.sim_buf.root_states[:] = self.robot.data.root_link_state_w
        self.sim_buf.dof_pos[:] = self.robot.data.joint_pos[:, self._dof_indexes]
        self.sim_buf.dof_vel[:] = self.robot.data.joint_vel[:, self._dof_indexes]
        self.sim_buf.joint_torque[:] = self.robot.data.applied_torque[:, self._dof_indexes]
        self.sim_buf.body_states[:] = self.robot.data.body_link_state_w
        if self.contact.data.net_forces_w is not None:
            self.sim_buf.contact_forces[:, self._contact_body_indices] = self.contact.data.net_forces_w

        if self.contact.data.force_matrix_w is not None:
            self.sim_buf.extras["filtered_contact"][:, self._contact_body_indices] = self.contact.data.force_matrix_w

    def reset(self, env_ids: Tensor) -> SimBuffer:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self.scene.reset(env_ids)

        # if len(env_ids) > 0:
        #     self._reset_dofs(env_ids)
        #     self._reset_root_states(env_ids)

        if env_ids.size(0) > 0:
            for name, (func, params) in self._domain_randomizers.items():
                func(self, env_ids, **params)

        return super().reset(env_ids)

    def _start_simulate(self) -> None:
        pass

    def _simulate_step(self, ctrls: Tensor):
        if self.drive_mode == DriveMode.POSITION:
            self.robot.set_joint_position_target(ctrls, joint_ids=self._dof_indexes)
        elif self.drive_mode == DriveMode.EFFORT:
            self.robot.set_joint_effort_target(ctrls, joint_ids=self._dof_indexes)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

    def set_push_force(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self.robot.set_external_force_and_torque(self.sim_buf.push_force[env_ids], self.sim_buf.push_torque[env_ids], env_ids=env_ids)
        self.robot.write_data_to_sim()

    def set_dof_state(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if len(env_ids) == 0:
            return

        self.robot.write_joint_state_to_sim(self.sim_buf.dof_pos[env_ids].clone(), self.sim_buf.dof_vel[env_ids].clone(), self._dof_indexes, env_ids)

    def set_root_state(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == 0:
            return
        self.robot.write_root_pose_to_sim(self.sim_buf.root_states[env_ids, :7].clone(), env_ids)
        self.robot.write_root_velocity_to_sim(self.sim_buf.root_states[env_ids, 7:].clone(), env_ids)

    def _reset_dofs(self, env_ids: Sequence[int]):
        self.sim_buf.dof_pos[env_ids] = randomize(self.default_dof_pos.repeat(len(env_ids), 1), self.cfg.randomization.init_dof_pos)
        self.sim_buf.dof_vel[env_ids] = 0.0
        self.set_dof_state(env_ids)

    def _reset_root_states(self, env_ids):

        self.sim_buf.root_states[env_ids, :] = self.robot.data.default_root_state[env_ids]
        self.sim_buf.root_states[env_ids, :3] += self.scene.env_origins[env_ids]

        self.sim_buf.root_states[env_ids, :2] = randomize(self.sim_buf.root_states[env_ids, :2], self.cfg.randomization.init_base_pos_xy)

        self.sim_buf.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), **type_and_device()),
            torch.zeros(len(env_ids), **type_and_device()),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
        self.sim_buf.root_states[env_ids, 7:9] = randomize(torch.zeros(len(env_ids), 2, **type_and_device()), self.cfg.randomization.init_base_lin_vel_xy)
        self.set_root_state(env_ids)

    def _get_joint_limit(self):
        self.dof_pos_limits = self.robot.data.joint_limits[0, self._dof_indexes].clone()
        self.dof_vel_limits = self.robot.data.joint_velocity_limits[0, self._dof_indexes].clone()

        dof_torque_limits = torch.zeros_like(self.dof_vel_limits)
        for name, actuator_group in self.robot.actuators.items():
            dof_torque_limits[actuator_group.joint_indices] = actuator_group.effort_limit[0, :]
        self.dof_torque_limits = dof_torque_limits[self._dof_indexes]

    def _get_body_indices(self, asset_cfg: ConfigDict):
        self.body_indices = {name: idx for idx, name in enumerate(self.robot.body_names)}
        self._contact_body_indices = [self.body_indices[name] for name in self.contact.body_names]
        self.base_index = self.body_indices[asset_cfg["base_name"]]
