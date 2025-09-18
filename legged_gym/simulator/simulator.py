from legged_gym import *

if SIMULATOR == "genesis":
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
    from genesis.engine.solvers.avatar_solver import AvatarSolver
    from genesis.utils.geom import transform_by_quat, inv_quat
elif SIMULATOR == "isaacgym":
    from isaacgym import gymtorch, gymapi, gymutil

    # from isaacgym.torch_utils import *
elif SIMULATOR == "isaaclab":
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
import torch
import numpy as np
import os

from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
from legged_gym.utils.gs_utils import *

""" ********** Base Simulator ********** """


class Simulator:
    def __init__(self, cfg, sim_params: dict, sim_device: str = "cuda:0", headless: bool = False):
        self.height_samples = None
        self.device = sim_device
        self.headless = headless
        self.cfg = cfg
        self.num_envs = self.cfg.env.num_envs
        self.num_actions = self.cfg.env.num_actions
        self._parse_cfg()
        self._create_sim()
        self._create_envs()
        self._init_buffers()

    def _parse_cfg(self):
        raise NotImplementedError("Subclasses should implement this method")

    def _create_sim(self):
        raise NotImplementedError("Subclasses should implement this method")

    def _create_envs(self):
        raise NotImplementedError("Subclasses should implement this method")

    def _init_buffers(self):
        raise NotImplementedError("Subclasses should implement this method")

    def step(self):
        raise NotImplementedError("Subclasses should implement this method")

    def post_physics_step(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_heights(self, env_ids=None):
        raise NotImplementedError("Subclasses should implement this method")

    def push_robots(self):
        raise NotImplementedError("Subclasses should implement this method")

    def reset_idx(self, env_ids):
        raise NotImplementedError("Subclasses should implement this method")

    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        raise NotImplementedError("Subclasses should implement this method")
