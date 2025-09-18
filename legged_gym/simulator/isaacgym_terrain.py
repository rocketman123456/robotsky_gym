import numpy as np
from isaacgym import gymapi, terrain_utils
import torch
from .terrain import Terrain

class IsaacGymTerrain(Terrain):

    def __init__(self, gym, sim, device, terrain_cfg):
        self.terrain_cfg = terrain_cfg
        self.gym = gym
        self.sim = sim
        self.device = device
        self.type = self.terrain_cfg["type"]

        if self.type == "plane":
            self._create_ground_plane()
        elif self.type == "trimesh":
            self._create_trimesh()
            self._bounds = torch.tensor(
                [[-self.border_size, -self.border_size],
                 [self.env_width + self.border_size, self.env_length + self.border_size]],
                dtype=torch.float, device=self.device)
            self._scale = self.horizontal_scale
        else:
            raise ValueError(f"Invalid terrain type: {self.type}")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.terrain_cfg["static_friction"]
        plane_params.dynamic_friction = self.terrain_cfg["dynamic_friction"]
        plane_params.restitution = self.terrain_cfg["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.cols = self.terrain_cfg["cols"]
        self.rows = self.terrain_cfg["rows"]
        self.env_width = self.cols * self.terrain_cfg["terrain_width"]
        self.env_length = self.rows * self.terrain_cfg["terrain_length"]

        self.border_size = self.terrain_cfg["border_size"]
        self.horizontal_scale = self.terrain_cfg["horizontal_scale"]
        self.vertical_scale = self.terrain_cfg["vertical_scale"]
        self.border_pixels = int(self.border_size / self.horizontal_scale)
        terrain_width_pixels = int(self.terrain_cfg["terrain_width"] / self.horizontal_scale)
        terrain_length_pixels = int(self.terrain_cfg["terrain_length"] / self.horizontal_scale)
        self.height_field_raw = np.zeros(
            (
                self.cols * terrain_width_pixels + 2 * self.border_pixels,
                self.rows * terrain_length_pixels + 2 * self.border_pixels,
            ),
            dtype=np.int16,
        )
        proportions = [subterr['proportion'] for subterr in self.terrain_cfg.subterrains]
        cum_num = np.floor(np.cumsum(proportions) / np.sum(proportions) * self.cols).astype(int)

        start = 0
        for idx, sub in enumerate(self.terrain_cfg.subterrains):
            for i in range(start, cum_num[idx]):
                for j in range(self.rows):
                    terrain = terrain_utils.SubTerrain(
                        "terrain",
                        width=terrain_width_pixels,
                        length=terrain_length_pixels,
                        vertical_scale=self.vertical_scale,
                        horizontal_scale=self.horizontal_scale,
                    )
                    if sub['type'] == 'plane':
                        pass
                    elif sub['type'] == 'pyramid':
                        terrain_utils.pyramid_sloped_terrain(terrain, **sub['params'])

                    elif sub['type'] == 'random':
                        terrain_utils.random_uniform_terrain(terrain, **sub['params'])
                    elif sub['type'] == 'discrete':
                        terrain_utils.discrete_obstacles_terrain(terrain, **sub['params'])
                    start_x = self.border_pixels + i * terrain_width_pixels
                    end_x = self.border_pixels + (i + 1) * terrain_width_pixels
                    start_y = self.border_pixels + j * terrain_length_pixels
                    end_y = self.border_pixels + (j + 1) * terrain_length_pixels
                    self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
            start = cum_num[idx]

        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.height_field_raw, self.horizontal_scale, self.vertical_scale, self.terrain_cfg["slope_threshold"]
        )

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -self.border_size
        tm_params.transform.p.y = -self.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.terrain_cfg["static_friction"]
        tm_params.dynamic_friction = self.terrain_cfg["dynamic_friction"]
        tm_params.restitution = self.terrain_cfg["restitution"]
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(order="C"), triangles.flatten(order="C"), tm_params)

    def height(self, base_pos):
        if self.type == "plane":
            return torch.zeros(len(base_pos), dtype=torch.float, device=self.device)
        else:
            x = self.border_pixels + base_pos[:, 0].cpu().numpy() / self.horizontal_scale
            y = self.border_pixels + base_pos[:, 1].cpu().numpy() / self.horizontal_scale
            x1 = np.floor(x).astype(int)
            x2 = x1 + 1
            y1 = np.floor(y).astype(int)
            y2 = y1 + 1
            return torch.tensor(
                (
                    (x2 - x) * (y2 - y) * self.height_field_raw[x1, y1]
                    + (x - x1) * (y2 - y) * self.height_field_raw[x2, y1]
                    + (x2 - x) * (y - y1) * self.height_field_raw[x1, y2]
                    + (x - x1) * (y - y1) * self.height_field_raw[x2, y2]
                ) * self.vertical_scale,
                dtype=torch.float,
                device=self.device,
            )
