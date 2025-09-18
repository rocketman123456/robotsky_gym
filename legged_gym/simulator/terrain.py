
from typing import Literal, Tuple

import torch

class Terrain:
    type: Literal['plane', 'trimesh']

    def __init__(self, type:Literal['plane', 'trimesh'], bounds=None, scale=None, height_field=None):
        self.type = type
        if type == 'trimesh':
            assert (bounds is not None and scale is not None)
            self._bounds = bounds
            self._scale = scale
            self._height_field = height_field
    @property
    def bounds(self) -> torch.Tensor:
        '''
            Returns:
                Tenor containing [[min x, min y], [max x, max y]]
        '''
        return self._bounds

    def height(self, pos:torch.Tensor):
        '''
            compute terrain heights at pos

            Params:
                - pos: (n, 2) Tensor

            Returns:
                Tensor of (n,)
        '''
        if self.type == 'plane':
            return torch.zeros_like(pos[:, 0])

        if self._height_field is None:
            raise NotImplementedError(f'height field is not provided for {self.type}')

        pos = pos[:, :2] - self._bounds[:1, :]
        x = pos[:, 0] / self._scale
        y = pos[:, 1] / self._scale
        x1 = torch.floor(x).to(dtype=torch.int)
        y1 = torch.floor(y).to(dtype=torch.int)
        x2 = x1 + 1
        y2 = y1 + 1
        return (
            (x2 - x) * (y2 - y) * self._height_field[x1, y1]
            + (x - x1) * (y2 - y) * self._height_field[x2, y1]
            + (x2 - x) * (y - y1) * self._height_field[x1, y2]
            + (x - x1) * (y - y1) * self._height_field[x2, y2]
        )
