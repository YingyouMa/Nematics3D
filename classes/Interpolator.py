import numpy as np

from Nematics3D.datatypes import Vect, as_Vect, Tensor, as_Tensor
from Nematics3D.field import apply_linear_transform


class Interpolator:

    def __init__(
        self,
        interpolator,
        grid_max,
        grid_offset: Vect(3) = np.array([0, 0, 0]),
        grid_transform: Tensor((3,3)) = np.eye(3),
    ):


        self._interpolator = interpolator
        self._grid_offset = as_Vect(grid_offset, name='grid_offset')
        self._grid_transform = as_Tensor(grid_transform, (3,3), name="grid_transform")

    def interpolate(self, points: np.ndarray, is_index=False):

        if not is_index:
            points = apply_linear_transform(
                points, transform=np.linalg.inv(self._transform), offset=-self._offset
            )

        u, v, w = self._grid_max

        u = np.arange(u)
        v = np.arange(v)
        w = np.arange(w)

        pts = np.asarray(points, dtype=float).copy()
        pts[:, 0] = np.clip(pts[:, 0], u[0], u[-1])
        pts[:, 1] = np.clip(pts[:, 1], v[0], v[-1])
        pts[:, 2] = np.clip(pts[:, 2], w[0], w[-1])

        return self._interpolator(pts)
