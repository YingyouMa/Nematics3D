import numpy as np

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.field import generate_coordinate_grid, apply_linear_transform
from Nematics3D.general import select_grid_in_box
from ..opts import OptsPlaneGrid, merge_opts


class PlotPlaneGrid:

    @logging_and_warning_decorator
    def __init__(
        self,
        opts = OptsPlaneGrid(),
        logger=None,
        **kwargs
    ):
        
        for name, value in {"normal": opts.normal, "spacing1": opts.spacing1, "spacing2": opts.spacing2, "size": opts.size}.items():
            if value is None:
                raise ValueError(f"Missing required variable {name} to generate plane_grid")
        
        opts = merge_opts(opts, kwargs)
        
        self._internal_opts = opts

        self.commit(
            opts = self._internal_opts,
            logger=logger,
        )
        
        for key, value in vars(self._internal_opts).items():
            setattr(self, f"opts_{key}", value)

    def commit(
        self,
        logger=None,
        **kwargs,
    ):

        space1 = opts.spacing1
        space2 = opts.spacing2
        size = opts.size
        origin = opts.origin
        normal = opts.normal
        axis1 = opts.axis1

        num1 = int(size / space1)
        num2 = int(size / space2)

        # space1 = (size-1)/(num1-1)
        # space1 = (size-1)/(num2-1)

        if axis1 is not None:
            if normal @ axis1 != 0:
                msg = "normal must be perpendicular to axis1.\n"
                msg = "Got {normal} and {axis1}. \n"
                msg += "Discard the component aligned with normal along axis1 in the following."
                logger.info(msg)
        if axis1 is None:
            axis1 = np.random.randn(3)
            axis1 /= np.linalg.norm(axis1)

        axis1 = axis1 - axis1 @ normal * normal
        axis1 /= np.linalg.norm(axis1)
        axis_both = np.array([axis1, np.cross(normal, axis1)])

        source_shape = (size, size)
        target_shape = (num1, num2)

        grid, grid_int, spaces = generate_coordinate_grid(source_shape, target_shape)
        grid_int = np.reshape(grid_int, (-1, 2))
        grid = np.reshape(grid, (-1, 2))
        grid = np.einsum("ai, ib -> ab", grid, axis_both)

        offset = -np.average(grid, axis=0) + origin
        grid = grid + offset
        grid = apply_linear_transform(
            grid, transform=opts.grid_transform, offset=opts.grid_offset
        )
            
        if opts.corners_limit is not None:
            grid_select = select_grid_in_box(
                grid, corners_limit=opts.corners_limit, logger=logger
            )
        else:
            grid_select = grid

        self._grid = grid_select
        self._axis1 = axis1
        self._normal = normal
        self._origin = opts.origin
        self._shape = opts.shape
        self._size = opts.size
        self._spacing1 = spaces[0]
        self._spacing2 = spaces[1]
        self._grid_all = np.reshape(grid, (*target_shape, 3))
        self._offset = offset
        self._grid_int = grid_int
        self._grid_transform = opts.grid_transform
        self._grid_offset = opts.grid_offset
        self._corners_limit = opts.corners_limit
