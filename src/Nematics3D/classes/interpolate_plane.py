import numpy as np
from typing import Mapping, Any

from .QInterpolator import QInterpolator
from Nematics3D.logging_decorator import logging_and_warning_decorator
from .plane_grid import PlaneGrid, OptsPlaneGrid
from .plane_grid_polar import PlaneGridPolar, OptsPlaneGridPolar
from .class_base import ClassBase


# InterpolatePlane is a lightweight bridge between an interpolator and a
# plane-grid sampling object.
#
# Subclasses should preserve the binding contract between `grid` and `field`,
# keep `_calc_result` synchronized with the current grid mask, and be careful
# when changing grid construction because this class currently accepts either
# Cartesian or polar plane-grid implementations.
class InterpolatePlane(ClassBase):
    """
    InterpolatePlane samples a `QInterpolator` on a plane grid.

    Normal users pass in an interpolator plus either an existing plane grid
    or plane-grid options. The sampled values are then available through
    `plane.result`. Use `plane.show_relations()` to inspect the bound grid and
    `plane.grid.show_modifiable_attrs()` to inspect grid settings.
    """

    __attrs__ = {
        **(ClassBase.__attrs__),
        "raw_name": "The name identifier of this plane object",
        "_calc_result": "The interpolated value of the physics quantity on the 2D plane grid.",
    }
    # Each interpolated plane binds to at most one grid and one interpolator at a time.
    __relations__ = {
        **(ClassBase.__relations__),
        "grid": "The plane grid associated with this interpolated field.",
        "interpolator": "The QInterpolator object used to sample this plane.",
    }

    # ==================== OVERRIDE ====================
    # InterpolatePlane overrides ClassBase.__init__ because it must create or
    # update the plane grid binding before validating the interpolator and
    # computing the first sampled result.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        interpolator: QInterpolator,
        name: str = "interpolate plane",
        grid: PlaneGrid | PlaneGridPolar | None = None,
        opts: OptsPlaneGrid | OptsPlaneGridPolar | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):

        super().__init__(name=name, name_replace="interpolate plane")

        if grid is not None:
            grid = grid.act_copy(name=self.name + "-grid")
            if opts is not None or kwargs:
                grid.act_commit(opts=opts, **kwargs)
        else:
            grid = PlaneGrid(
                opts=opts,
                opts_defaults_override=opts_defaults_override,
                name=self.name + "-grid",
                **kwargs,
            )

        self.act_bind_relation_base("grid", grid, is_weak=False)
        grid.act_bind_relation_base("field", self, is_weak=True)

        if not isinstance(interpolator, QInterpolator):
            raise TypeError(
                "Interpolator for InterplatePlane must be the class of Nematics3D.classes.QInterpolator.QInterpolator"
            )
        self.act_bind_relation_base("interpolator", interpolator, is_weak=True)

        self._helper_commit()

    # InterpolatePlane adds `_helper_commit` as its internal recomputation step
    # for re-sampling interpolated values on the currently bound plane grid.
    @logging_and_warning_decorator()
    def _helper_commit(self, logger=None):

        plane_grid = self.grid

        grid_all = plane_grid._entity_grid_all
        grid_all_flatten = np.reshape(grid_all, (-1, 3))

        result = self.interpolator.interpolate(grid_all_flatten)
        object.__setattr__(self, "_calc_result", result[plane_grid._calc_box_mask])

    @property
    def result(self):
        return self._calc_result
