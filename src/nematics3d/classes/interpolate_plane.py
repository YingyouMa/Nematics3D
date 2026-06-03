"""Plane-based interpolation results built on physical-space PlaneGrid samplers."""

from typing import Any, Mapping
import numpy as np


from .class_base import AttrDef, ClassBase
from .grid_field import GridInterpolator
from .plane_grid import OptsPlaneGrid, PlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar


# InterpolatePlane is a lightweight bridge between an interpolator and a
# plane-grid sampling object.
#
# Subclasses should preserve the binding contract between `grid` and `field`,
# keep `calc_result` synchronized with the current grid mask, and be careful
# when changing grid construction because this class currently accepts either
# Cartesian physical-basis PlaneGrid instances or polar plane-grid
# implementations.
class InterpolatePlane(ClassBase):
    """
    InterpolatePlane samples a `GridInterpolator` on a plane grid.

    Normal users pass in an interpolator plus either an existing plane grid
    or plane-grid options. The sampled values are then available through
    `plane.result`. For Cartesian `PlaneGrid`, the sample coordinates come
    directly from the physical-space `origin`, `normal`, `axis1`, and spacing
    settings of the bound grid. Use `plane.show_relations()` to inspect the
    bound grid and `plane.grid.show_modifiable_attrs()` to inspect grid
    settings.
    """

    __attr_defs__ = {
        "calc_result": AttrDef(
            doc="The interpolated physics values sampled on the current plane grid.",
            kind="calc",
        ),
        "grid": AttrDef(
            doc="The plane grid associated with this interpolated field.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "interpolator": AttrDef(
            doc="The grid interpolator object used to sample this plane.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "result": AttrDef(
            doc="Read-only: Alias of `calc_result`.",
            kind="property",
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property")
        and name not in ClassBase.__slots__
    )

    # ==================== OVERRIDE ====================
    # InterpolatePlane overrides ClassBase.__init__ because it must create or
    # update the plane grid binding before validating the interpolator and
    # computing the first sampled result.
    # ==================================================
    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "interpolate plane",
        grid: PlaneGrid | PlaneGridPolar | None = None,
        opts: OptsPlaneGrid | OptsPlaneGridPolar | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
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

        if not isinstance(interpolator, GridInterpolator):
            raise TypeError(
                "Interpolator for InterplatePlane must be an instance of "
                "nematics3d.classes.grid_field.grid_interpolator.GridInterpolator."
            )
        self.act_bind_relation_base("interpolator", interpolator, is_weak=True)

        self.act_refresh()

    # InterpolatePlane adds `_helper_commit` as its internal recomputation step
    # for re-sampling interpolated values on the currently bound plane grid.
    def act_refresh(self):
        """Re-sample the bound interpolator on the current plane grid mask."""
        plane_grid = self.grid

        grid_all = plane_grid.entity_grid_all
        grid_all_flatten = np.reshape(grid_all, (-1, 3))

        result = self.interpolator.interpolate(grid_all_flatten)
        object.__setattr__(self, "calc_result", result[plane_grid.calc_box_mask])

    @property
    def result(self):
        """Return the interpolated values sampled on the current plane grid."""
        return self.calc_result
