"""Plane-based interpolation results built on top of PlaneGrid sampling objects."""

from typing import Any, ClassVar, Mapping

import numpy as np

from nematics3d.logging_decorator import logging_and_warning_decorator

from .class_base import ClassBase
from .plane_grid import OptsPlaneGrid, PlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar
from .q_interpolator import QInterpolator


# InterpolatePlane is a lightweight bridge between an interpolator and a
# plane-grid sampling object.
#
# Subclasses should preserve the binding contract between `grid` and `field`,
# keep `calc_result` synchronized with the current grid mask, and be careful
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

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(ClassBase.__attr_defs__),
        "raw_name": {
            **dict(ClassBase.__attr_defs__["raw_name"]),
            "doc": "The name identifier of this plane object.",
        },
        "calc_result": {
            "doc": "The interpolated physics values sampled on the current plane grid.",
            "kind": "calc",
        },
        "grid": {
            "doc": "The plane grid associated with this interpolated field.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "interpolator": {
            "doc": "The QInterpolator object used to sample this plane.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "result": {
            "doc": "Read-only: Alias of `calc_result`.",
            "kind": "property",
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in ClassBase.__slots__
    )

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
                "Interpolator for InterplatePlane must be an instance of "
                "nematics3d.classes.q_interpolator.QInterpolator."
            )
        self.act_bind_relation_base("interpolator", interpolator, is_weak=True)

        self._helper_commit()

    # InterpolatePlane adds `_helper_commit` as its internal recomputation step
    # for re-sampling interpolated values on the currently bound plane grid.
    @logging_and_warning_decorator()
    def _helper_commit(self, logger=None):
        plane_grid = self.grid

        grid_all = plane_grid.entity_grid_all
        grid_all_flatten = np.reshape(grid_all, (-1, 3))

        result = self.interpolator.interpolate(grid_all_flatten)
        object.__setattr__(self, "calc_result", result[plane_grid.calc_box_mask])

    @property
    def result(self):
        """Return the interpolated values sampled on the current plane grid."""
        return self.calc_result
