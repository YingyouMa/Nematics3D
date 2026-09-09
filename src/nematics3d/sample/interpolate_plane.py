"""Plane-based interpolation results built on physical-space plane samplers."""

from typing import Any, Mapping

import numpy as np

from ..grid.field import GridInterpolator
from ..core.class_base import AttrDef, ClassBase
from .plane_grid import OptsPlaneGrid, PlaneGrid
from .plane_grid_base import PlaneGridBase
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar


PlaneGridType = PlaneGrid | PlaneGridPolar
PlaneGridOptsType = OptsPlaneGrid | OptsPlaneGridPolar


class InterpolatePlane(ClassBase):
    """Sample a :class:`GridInterpolator` on a physical-space plane grid."""

    __attr_defs__ = {
        "calc_result_all": AttrDef(
            doc="Interpolated values on the complete plane grid before bounds filtering.",
            kind="calc",
        ),
        "calc_result": AttrDef(
            doc="Interpolated values sampled on the current plane grid.", kind="calc"
        ),
        "grid": AttrDef(
            doc="Plane sampling grid associated with this result.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "interpolator": AttrDef(
            doc="GridInterpolator used to sample this plane.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "result": AttrDef(doc="Read-only alias of calc_result.", kind="property"),
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property") and name not in ClassBase.__slots__
    )

    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "interpolate plane",
        grid: PlaneGridType | None = None,
        opts: PlaneGridOptsType | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(name=name, name_replace="interpolate plane")
        self._helper_validate_interpolator(interpolator)
        grid = self._helper_resolve_grid(
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )
        self.act_bind_relation_base("grid", grid, is_weak=False)
        grid.act_bind_relation_base("field", self, is_weak=True)
        self.act_bind_relation_base("interpolator", interpolator, is_weak=True)
        self.act_refresh()

    @staticmethod
    def _helper_validate_interpolator(interpolator) -> None:
        if not isinstance(interpolator, GridInterpolator):
            raise TypeError("`interpolator` must be a GridInterpolator instance.")

    def _helper_resolve_grid(
        self,
        *,
        grid: PlaneGridType | None,
        opts: PlaneGridOptsType | None,
        opts_defaults_override: Mapping[str, Any] | None,
        **kwargs,
    ) -> PlaneGridType:
        if grid is not None:
            if not isinstance(grid, PlaneGridBase):
                raise TypeError(
                    "`grid` must be a PlaneGrid or PlaneGridPolar instance."
                )
            grid_new = grid.act_copy(name=self.name + "-grid")
            if opts is not None:
                expected_opts_type = type(grid_new.opts)
                if not isinstance(opts, expected_opts_type):
                    raise TypeError(
                        f"`opts` must match the provided grid type ({expected_opts_type.__name__})."
                    )
            if opts is not None or kwargs:
                grid_new.act_commit(opts=opts, **kwargs)
            return grid_new

        if isinstance(opts, OptsPlaneGridPolar):
            grid_type = PlaneGridPolar
        elif opts is None or isinstance(opts, OptsPlaneGrid):
            grid_type = PlaneGrid
        else:
            raise TypeError(
                "`opts` must be an OptsPlaneGrid or OptsPlaneGridPolar instance."
            )
        return grid_type(
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            name=self.name + "-grid",
            **kwargs,
        )

    def act_refresh(self):
        """Re-sample the bound interpolator on the current plane-grid mask."""
        grid_all = np.reshape(self.grid.entity_grid_all, (-1, 3))
        sampled = self.interpolator.interpolate(grid_all)
        object.__setattr__(self, "calc_result_all", sampled)
        object.__setattr__(self, "calc_result", sampled[self.grid.calc_box_mask])

    @property
    def result(self):
        return self.calc_result


__all__ = ["InterpolatePlane", "PlaneGridOptsType", "PlaneGridType"]
