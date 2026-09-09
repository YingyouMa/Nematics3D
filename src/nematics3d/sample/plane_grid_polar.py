"""Polar plane sampling grids embedded in 3D physical space."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from nematics3d.datatypes import UNSET, Unset, Vect, as_bool, as_number, as_vector
from nematics3d.grid import resolve_plane_physical_axes
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..analysis.bounds import Bounds
from ..core.class_base import AttrDef
from ..core.host_base import OptsBase
from ..core.opts import cover_value
from .plane_grid_base import PlaneGridBase


@dataclass(slots=True, repr=False)
class OptsPlaneGridPolar(OptsBase):
    """Options for generating a polar point lattice directly in physical space."""

    origin: Vect(3) | Unset = UNSET
    normal: Vect(3) | Unset = UNSET
    theta0_axis: Vect(3) | None | Unset = UNSET
    r_min: float | Unset = UNSET
    layers: int | Unset = UNSET
    dr: float | Unset = UNSET
    arc_dist: float | Unset = UNSET
    is_clip_inside: bool | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsBase.__attrs__),
        "origin": "physical-space center of the polar grid",
        "normal": "physical-space unit normal of the plane",
        "theta0_axis": "physical-space in-plane reference axis defining theta=0",
        "r_min": "minimum physical radius of the first ring",
        "layers": "total number of rings/layers",
        "dr": "physical radial spacing between rings",
        "arc_dist": "target physical arc-length spacing along each ring",
        "is_clip_inside": "Whether bounds filtering keeps points inside the bounds",
    }
    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsBase.impl_validators),
        "origin": lambda v, d: as_vector(v, name=d),
        "normal": lambda v, d: as_vector(v, name=d, is_normalized=True),
        "theta0_axis": lambda v, d: (
            None if v is None else as_vector(v, name=d, is_normalized=True)
        ),
        "r_min": lambda v, d: (
            None if v is None else as_number(v, name=d, value_range=(0, np.inf))
        ),
        "layers": lambda v, d: as_number(
            v, name=d, value_range=(1, np.inf), is_integer=True
        ),
        "dr": lambda v, d: as_number(v, name=d, value_range=(1e-6, np.inf)),
        "arc_dist": lambda v, d: (
            None if v is None else as_number(v, name=d, value_range=(1e-6, np.inf))
        ),
        "is_clip_inside": lambda v, d: as_bool(v, name=d),
    }
    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "tag": "polar plane grid options",
            "theta0_axis": None,
            "r_min": None,
            "layers": 4,
            "dr": 0.5,
            "arc_dist": None,
            "is_clip_inside": True,
        }
    )


class PlaneGridPolar(PlaneGridBase):
    """Generate a polar sampling grid embedded in a 3D plane."""

    __attr_defs__ = {
        "entity_polar": AttrDef(
            doc="Polar coordinates (r, theta) of every point in the full grid.",
            kind="entity",
        ),
        "calc_ring_offsets": AttrDef(
            doc="Cumulative start/end offsets of each polar ring.", kind="calc"
        ),
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in PlaneGridBase.__slots__
    )

    def __init__(
        self,
        name: str | None = None,
        name_replace: str = "polar grid",
        opts: OptsPlaneGridPolar | None = None,
        bounds: Bounds | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            OptsPlaneGridPolar,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs,
        )
        self._helper_init_plane_grid_runtime(
            bounds=bounds, sync_prefix="plane_grid_polar_bounds"
        )
        for key, value in {
            "origin": self.opts.origin,
            "normal": self.opts.normal,
        }.items():
            if value is UNSET:
                raise ValueError(
                    f"Missing required variable {key!r} to generate polar plane grid"
                )
        self.opts.act_finalize(defaults=self.opts_defaults)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):
        if not is_reapply_opts and not kwargs:
            return
        with self.opts.act_internal_update():
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )
        arc_dist = self.opts.dr if self.opts.arc_dist is None else self.opts.arc_dist
        r_min = self.opts.dr if self.opts.r_min is None else self.opts.r_min
        theta0_axis, axis2 = resolve_plane_physical_axes(
            self.opts.normal,
            self.opts.theta0_axis,
            is_warn=self.impl_is_warn_orthogonal,
        )
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        points_list = []
        polar_list = []
        ring_sizes = []
        for i in range(self.opts.layers):
            r = r_min + i * self.opts.dr
            if np.isclose(r, 0):
                points_list.append(self.opts.origin.copy()[None, :])
                polar_list.append(np.array([[0.0, 0.0]]))
                ring_sizes.append(1)
                continue
            n_theta = max(1, int(np.round(2.0 * np.pi * r / arc_dist)))
            phi = (i * golden_angle) % (2.0 * np.pi)
            thetas = (2.0 * np.pi * np.arange(n_theta) / n_theta + phi) % (2.0 * np.pi)
            ring_points = (
                self.opts.origin
                + (r * np.cos(thetas))[:, None] * theta0_axis[None, :]
                + (r * np.sin(thetas))[:, None] * axis2[None, :]
            )
            points_list.append(ring_points)
            polar_list.append(np.column_stack([np.full(n_theta, r), thetas]))
            ring_sizes.append(n_theta)
        points = np.vstack(points_list)
        polar = np.vstack(polar_list)
        ring_offsets = np.empty(len(ring_sizes) + 1, dtype=np.int64)
        ring_offsets[0] = 0
        ring_offsets[1:] = np.cumsum(ring_sizes, dtype=np.int64)
        points_select, mask = self._helper_filter_points_by_bounds(
            points, is_clip_inside=self.opts.is_clip_inside
        )
        object.__setattr__(self, "entity_grid", points_select)
        object.__setattr__(self, "entity_grid_all", points)
        object.__setattr__(self, "entity_polar", polar)
        object.__setattr__(self, "calc_ring_offsets", ring_offsets)
        object.__setattr__(self, "calc_box_mask", mask)
        object.__setattr__(self.opts, "theta0_axis", theta0_axis)
        self._helper_refresh_bound_field()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}, with normal={self.opts.normal} "
            f"and origin={self.opts.origin}"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


__all__ = ["OptsPlaneGridPolar", "PlaneGridPolar"]
