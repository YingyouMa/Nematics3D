"""Cartesian plane sampling grids defined in physical space."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping

import numpy as np

from nematics3d.datatypes import (
    Number,
    UNSET,
    Unset,
    Vect,
    as_bool,
    as_number,
    as_str,
    as_vector,
)
from nematics3d.grid import generate_fixed_step_grid, resolve_plane_physical_axes
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..analysis.bounds import Bounds
from ..core.class_base import AttrDef
from ..core.host_base import OptsBase
from ..core.opts import cover_value
from .plane_grid_base import PlaneGridBase


@dataclass(slots=True, repr=False)
class OptsPlaneGrid(OptsBase):
    """Options controlling one Cartesian plane sampling grid."""

    normal: Vect(3) | Unset = UNSET
    spacing: Number | Unset = UNSET
    spacing_extra: Number | Unset = UNSET
    size: Number | Unset = UNSET
    size_extra: Number | Unset = UNSET
    origin: Vect(3) | Unset = UNSET
    alignment: Literal["center", "bottom-left"] | Unset = UNSET
    axis1: Vect(3) | None | Unset = UNSET
    is_clip_inside: bool | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsBase.__attrs__),
        "normal": "physical-space unit normal of the plane",
        "spacing": "physical step length along the physical in-plane axis1",
        "spacing_extra": "physical step length along the derived physical in-plane axis2",
        "size": "physical size of the plane along axis1",
        "size_extra": "physical size of the plane along axis2",
        "origin": "physical-space reference point on the plane",
        "alignment": "Interpretation of origin: center or bottom-left grid point",
        "axis1": "physical-space unit reference axis lying in the plane",
        "is_clip_inside": "Whether bounds filtering keeps points inside the bounds",
    }
    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsBase.impl_validators),
        "normal": lambda v, d: as_vector(v, name=d, is_normalized=True),
        "spacing": lambda v, d: as_number(v, name=d, value_range=(1e-12, np.inf)),
        "spacing_extra": lambda v, d: (
            None if v is None else as_number(v, name=d, value_range=(1e-12, np.inf))
        ),
        "size": lambda v, d: as_number(v, name=d),
        "size_extra": lambda v, d: None if v is None else as_number(v, name=d),
        "origin": lambda v, d: as_vector(v, name=d),
        "alignment": lambda v, d: as_str(v, name=d, pool=("center", "bottom-left")),
        "axis1": lambda v, d: (
            None if v is None else as_vector(v, name=d, is_normalized=True)
        ),
        "is_clip_inside": lambda v, d: as_bool(v, name=d),
    }
    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "tag": "plane grid options",
            "spacing_extra": None,
            "size_extra": None,
            "origin": (0, 0, 0),
            "alignment": "center",
            "axis1": None,
            "is_clip_inside": True,
        }
    )


class PlaneGrid(PlaneGridBase):
    """Generate a fixed-step Cartesian sampling grid embedded in a 3D plane.

    A future convenience layer may add a lightweight interactive preview for
    tuning plane origin, orientation, spacing, and size. That visualization
    concern is intentionally not part of this sampling-domain object for now.
    """

    __attr_defs__ = {
        "entity_grid_int": AttrDef(
            doc="Integer lattice indices describing the 2D sampling topology.",
            kind="entity",
        ),
        "calc_axis2": AttrDef(
            doc="Derived secondary physical in-plane unit axis.", kind="calc"
        ),
        "calc_origin_grid0": AttrDef(
            doc="Physical-space point corresponding to lattice index [0, 0].",
            kind="calc",
        ),
        "calc_size": AttrDef(doc="Effective realized size along axis1.", kind="calc"),
        "calc_size_extra": AttrDef(
            doc="Effective realized size along axis2.", kind="calc"
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
        name_replace: str = "2d grid",
        opts: OptsPlaneGrid | None = None,
        bounds: Bounds | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            OptsPlaneGrid,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs,
        )
        self._helper_init_plane_grid_runtime(
            bounds=bounds, sync_prefix="plane_grid_bounds"
        )
        for attr_name, value in {
            "normal": self.opts.normal,
            "spacing": self.opts.spacing,
            "size": self.opts.size,
        }.items():
            if value is UNSET:
                raise ValueError(
                    f"Missing required variable {attr_name!r} to generate plane_grid"
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
        space1 = self.opts.spacing
        space2 = space1 if self.opts.spacing_extra is None else self.opts.spacing_extra
        size1 = self.opts.size
        size2 = size1 if self.opts.size_extra is None else self.opts.size_extra
        axis1, axis2 = resolve_plane_physical_axes(
            self.opts.normal,
            self.opts.axis1,
            is_warn=self.impl_is_warn_orthogonal,
        )
        _, grid_int, sizes = generate_fixed_step_grid(
            size1, size2, space1, space2, alignment=self.opts.alignment
        )
        size1, size2 = sizes
        target_shape = np.shape(grid_int)[:2]
        grid, grid_int, offset = self._helper_grid_indices_to_physical_points(
            grid_int=grid_int,
            origin=self.opts.origin,
            axis1=axis1,
            axis2=axis2,
            spacing=space1,
            spacing_extra=space2,
            alignment=self.opts.alignment,
        )
        grid_select, mask = self._helper_filter_points_by_bounds(
            grid, is_clip_inside=self.opts.is_clip_inside
        )
        object.__setattr__(self, "entity_grid", grid_select)
        object.__setattr__(
            self, "entity_grid_all", np.reshape(grid, (*target_shape, 3))
        )
        object.__setattr__(self, "entity_grid_int", grid_int)
        object.__setattr__(self, "calc_origin_grid0", offset)
        object.__setattr__(self, "calc_axis2", axis2)
        object.__setattr__(self, "calc_box_mask", mask)
        object.__setattr__(self, "calc_size", size1)
        object.__setattr__(self, "calc_size_extra", size2)
        object.__setattr__(self.opts, "axis1", axis1)
        self._helper_refresh_bound_field()

    @staticmethod
    def _helper_get_alignment_index_shift(target_shape, alignment):
        if alignment == "center":
            return 0.5 * (np.asarray(target_shape, dtype=float) - 1.0)
        return np.zeros(2, dtype=float)

    def _helper_grid_indices_to_physical_points(
        self,
        grid_int,
        origin,
        axis1,
        axis2,
        spacing,
        spacing_extra,
        alignment,
    ):
        target_shape = np.shape(grid_int)[:2]
        grid_index_flat = np.reshape(grid_int, (-1, 2))
        step_both = np.array([axis1 * spacing, axis2 * spacing_extra])
        index_origin_shift = self._helper_get_alignment_index_shift(
            target_shape, alignment
        )
        offset = origin - np.einsum("i, ib -> b", index_origin_shift, step_both)
        grid_points = np.einsum("ai, ib -> ab", grid_index_flat, step_both) + offset
        return grid_points, grid_index_flat, offset

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}, with normal={self.opts.normal}, "
            f"axis1={self.opts.axis1}, origin={self.opts.origin} at {self.opts.alignment}"
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


__all__ = ["OptsPlaneGrid", "PlaneGrid"]
