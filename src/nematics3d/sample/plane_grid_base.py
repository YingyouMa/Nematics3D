"""Shared lifecycle for physical-space plane sampling grids."""

from __future__ import annotations

import numpy as np

from nematics3d.geometry import select_points_in_box
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..analysis.bounds import as_bounds
from ..core.class_base import AttrDef
from ..core.host_base import HostBase


class PlaneGridBase(HostBase):
    """Common bounds, refresh, and array behavior for plane-grid samplers."""

    __attr_defs__ = {
        "entity_grid": AttrDef(
            doc="Selected physical-space grid points after optional bounds filtering.",
            kind="entity",
        ),
        "entity_grid_all": AttrDef(
            doc="Complete physical-space grid points before optional bounds filtering.",
            kind="entity",
        ),
        "calc_box_mask": AttrDef(
            doc="Boolean mask selecting points kept after optional bounds filtering.",
            kind="calc",
        ),
        "impl_name_bounds_sync": AttrDef(
            doc="Internal sync-task name used to react to bounds geometry updates.",
            kind="impl",
        ),
        "impl_is_bounds_enabled": AttrDef(
            doc="Internal runtime switch controlling whether bound Bounds is applied.",
            kind="impl",
        ),
        "impl_is_warn_orthogonal": AttrDef(
            doc="Internal switch controlling automatic-axis orthogonalization warnings.",
            kind="impl",
        ),
        "field": AttrDef(
            doc="Interpolated field object attached to this plane grid.",
            kind="relation",
            is_weak_by_default=True,
        ),
        "bounds": AttrDef(
            doc="Bounds instance limiting this plane grid.",
            kind="relation",
            is_weak_by_default=True,
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in HostBase.__slots__
    )

    def _helper_init_plane_grid_runtime(self, *, bounds, sync_prefix: str) -> None:
        object.__setattr__(self, "impl_name_bounds_sync", f"{sync_prefix}::{id(self)}")
        object.__setattr__(self, "impl_is_bounds_enabled", True)
        object.__setattr__(self, "impl_is_warn_orthogonal", True)
        self.act_bind_bounds(bounds, is_apply=False)

    def _helper_filter_points_by_bounds(self, points, *, is_clip_inside: bool):
        points = np.asarray(points, dtype=float)
        bounds = self.bounds if self.impl_is_bounds_enabled else None
        if bounds is None:
            mask = np.ones(len(points), dtype=bool)
            return points, mask
        _, mask_inside = select_points_in_box(
            points, bounds.corners, is_return_mask=True
        )
        mask = mask_inside if is_clip_inside else ~mask_inside
        return points[mask], mask

    def _helper_refresh_bound_field(self) -> None:
        if self.field:
            self.field.act_refresh()

    def __iter__(self):
        return iter(self.entity_grid)

    def __getitem__(self, idx):
        return self.entity_grid[idx]

    def __array__(self, dtype=None):
        arr = self.entity_grid
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr

    def __call__(self):
        return self.entity_grid

    def act_copy(self, name: str | None = None, is_bind_same_bounds: bool = True):
        opts_new = type(self.opts)(**self.opts.act_asdict())
        bounds_new = self.bounds if is_bind_same_bounds else None
        name_new = self.name if name is None else name
        return type(self)(name=name_new, opts=opts_new, bounds=bounds_new)

    def act_unbind_bounds(self, is_apply=True):
        bounds_old = self.bounds
        if bounds_old is None:
            return
        bounds_old.act_unregister_subscriber(
            sync_name=self.impl_name_bounds_sync, host=self
        )
        self.act_unbind_relation_base("bounds")
        if is_apply:
            self.act_commit(is_reapply_opts=True)

    def act_bounds_enable(self):
        object.__setattr__(self, "impl_is_bounds_enabled", True)
        self.act_commit(is_reapply_opts=True)

    def act_bounds_disable(self):
        object.__setattr__(self, "impl_is_bounds_enabled", False)
        self.act_commit(is_reapply_opts=True)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_bind_bounds(self, bounds, is_apply=True, is_replace=True, logger=None):
        if bounds is None:
            self.act_unbind_bounds(is_apply=is_apply)
            return
        try:
            bounds = as_bounds(bounds, name="The bounds limiting this plane grid")
        except (TypeError, ValueError, AttributeError, KeyError):
            logger.exception("Check input.")
            logger.recovery(
                "Ignore this bounds input and continue without modifying the current binding."
            )
            return
        bounds_old = self.bounds
        if bounds_old is bounds:
            if is_apply:
                self.act_commit(is_reapply_opts=True)
            return
        if bounds_old is not None:
            if not is_replace:
                raise RuntimeError(
                    "This plane grid is already bound to a Bounds object."
                )
            self.act_unbind_bounds(is_apply=False)
        self.act_bind_relation_base("bounds", bounds, is_weak=True)
        bounds.act_attach_sync_task(
            self.impl_name_bounds_sync,
            lambda **kwargs: self.act_commit(is_reapply_opts=True),
        )
        bounds.act_register_subscriber(
            self,
            sync_name=self.impl_name_bounds_sync,
            kind="plane_grid",
        )
        if is_apply:
            self.act_commit(is_reapply_opts=True)


__all__ = ["PlaneGridBase"]
