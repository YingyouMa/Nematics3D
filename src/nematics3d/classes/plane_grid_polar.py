"""Polar plane sampling grids embedded in 3D space with optional bounds filtering."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from nematics3d.datatypes import (
    Tensor,
    UNSET,
    Unset,
    Vect,
    as_Number,
    as_Vect,
    as_bool,
)
from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    apply_linear_transform,
    as_grid_transform,
)
from nematics3d.general import rotation_matrix_from_vectors, select_grid_in_box
from nematics3d.logging_decorator import logging_and_warning_decorator

from .bounds import Bounds, as_bounds
from .host_base import HostBase, OptsBase
from .opts import cover_value


@dataclass(slots=True, repr=False)
class OptsPlaneGridPolar(OptsBase):
    """
    Options for generating a polar (concentric-ring) point lattice on a plane.

    This option set targets the "ring + equal arc-length" strategy:
    - Rings are placed at radii r_i = r_min + i * dr
    - Points on each ring are spaced by approximately constant arc length
      (via N_theta(i) ~ round(2*pi*r_i / arc_dist))
    - Rings are angularly staggered using the golden angle for reduced aliasing
      (a deterministic, reproducible staggering scheme)
    """

    origin: Vect(3) | Unset = UNSET
    normal: Vect(3) | Unset = UNSET
    theta0_axis: Vect(3) | None | Unset = UNSET
    r_min: float | Unset = UNSET
    layers: int | Unset = UNSET
    dr: float | Unset = UNSET
    arc_dist: float | Unset = UNSET
    is_clip_inside: bool | Unset = UNSET
    grid_offset: Vect(3) | None | Unset = UNSET
    grid_transform: Tensor((3, 3)) | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsBase.__attrs__),
        "origin": "center of the polar grid in index coordinates",
        "normal": "normal of the plane (unit vector)",
        "theta0_axis": (
            "in-plane reference axis defining theta=0; will be projected onto "
            "the plane and normalized (None uses the default axis)"
        ),
        "r_min": "minimum radius of the first ring (or 0 for center point)",
        "layers": "total number of rings/layers to generate",
        "dr": "radial spacing between rings; rings at r_i = r_min + i * dr",
        "arc_dist": (
            "target arc-length spacing between adjacent points along each ring"
        ),
        "is_clip_inside": (
            "Whether bounds filtering keeps the grid points inside the bounds "
            "(True) or outside (False)."
        ),
        "grid_offset": (
            "grid translation offset to map lattice indices to real-space coordinates"
        ),
        "grid_transform": (
            "grid transform matrix to map lattice indices to real-space coordinates "
            "(3x3 matrix)"
        ),
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsBase.impl_validators),
        "origin": lambda v, d: as_Vect(v, name=d),
        "normal": lambda v, d: as_Vect(v, name=d, is_norm=True),
        "theta0_axis": lambda v, d: (
            None if v is None else as_Vect(v, name=d, is_norm=True)
        ),
        "r_min": lambda v, d: (
            None if v is None else as_Number(v, name=d, value_range=(0, np.inf))
        ),
        "layers": lambda v, d: as_Number(
            v, name=d, value_range=(1, np.inf), is_int=True
        ),
        "dr": lambda v, d: as_Number(v, name=d, value_range=(1e-6, np.inf)),
        "arc_dist": lambda v, d: (
            None if v is None else as_Number(v, name=d, value_range=(1e-6, np.inf))
        ),
        "is_clip_inside": lambda v, d: as_bool(v, name=d),
        "grid_offset": lambda v, d: None if v is None else as_Vect(v, name=d),
        "grid_transform": lambda v, d: as_grid_transform(v, name=d),
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
            "grid_offset": None,
            "grid_transform": GRID_TRANSFORM_IDENTITY,
        }
    )


# PlaneGridPolar keeps the HostBase option pipeline but specializes it for
# generating concentric-ring sampling points on a plane with optional bounds
# clipping and diagnostic visualization.
#
# Subclasses should preserve the coupling among the polar point caches, ring
# offsets, and the theta reference axis. If the polar lattice generation is
# overridden, keep `entity_grid`, `entity_grid_all`, `entity_polar`, and
# `calc_ring_offsets` synchronized.
class PlaneGridPolar(HostBase):
    """
    PlaneGridPolar generates a polar sampling grid embedded in 3D space.

    Normal users configure the polar lattice through `grid.opts` or
    `grid.act_commit(...)`, and can iterate over the selected grid points or
    convert the object to a NumPy array directly. Use
    `grid.show_modifiable_attrs()` to inspect available settings and
    `grid.show_relations()` to check the current `field` and `bounds`
    bindings.
    """

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(HostBase.__attr_defs__),
        "entity_grid": {
            "doc": (
                "Selected 3D polar grid points after applying transforms and optional "
                "bounds filtering (array of shape N x 3)."
            ),
        },
        "entity_grid_all": {
            "doc": (
                "Complete 3D polar grid points before filtering, stored as an "
                "array of shape (N, 3)."
            ),
        },
        "entity_polar": {
            "doc": "The polar coordinates (r, theta) of every point in the full grid.",
        },
        "calc_ring_offsets": {
            "doc": "Cumulative offsets defining the start/end indices of each polar ring.",
        },
        "calc_box_mask": {
            "doc": "Boolean mask selecting the grid points kept after optional bounds filtering.",
        },
        "impl_name_bounds_sync": {
            "doc": "Internal sync-task name used to react to bounds geometry updates.",
        },
        "impl_is_bounds_enabled": {
            "doc": "Internal runtime switch controlling whether the bound bounds is applied.",
        },
        "impl_is_warn_orthogonal": {
            "doc": (
                "Internal runtime switch controlling whether automatic axis "
                "orthogonalization emits warnings."
            ),
        },
        "field": {
            "doc": "The interpolated field object attached to this polar plane grid.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "bounds": {
            "doc": "The Bounds instance limiting this polar plane grid.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in HostBase.__slots__
    )

    # PlaneGridPolar overrides HostBase.__init__ because it must validate
    # required polar-plane parameters, install the bounds-sync helper state,
    # and trigger the first polar-grid generation after opts finalization.
    # ==================================================
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

        object.__setattr__(
            self, "impl_name_bounds_sync", f"plane_grid_polar_bounds::{id(self)}"
        )
        object.__setattr__(self, "impl_is_bounds_enabled", True)
        object.__setattr__(self, "impl_is_warn_orthogonal", True)

        for key, value in {
            "origin": self.opts.origin,
            "normal": self.opts.normal,
        }.items():
            if value is UNSET:
                raise ValueError(
                    f"Missing required variable {key!r} to generate polar plane grid"
                )

        self.opts.act_finalize(defaults=self.opts_defaults)
        self.act_bind_bounds(bounds, is_apply=False)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    # ==================== OVERRIDE ====================
    # PlaneGridPolar overrides HostBase._helper_commit_apply_opts_main because
    # polar-grid opts require custom ring generation, theta-axis handling,
    # optional bounds filtering, and cache updates specific to polar sampling.
    # ==================================================
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
        origin = self.opts.origin
        dr = self.opts.dr
        normal = self.opts.normal
        theta0_axis = self.opts.theta0_axis
        layers = self.opts.layers

        if theta0_axis is not None:
            dot_product = normal @ theta0_axis
            if np.isclose(abs(dot_product), 1.0, atol=1e-8):
                old_theta0_axis = theta0_axis.copy()
                theta0_axis = None
                if self.impl_is_warn_orthogonal:
                    logger.warning(
                        f"Invalid geometry: theta0_axis is collinear with normal "
                        f"(dot product: {dot_product:.4e}). Ignore original "
                        f"theta0_axis {old_theta0_axis} and fall back to the "
                        f"automatic reference axis for normal {normal}."
                    )
            elif not np.isclose(dot_product, 0, atol=1e-8):
                old_theta0_axis = theta0_axis.copy()
                theta0_axis = theta0_axis - dot_product * normal
                theta0_axis /= np.linalg.norm(theta0_axis)
                if self.impl_is_warn_orthogonal:
                    logger.warning(
                        f"Invalid geometry: theta0_axis is not perpendicular to "
                        f"normal (dot product: {dot_product:.4e}). Projecting "
                        f"original theta0_axis {old_theta0_axis} onto the plane "
                        f"defined by normal {normal}. New orthonormal "
                        f"theta0_axis: {theta0_axis}."
                    )
        if theta0_axis is None:
            rotation_matrix = rotation_matrix_from_vectors((0, 0, 1), normal)
            theta0_axis = rotation_matrix @ np.array([1, 0, 0])
            logger.debug(
                f"theta0_axis not provided. Automatically generated a reference "
                f"theta0_axis {theta0_axis} perpendicular to normal {normal}."
            )

        e1 = theta0_axis
        e2 = np.cross(normal, e1)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        points_list = []
        polar_list = []
        ring_sizes = []

        for i in range(layers):
            r = r_min + i * dr

            if np.isclose(r, 0):
                points_list.append(origin.copy()[None, :])
                polar_list.append(np.array([[0.0, 0.0]]))
                ring_sizes.append(1)
                continue

            n_theta = int(np.round(2.0 * np.pi * r / arc_dist))
            n_theta = max(1, n_theta)

            phi = (i * golden_angle) % (2.0 * np.pi)
            thetas = (2.0 * np.pi * np.arange(n_theta) / n_theta + phi) % (2.0 * np.pi)

            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)
            ring_points = (
                origin
                + (r * cos_t)[:, None] * e1[None, :]
                + (r * sin_t)[:, None] * e2[None, :]
            )

            points_list.append(ring_points)
            polar_list.append(np.column_stack([np.full(n_theta, r), thetas]))
            ring_sizes.append(n_theta)

        points = np.vstack(points_list)
        polar = np.vstack(polar_list)

        ring_offsets = np.empty(len(ring_sizes) + 1, dtype=np.int64)
        ring_offsets[0] = 0
        ring_offsets[1:] = np.cumsum(ring_sizes, dtype=np.int64)

        points = apply_linear_transform(
            points, transform=self.opts.grid_transform, offset=self.opts.grid_offset
        )

        bounds = self.bounds if self.impl_is_bounds_enabled else None
        if bounds is None:
            points_select = points
            mask = np.ones(len(points), dtype=bool)
        else:
            _, mask_inside = select_grid_in_box(
                points, bounds.corners, is_return_mask=True
            )
            mask = mask_inside if self.opts.is_clip_inside else ~mask_inside
            points_select = points[mask]

        object.__setattr__(self, "entity_grid", points_select)
        object.__setattr__(self, "entity_grid_all", points)
        object.__setattr__(self, "entity_polar", polar)
        object.__setattr__(self, "calc_ring_offsets", ring_offsets)
        object.__setattr__(self, "calc_box_mask", mask)
        object.__setattr__(self.opts, "theta0_axis", theta0_axis)

        if self.field:
            self.field.act_refresh()

    # ==================== OVERRIDE ====================
    # PlaneGridPolar overrides ClassBase.__repr__ because a polar grid is more
    # useful when represented by its plane orientation and radial settings than
    # by name alone.
    # ==================================================
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}, with normal={self.opts.normal} and origin={self.opts.origin}"
        )

    def __iter__(self):
        """Iterate over the currently selected polar grid points."""
        return iter(self.entity_grid)

    def __getitem__(self, idx):
        """Return one selected polar grid point or slice."""
        return self.entity_grid[idx]

    # ==================== OVERRIDE ====================
    # PlaneGridPolar overrides ClassBase.__str__ to keep the plain string form
    # short and aligned with the repository-wide default identity style.
    # ==================================================
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def __array__(self, dtype=None):
        """Expose the selected polar grid points as a NumPy array."""
        arr = self.entity_grid
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr

    def __call__(self):
        """Return the currently selected polar grid points."""
        return self.entity_grid

    def act_copy(self, name: str | None = None, is_bind_same_bounds: bool = True):
        """Create one copied PlaneGridPolar with duplicated opts and optional shared bounds."""
        opts_new = type(self.opts)(**self.opts.act_asdict())
        bounds_new = self.bounds if is_bind_same_bounds else None
        name_new = self.name if name is None else name
        return type(self)(name=name_new, opts=opts_new, bounds=bounds_new)

    def act_unbind_bounds(self, is_apply=True):
        """Detach the current bounds object and optionally rebuild the polar grid."""
        bounds_old = self.bounds
        if bounds_old is None:
            return
        bounds_old.act_unregister_subscriber(
            sync_name=self.impl_name_bounds_sync,
            host=self,
        )
        self.act_unbind_relation_base("bounds")
        if is_apply:
            self.act_commit(is_reapply_opts=True)

    def act_bounds_enable(self):
        """Enable the effect of the currently bound bounds without unbinding it."""
        object.__setattr__(self, "impl_is_bounds_enabled", True)
        self.act_commit(is_reapply_opts=True)

    def act_bounds_disable(self):
        """Disable the effect of the currently bound bounds without unbinding it."""
        object.__setattr__(self, "impl_is_bounds_enabled", False)
        self.act_commit(is_reapply_opts=True)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_bind_bounds(self, bounds, is_apply=True, is_replace=True, logger=None):
        """Bind one bounds object to this polar plane grid and optionally rebuild it."""
        if bounds is None:
            self.act_unbind_bounds(is_apply=is_apply)
            return

        try:
            bounds = as_bounds(bounds, name="The bounds limiting this polar plane grid")
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
                    "This polar plane grid is already bound to a Bounds object."
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
