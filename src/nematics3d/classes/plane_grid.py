"""Plane sampling grids embedded in 3D space with optional bounds filtering."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping

import numpy as np

from nematics3d.datatypes import (
    Number,
    Tensor,
    UNSET,
    Unset,
    Vect,
    as_Number,
    as_Vect,
    as_bool,
    as_str,
)
from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    apply_linear_transform,
    as_grid_transform,
    generate_fixed_step_grid,
)
from nematics3d.logging_decorator import logging_and_warning_decorator

from .bounds import Bounds, as_bounds
from .host_base import HostBase, OptsBase
from .opts import cover_value
from ..general import rotation_matrix_from_vectors, select_grid_in_box


#!!! grid unit
#!!! asdict
#!!! axis normal figdemo


# --- Plane Options ---
@dataclass(slots=True, repr=False)
class OptsPlaneGrid(OptsBase):
    """Options object controlling the geometry and filtering of a PlaneGrid."""

    normal: Vect(3) | Unset = UNSET
    spacing: Number | Unset = UNSET
    spacing_extra: Number | Unset = UNSET
    size: Number | Unset = UNSET
    size_extra: Number | Unset = UNSET
    origin: Vect(3) | Unset = UNSET
    alignment: Literal["center", "bottom-left"] | Unset = UNSET
    axis1: Vect(3) | None | Unset = UNSET
    is_clip_inside: bool | Unset = UNSET
    grid_offset: Vect(3) | None | Unset = UNSET
    grid_transform: Tensor((3, 3)) | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsBase.__attrs__),
        "normal": "normal of plane",
        "spacing": "grid spacing along axis1",
        "spacing_extra": "grid spacing along axis2",
        "size": "size of plane",
        "size_extra": "size of plane along axis2",
        "origin": "origin of plane",
        "alignment": (
            "Grid reference point to be placed at 'origin' "
            "('center' for geometric middle, 'bottom-left' for the first grid point [0,0])"
        ),
        "axis1": "first in-plane axis",
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
        "normal": lambda v, d: as_Vect(v, name=d, is_norm=True),
        "spacing": lambda v, d: as_Number(v, name=d),
        "spacing_extra": lambda v, d: None if v is None else as_Number(v, name=d),
        "size": lambda v, d: as_Number(v, name=d),
        "size_extra": lambda v, d: None if v is None else as_Number(v, name=d),
        "origin": lambda v, d: as_Vect(v, name=d),
        "alignment": lambda v, d: as_str(
            v,
            name=d,
            pool=("center", "bottom-left"),
        ),
        "axis1": lambda v, d: None if v is None else as_Vect(v, name=d, is_norm=True),
        "is_clip_inside": lambda v, d: as_bool(v, name=d),
        "grid_offset": lambda v, d: None if v is None else as_Vect(v, name=d),
        "grid_transform": lambda v, d: as_grid_transform(v, name=d),
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
            "grid_offset": None,
            "grid_transform": GRID_TRANSFORM_IDENTITY,
        }
    )


# PlaneGrid keeps the HostBase option pipeline but specializes it for
# generating a 2D lattice embedded in 3D space with optional bounds clipping.
#
# Subclasses should preserve the relationship among `normal`, `axis1`,
# the derived in-plane axis, and the generated grid caches. If grid
# generation is overridden, keep `entity_grid`, `entity_grid_all`,
# `entity_grid_int`, and the derived size/offset fields synchronized.
class PlaneGrid(HostBase):
    """
    PlaneGrid generates a 2D sampling grid embedded in 3D space.

    Normal users configure the grid through `grid.opts` or
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
                "Selected 3D grid points after applying transforms and optional "
                "bounding-box filtering (array of shape N x 3)."
            ),
        },
        "entity_grid_all": {
            "doc": (
                "Complete 3D grid points before filtering, reshaped as "
                "(num1 x num2 x 3)."
            ),
        },
        "entity_grid_int": {
            "doc": (
                "Integer lattice indices corresponding to 2D grid positions "
                "(num1 x num2 x 3)."
            ),
        },
        "calc_axis2": {
            "doc": "The second in-plane axis perpendicular to both axis1 and normal.",
        },
        "calc_offset_real": {
            "doc": (
                "Base 3D offset that maps 2D array indices [i, j] into plane "
                "coordinates before the global grid transform."
            ),
        },
        "calc_box_mask": {
            "doc": "Boolean mask selecting the grid points kept after optional bounds filtering.",
        },
        "calc_size": {
            "doc": "The actual size calculated from opts.size.",
        },
        "calc_size_extra": {
            "doc": "The actual secondary size calculated from opts.size and opts.size_extra.",
        },
        "entity_fig_demo": {
            "doc": (
                "Diagnostic plot showing the generated 2D grid points, axes, "
                "and normal vector for verification."
            ),
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
            "doc": "The interpolated field object attached to this plane grid.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "bounds": {
            "doc": "The Bounds instance limiting this plane grid.",
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

    # ==================== OVERRIDE ====================
    # PlaneGrid overrides HostBase.__init__ because it must validate required
    # plane parameters, install the bounds-sync helper state, and trigger the
    # first grid generation immediately after opts finalization.
    # ==================================================
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

        object.__setattr__(self, "entity_fig_demo", None)
        object.__setattr__(
            self, "impl_name_bounds_sync", f"plane_grid_bounds::{id(self)}"
        )
        object.__setattr__(self, "impl_is_bounds_enabled", True)
        object.__setattr__(self, "impl_is_warn_orthogonal", True)

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
        self.act_bind_bounds(bounds, is_apply=False)

        self._helper_commit_apply_opts(is_reapply_opts=True)

    # ==================== OVERRIDE ====================
    # PlaneGrid overrides HostBase._helper_commit_apply_opts_main because
    # plane-grid opts require custom axis construction, grid generation,
    # optional bounds filtering, and cache updates specific to plane sampling.
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

        logger.debug("Start to generate a new 2D grid.")

        space1 = self.opts.spacing
        space2 = space1 if self.opts.spacing_extra is None else self.opts.spacing_extra
        size1 = self.opts.size
        size2 = size1 if self.opts.size_extra is None else self.opts.size_extra
        origin = self.opts.origin
        normal = self.opts.normal
        axis1 = self.opts.axis1
        grid_transform = self.opts.grid_transform
        grid_offset = self.opts.grid_offset
        alignment = self.opts.alignment
        is_clip_inside = self.opts.is_clip_inside

        if axis1 is not None:
            dot_product = normal @ axis1
            if not np.isclose(dot_product, 0, atol=1e-8):
                old_axis1 = axis1.copy()
                axis1 = axis1 - dot_product * normal
                axis1 /= np.linalg.norm(axis1)
                if self.impl_is_warn_orthogonal:
                    logger.warning(
                        f"Invalid geometry: axis1 is not perpendicular to normal "
                        f"(dot product: {dot_product:.4e}). Projecting original "
                        f"axis1 {old_axis1} onto the plane defined by normal "
                        f"{normal}. New orthonormal axis1: {axis1}."
                    )
        else:
            rotation_matrix = rotation_matrix_from_vectors((0, 0, 1), normal)
            axis1 = rotation_matrix @ np.array([1, 0, 0])
            logger.debug(
                f"axis1 not provided. Automatically generated a reference axis1 "
                f"{axis1} perpendicular to normal {normal}."
            )

        axis2 = np.cross(normal, axis1)
        logger.debug(f"axis2={axis2}")

        grid, grid_int, sizes = generate_fixed_step_grid(
            size1, size2, space1, space2, alignment=alignment
        )
        size1, size2 = sizes
        target_shape = np.shape(grid)[:2]
        grid_int = np.reshape(grid_int, (-1, 2))

        step_both = np.array([axis1 * space1, axis2 * space2])
        index_origin_shift = np.zeros(2, dtype=float)
        if alignment == "center":
            index_origin_shift = 0.5 * (np.asarray(target_shape, dtype=float) - 1.0)

        offset = origin - np.einsum("i, ib -> b", index_origin_shift, step_both)
        grid = np.einsum("ai, ib -> ab", grid_int, step_both) + offset

        grid = apply_linear_transform(
            grid, transform=grid_transform, offset=grid_offset
        )

        bounds = self.bounds if self.impl_is_bounds_enabled else None
        if bounds is None:
            logger.debug("No bounds filtering applied to this plane grid.")
            mask = np.ones(len(grid), dtype=bool)
            grid_select = grid
        else:
            logger.debug(
                f"Select the grids against bounds {bounds!r}, keep inside={is_clip_inside}."
            )
            _, mask_inside = select_grid_in_box(
                grid, bounds.corners, is_return_mask=True
            )
            mask = mask_inside if is_clip_inside else ~mask_inside
            grid_select = grid[mask]

        object.__setattr__(self, "entity_grid", grid_select)
        object.__setattr__(
            self, "entity_grid_all", np.reshape(grid, (*target_shape, 3))
        )
        object.__setattr__(self, "entity_grid_int", grid_int)
        object.__setattr__(self, "calc_offset_real", offset)
        object.__setattr__(self, "calc_axis2", axis2)
        object.__setattr__(self, "calc_box_mask", mask)
        object.__setattr__(self, "calc_size", size1)
        object.__setattr__(self, "calc_size_extra", size2)
        object.__setattr__(self.opts, "axis1", axis1)

        if self.field:
            self.field.act_refresh()

    # ==================== OVERRIDE ====================
    # PlaneGrid overrides ClassBase.__repr__ because a plane grid is more useful
    # when represented by its geometric orientation and origin than by name
    # alone.
    # ==================================================
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = (
            f"{cls_name}, with normal={self.opts.normal}, axis1={self.opts.axis1}, "
            f"origin={self.opts.origin} at {self.opts.alignment}"
        )
        return msg

    # ==================== OVERRIDE ====================
    # PlaneGrid overrides ClassBase.__str__ to keep the plain string form
    # short and aligned with the repository-wide default identity style.
    # ==================================================
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"

    def __iter__(self):
        """Iterate over the currently selected grid points."""
        return iter(self.entity_grid)

    def __getitem__(self, idx):
        """Return one selected grid point or slice."""
        return self.entity_grid[idx]

    def __array__(self, dtype=None):
        """Expose the selected grid points as a NumPy array."""
        arr = self.entity_grid
        return np.asarray(arr, dtype=dtype) if dtype is not None else arr

    def __call__(self):
        """Return the currently selected grid points."""
        return self.entity_grid

    def act_copy(self, name: str | None = None, is_bind_same_bounds: bool = True):
        """Create one copied PlaneGrid with duplicated opts and optional shared bounds."""
        opts_new = type(self.opts)(**self.opts.act_asdict())
        bounds_new = self.bounds if is_bind_same_bounds else None
        name_new = self.name if name is None else name
        return type(self)(name=name_new, opts=opts_new, bounds=bounds_new)

    def act_unbind_bounds(self, is_apply=True):
        """Detach the current bounds object and optionally rebuild the grid."""
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
        """Enable the effect of the currently bound bounds without unbinding it."""
        object.__setattr__(self, "impl_is_bounds_enabled", True)
        self.act_commit(is_reapply_opts=True)

    def act_bounds_disable(self):
        """Disable the effect of the currently bound bounds without unbinding it."""
        object.__setattr__(self, "impl_is_bounds_enabled", False)
        self.act_commit(is_reapply_opts=True)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_bind_bounds(self, bounds, is_apply=True, is_replace=True, logger=None):
        """Bind one bounds object to this plane grid and optionally rebuild it."""
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


#     def act_debug_plot(self,
#                        opts_extent: OptsTube | None = None,
#                        opts_points: OptsSphere | None = None,
#                        opts_figure: OptsFigure | None = None,
#                        opts_origin: OptsSphere | None = None,
#                        **kwargs
#                        ):
#
#         if opts_extent is None:
#             opts_extent = OptsTube()
#         if opts_points is None:
#             opts_points = OptsSphere()
#         if opts_figure is None:
#             opts_figure = OptsFigure()
#         if opts_origin is None:
#             opts_origin = OptsSphere()
#
#         merge = merge_opts_all(
#             {
#                 "figure_": opts_figure,
#                 "point_": opts_points,
#                 "extent_": opts_extent,
#                 "origin_": opts_origin
#             },
#             kwargs, type(self).__name__)
#
#         opts_figure = merge["figure_"]
#         opts_points = merge["point_"]
#         opts_extent = merge["extent_"]
#         opts_origin = merge["origin_"]
#
#         figure = PlotFigure(
#             opts=opts_figure,
#             name=f"Diagnostic plot of plane {self.name!r}"
#         )
#         bulk = PlotSphere(
#             coords=self.entity_grid,
#             opts=opts_points,
#             figure=figure,
#             category="plane_grid_test",
#             name="grid"
#             )
#         PlotSphere(
#             coords=self.opts.origin,
#             opts=opts_origin,
#             figure=figure,
#             opts_defaults_override={
#                 "color": (1,0,0),
#                 "radius": 1.2*bulk._calc_radius[0]
#             },
#             category="plane_grid_test",
#             name="origin"
#         )
#         if self.bounds is not None:
#             self.bounds.act_visualize(
#                 opts=opts_extent,
#                 figure=figure,
#                 category="plane_grid_test",
#                 name="grid_extent",
#             )
#
#         object.__setattr__(self, "entity_fig_demo", figure)
#
#         return figure
#
#
# @logging_and_warning_decorator()
# def act_log_parameters(self, is_return: bool = False, logger=None) -> None:
#     """
#     Log internal filter and output parameters for inspection.

#     This is the standard logging interface used in this library, which
#     can be redirected to console or to a file depending on the logger
#     configuration and the behavior of ``logging_and_warning_decorator``.

#     All attributes listed in ``__attrs__`` are included,
#     formatted in a single log entry with a clear separator.
#     """
#     lines = []
#     lines.append("-------------- PlaneGrid Parameters --------------")

#     lines.append("PlaneGrid parameters and results:")
#     for attr in self.__slots__:
#         desc = self.__attrs__.get(attr, "(no description)")
#         value = getattr(self, attr, None)

#         if attr in ("opts.axis1", "opts.spacing", "opts.spacing_extra"):
#             lines.append(f"  {attr}: {value!r}  # {desc} (derived final value)")
#         else:
#             lines.append(f"  {attr}: {value!r}  # {desc}")

#     lines.append("-----------------------------------------------------")

#     msg = "\n".join(lines)

#     if is_return:
#         return msg
#     else:
#         logger.info(msg)

# def act_save(self, path: str = "save/PlaneGrid.json") -> None:
#     import json
#     import os
#     data = asdict(self._opts_all)
#     path = as_str(path, name="The path to save PlaneGrid")
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)

# @classmethod
# def act_load(cls, path: str = "save/PlaneGrid.json") -> "PlaneGrid":
#     import json
#     path = as_str(path, name="The path to load PlaneGrid")
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     opts = OptsPlaneGrid(**data)
#     return cls(opts=opts)
