"""Bounds host objects and box-like input conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping, TypeAlias
import weakref

import numpy as np
import pyvista as pv

from ..core.class_base import AttrDef
from ..core.host_base import HostBase, OptsBase
from ..datatypes import (
    Number,
    Tensor,
    UNSET,
    Unset,
    Vect,
    as_axes,
    as_dimension_info,
    as_number,
    as_points,
    as_str,
    as_vector,
)
from ..geometry import (
    OBBFit,
    get_box_corners,
    rotation_matrix_from_vectors,
    select_points_in_box,
)
from ..grid import apply_linear_transform
from ..logging_decorator import logging_and_warning_decorator


_DEF_TOL = 1e-8


BoundsData: TypeAlias = (
    "Bounds | Vect(6) | Tensor((4, 3)) | Tensor((8, 3)) | pv.PolyData"
)


@dataclass(slots=True)
class _BoundsSubscriberEntry:
    host_ref: weakref.ReferenceType
    sync_name: str
    kind: str

    @property
    def host(self):
        return self.host_ref()


@dataclass(slots=True, repr=False)
class OptsBounds(OptsBase):
    """Geometry options for an orthogonal box in physical space."""

    origin: Vect(3) | Unset = UNSET
    axis1: Vect(3) | Unset = UNSET
    axis2: Vect(3) | None | Unset = UNSET
    length1: Number | Unset = UNSET
    length2: Number | None | Unset = UNSET
    length3: Number | None | Unset = UNSET
    alignment: Literal["min_corner", "center"] | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "origin": "Anchor point; its meaning is controlled by alignment.",
        "axis1": "First unit axis of the orthogonal box.",
        "axis2": "Optional second unit axis; inferred when omitted.",
        "length1": "Side length along axis1.",
        "length2": "Side length along axis2; defaults to length1.",
        "length3": "Side length along axis3; defaults to length1.",
        "alignment": "Whether origin is the minimum corner or the box center.",
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase.impl_validators,
        "origin": lambda v, d: as_vector(v, name=d, d=3),
        "axis1": lambda v, d: as_vector(v, name=d, d=3, is_normalized=True),
        "axis2": lambda v, d: (
            None if v is None else as_vector(v, name=d, d=3, is_normalized=True)
        ),
        "length1": lambda v, d: as_number(v, name=d, value_range=(1e-12, np.inf)),
        "length2": lambda v, d: (
            None if v is None else as_number(v, name=d, value_range=(1e-12, np.inf))
        ),
        "length3": lambda v, d: (
            None if v is None else as_number(v, name=d, value_range=(1e-12, np.inf))
        ),
        "alignment": lambda v, d: as_str(v, name=d, pool=("min_corner", "center")),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsBase.impl_defaults_frozen),
            "tag": "bounds options",
            "origin": (0.0, 0.0, 0.0),
            "axis1": (1.0, 0.0, 0.0),
            "axis2": None,
            "length2": None,
            "length3": None,
            "alignment": "min_corner",
        }
    )


class Bounds(HostBase):
    """Mutable orthogonal box host with synchronization and visualization hooks."""

    __attr_defs__ = {
        "entity_corners": AttrDef(
            doc="Current box corners as an (8, 3) array.", kind="entity"
        ),
        "entity_clip_geometry": AttrDef(
            doc="Lazily materialized PyVista clipping surface.", kind="entity"
        ),
        "entity_visuals": AttrDef(
            doc="Visualization subscriptions for this bounds.", kind="entity"
        ),
        "entity_subscribers": AttrDef(
            doc="Weak records for hosts synchronized to this bounds.", kind="entity"
        ),
        "calc_axis2": AttrDef(doc="Resolved second box axis.", kind="calc"),
        "calc_axis3": AttrDef(doc="Resolved third box axis.", kind="calc"),
        "corners": AttrDef(doc="Read-only alias of entity_corners.", kind="property"),
        "clip_geometry": AttrDef(doc="Read-only clipping PolyData.", kind="property"),
        "lengths": AttrDef(doc="Resolved side lengths.", kind="property"),
        "subscribers": AttrDef(doc="Live synchronized hosts.", kind="property"),
        "glyph_subscribers": AttrDef(doc="Live glyph subscribers.", kind="property"),
        "plane_grid_subscribers": AttrDef(
            doc="Live plane-grid subscribers.", kind="property"
        ),
    }

    __slots__ = (
        "entity_corners",
        "entity_clip_geometry",
        "entity_visuals",
        "entity_subscribers",
        "calc_axis2",
        "calc_axis3",
    )

    def __init__(
        self,
        name: str | None = None,
        name_replace: str = "bounds",
        opts: OptsBounds | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            OptsBounds,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs,
        )
        object.__setattr__(self, "entity_corners", None)
        object.__setattr__(self, "entity_clip_geometry", None)
        object.__setattr__(self, "entity_visuals", [])
        object.__setattr__(self, "entity_subscribers", [])
        object.__setattr__(self, "calc_axis2", None)
        object.__setattr__(self, "calc_axis3", None)
        if self.opts.length1 is UNSET:
            raise ValueError("Missing required variable 'length1' to generate bounds")
        self.opts.act_finalize(defaults=self.opts_defaults)
        self._helper_commit_apply_opts(is_reapply_opts=True)

    @logging_and_warning_decorator()
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):
        if not is_reapply_opts and not kwargs:
            return

        with self.opts.act_internal_update():
            for key, value in kwargs.items():
                setattr(self.opts, key, value)

        origin = self.opts.origin
        axis1 = self.opts.axis1
        axis2 = self.opts.axis2
        lengths = self.lengths
        length1, length2, length3 = lengths

        if axis2 is None:
            rotation = rotation_matrix_from_vectors((1, 0, 0), axis1)
            axis2 = rotation @ np.array([0.0, 1.0, 0.0])
        else:
            dot = float(axis1 @ axis2)
            if not np.isclose(dot, 0.0, atol=_DEF_TOL):
                old_axis2 = axis2.copy()
                axis2 = axis2 - dot * axis1
                norm = float(np.linalg.norm(axis2))
                if norm <= _DEF_TOL:
                    raise ValueError(
                        "Invalid geometry: axis2 is parallel or nearly parallel to axis1."
                    )
                axis2 /= norm
                logger.warning(
                    "axis2 was not perpendicular to axis1 and was projected onto "
                    f"the orthogonal plane: {old_axis2} -> {axis2}."
                )

        axis3 = np.cross(axis1, axis2)
        corners_local = get_box_corners(length1, length2, length3)
        if self.opts.alignment == "min_corner":
            origin_min_corner = origin
        elif self.opts.alignment == "center":
            origin_min_corner = origin - 0.5 * (
                length1 * axis1 + length2 * axis2 + length3 * axis3
            )
        else:
            raise ValueError(f"Unsupported alignment {self.opts.alignment!r}.")

        corners = (
            origin_min_corner
            + corners_local[:, [0]] * axis1
            + corners_local[:, [1]] * axis2
            + corners_local[:, [2]] * axis3
        )
        object.__setattr__(self, "calc_axis2", axis2)
        object.__setattr__(self, "calc_axis3", axis3)
        object.__setattr__(self, "entity_corners", corners)
        object.__setattr__(self, "entity_clip_geometry", None)

    @property
    def corners(self):
        return self.entity_corners

    @property
    def clip_geometry(self):
        clip_geometry = self.entity_clip_geometry
        if clip_geometry is None:
            faces = np.hstack(
                [
                    [4, 0, 2, 4, 1],
                    [4, 3, 5, 7, 6],
                    [4, 0, 1, 5, 3],
                    [4, 2, 6, 7, 4],
                    [4, 0, 3, 6, 2],
                    [4, 1, 4, 7, 5],
                ]
            )
            clip_geometry = (
                pv.PolyData(self.corners, faces)
                .triangulate()
                .clean()
                .compute_normals(
                    cell_normals=True,
                    point_normals=True,
                    consistent_normals=True,
                    auto_orient_normals=True,
                    inplace=False,
                )
            )
            object.__setattr__(self, "entity_clip_geometry", clip_geometry)
        return clip_geometry

    @property
    def lengths(self):
        length1 = float(self.opts.length1)
        length2 = length1 if self.opts.length2 is None else float(self.opts.length2)
        length3 = length1 if self.opts.length3 is None else float(self.opts.length3)
        return np.asarray([length1, length2, length3], dtype=float)

    def act_contains_points(self, points, *, atol=1e-9):
        _, mask = select_points_in_box(
            points, self.corners, is_return_mask=True, atol=atol
        )
        return mask

    def act_copy(self, name: str | None = None):
        opts_new = type(self.opts)(**self.opts.act_asdict())
        return type(self)(
            name=f"{self.name}_2" if name is None else name,
            opts=opts_new,
        )

    def _helper_prune_subscribers(self):
        alive = []
        for entry in self.entity_subscribers:
            if entry.host is None:
                self.act_detach_sync_task(entry.sync_name)
            else:
                alive.append(entry)
        if len(alive) != len(self.entity_subscribers):
            object.__setattr__(self, "entity_subscribers", alive)

    def _helper_find_subscriber(self, *, host=None, sync_name: str | None = None):
        for entry in self.entity_subscribers:
            if sync_name is not None and entry.sync_name == sync_name:
                return entry
            if host is not None and entry.host is host:
                return entry
        return None

    def act_register_subscriber(self, host, *, sync_name: str, kind: str):
        self._helper_prune_subscribers()
        old = self._helper_find_subscriber(host=host, sync_name=sync_name)
        if old is not None:
            return old
        entry = _BoundsSubscriberEntry(weakref.ref(host), sync_name, str(kind))
        self.entity_subscribers.append(entry)
        return entry

    def act_unregister_subscriber(self, *, host=None, sync_name: str | None = None):
        keep = []
        for entry in self.entity_subscribers:
            is_match = (sync_name is not None and entry.sync_name == sync_name) or (
                host is not None and entry.host is host
            )
            if is_match:
                self.act_detach_sync_task(entry.sync_name)
            else:
                keep.append(entry)
        if len(keep) != len(self.entity_subscribers):
            object.__setattr__(self, "entity_subscribers", keep)

    @property
    def subscribers(self):
        self._helper_prune_subscribers()
        return tuple(entry.host for entry in self.entity_subscribers if entry.host)

    @property
    def glyph_subscribers(self):
        self._helper_prune_subscribers()
        return tuple(
            entry.host
            for entry in self.entity_subscribers
            if entry.kind == "glyph" and entry.host is not None
        )

    @property
    def plane_grid_subscribers(self):
        self._helper_prune_subscribers()
        return tuple(
            entry.host
            for entry in self.entity_subscribers
            if entry.kind == "plane_grid" and entry.host is not None
        )

    def act_visualize(
        self,
        figure=None,
        opts=None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        name: str | None = None,
        category: str = "bounds",
        is_reset_camera: bool = False,
        is_replace: bool = False,
        **kwargs,
    ):
        from ..visual.bounds import visualize_bounds

        return visualize_bounds(
            self,
            figure=figure,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            name=name,
            category=category,
            is_reset_camera=is_reset_camera,
            is_replace=is_replace,
            **kwargs,
        )


def _normalize_box_edge(edge: np.ndarray, *, name: str) -> tuple[np.ndarray, float]:
    edge = np.asarray(edge, dtype=float)
    length = float(np.linalg.norm(edge))
    if length <= _DEF_TOL:
        raise ValueError(f"{name} has near-zero length and cannot define a box axis.")
    return edge / length, length


def _opts_bounds_from_corner_edges(
    origin: np.ndarray,
    edge1: np.ndarray,
    edge2: np.ndarray,
    edge3: np.ndarray,
    *,
    is_preserve_axis_order: bool = True,
) -> OptsBounds:
    axis1, length1 = _normalize_box_edge(edge1, name="edge1")
    axis2, length2 = _normalize_box_edge(edge2, name="edge2")
    axis3, length3 = _normalize_box_edge(edge3, name="edge3")
    if any(
        abs(float(a @ b)) > _DEF_TOL
        for a, b in ((axis1, axis2), (axis1, axis3), (axis2, axis3))
    ):
        raise ValueError(
            "The input edges do not form an orthogonal box. "
            "Please convert this geometry to BoundsGeneral instead."
        )
    if float(np.dot(np.cross(axis1, axis2), axis3)) < 0:
        if is_preserve_axis_order:
            raise ValueError(
                "The input box edges form a left-handed frame under the given axis order."
            )
        axis2, axis3 = axis3, axis2
        length2, length3 = length3, length2
    return OptsBounds(
        origin=origin,
        axis1=axis1,
        axis2=axis2,
        length1=length1,
        length2=length2,
        length3=length3,
        alignment="min_corner",
    )


def _opts_bounds_from_8_points(points: np.ndarray) -> OptsBounds:
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 3):
        raise ValueError(f"Expected (8, 3) points for a box, got {points.shape}.")
    for i, origin in enumerate(points):
        others = np.delete(points, i, axis=0)
        order = np.argsort(np.linalg.norm(others - origin, axis=1))
        for ia, index1 in enumerate(order):
            for index2 in order[ia + 1 :]:
                edge1 = others[index1] - origin
                edge2 = others[index2] - origin
                try:
                    axis1, _ = _normalize_box_edge(edge1, name="edge1")
                    axis2, _ = _normalize_box_edge(edge2, name="edge2")
                except ValueError:
                    continue
                if abs(float(axis1 @ axis2)) > _DEF_TOL:
                    continue
                axis3_ref = np.cross(axis1, axis2)
                axis3_ref /= np.linalg.norm(axis3_ref)
                for candidate in others:
                    edge3 = candidate - origin
                    try:
                        axis3, _ = _normalize_box_edge(edge3, name="edge3")
                    except ValueError:
                        continue
                    if abs(abs(float(axis3 @ axis3_ref)) - 1.0) > _DEF_TOL:
                        continue
                    expected = np.array(
                        [
                            origin,
                            origin + edge1,
                            origin + edge2,
                            origin + edge3,
                            origin + edge1 + edge2,
                            origin + edge1 + edge3,
                            origin + edge2 + edge3,
                            origin + edge1 + edge2 + edge3,
                        ]
                    )
                    remaining = points.copy()
                    matched = True
                    for point in expected:
                        distances = np.linalg.norm(remaining - point, axis=1)
                        index = int(np.argmin(distances))
                        if distances[index] > _DEF_TOL:
                            matched = False
                            break
                        remaining = np.delete(remaining, index, axis=0)
                    if matched:
                        return _opts_bounds_from_corner_edges(
                            origin,
                            edge1,
                            edge2,
                            edge3,
                            is_preserve_axis_order=False,
                        )
    raise ValueError(
        "The input 8-point geometry does not describe an orthogonal box. "
        "Please convert this geometry to BoundsGeneral instead."
    )


def as_bounds(input_data, name: str = "bounds") -> Bounds | None:
    """Convert supported box-like inputs to a :class:`Bounds` instance."""
    if input_data is None:
        return None
    if isinstance(input_data, Bounds):
        return input_data
    if isinstance(input_data, pv.PolyData):
        surface = input_data.extract_surface().triangulate().clean()
        points = np.asarray(surface.points, dtype=float)
        if points.size == 0:
            raise ValueError("clip_geometry PolyData is empty.")
        rounded = np.round(points, decimals=10)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        points = points[np.sort(unique_idx)]
        if points.shape != (8, 3):
            raise ValueError(
                f"{name!r} PolyData does not look like a box: it has "
                f"{len(points)} unique points."
            )
        return Bounds(name=name, opts=_opts_bounds_from_8_points(points))

    arr = np.asarray(input_data, dtype=float)
    if arr.shape == (6,):
        xmin, xmax, ymin, ymax, zmin, zmax = arr.tolist()
        if not (xmax > xmin and ymax > ymin and zmax > zmin):
            raise ValueError(
                "Axis-aligned bounds must satisfy xmin<xmax, ymin<ymax, zmin<zmax."
            )
        return Bounds(
            name=name,
            opts=OptsBounds(
                origin=(xmin, ymin, zmin),
                axis1=(1.0, 0.0, 0.0),
                axis2=(0.0, 1.0, 0.0),
                length1=xmax - xmin,
                length2=ymax - ymin,
                length3=zmax - zmin,
            ),
        )
    if arr.shape == (4, 3):
        return Bounds(
            name=name,
            opts=_opts_bounds_from_corner_edges(
                arr[0], arr[1] - arr[0], arr[2] - arr[0], arr[3] - arr[0]
            ),
        )
    if arr.shape == (8, 3):
        return Bounds(name=name, opts=_opts_bounds_from_8_points(arr))
    raise TypeError(
        f"{name!r} could not be converted to Bounds. Supported inputs are None, "
        "Bounds, 6 axis-aligned limits, 4 box-defining points, 8 box corners, "
        "or box-like PyVista PolyData."
    )


def bounds_minimal_wrapping_points(
    points,
    axes,
    origin=None,
    name: str | None = "minimal bounds",
    min_lengths=None,
) -> Bounds:
    points = as_points(points, name="points", d=3, min_num=1)
    axes = as_axes(axes, name="axes", atol=_DEF_TOL)
    origin = (
        np.zeros(3, dtype=float)
        if origin is None
        else as_vector(origin, name="origin", d=3)
    )
    if min_lengths is None:
        min_lengths = np.full(3, _DEF_TOL, dtype=float)
    else:
        min_lengths = as_dimension_info(min_lengths, name="min_lengths").astype(float)
        if np.any(min_lengths <= 0):
            raise ValueError("`min_lengths` must contain only positive values.")
    local_points = (points - origin) @ axes
    local_min = np.min(local_points, axis=0)
    local_max = np.max(local_points, axis=0)
    local_center = 0.5 * (local_min + local_max)
    lengths = np.maximum(local_max - local_min, min_lengths)
    world_center = origin + axes @ local_center
    return Bounds(
        name=name,
        opts=OptsBounds(
            origin=world_center,
            axis1=axes[:, 0],
            axis2=axes[:, 1],
            length1=lengths[0],
            length2=lengths[1],
            length3=lengths[2],
            alignment="center",
        ),
    )


def bounds_expanded(
    bounds: Bounds,
    expand_factors,
    min_lengths=None,
    name: str | None = "expanded bounds",
) -> Bounds:
    if not isinstance(bounds, Bounds):
        raise TypeError("`bounds` must be a Bounds instance.")
    factors = np.asarray(expand_factors, dtype=float)
    if factors.shape != (3,) or np.any(factors <= 0):
        raise ValueError("`expand_factors` must be a positive shape-(3,) sequence.")
    if min_lengths is None:
        min_lengths = np.zeros(3, dtype=float)
    else:
        min_lengths = as_dimension_info(min_lengths, name="min_lengths").astype(float)
        if np.any(min_lengths < 0):
            raise ValueError("`min_lengths` cannot contain negative values.")
    lengths = np.maximum(bounds.lengths * factors, min_lengths)
    return Bounds(
        name=name,
        opts=OptsBounds(
            origin=bounds.opts.origin,
            axis1=bounds.opts.axis1,
            axis2=bounds.calc_axis2,
            length1=lengths[0],
            length2=lengths[1],
            length3=lengths[2],
            alignment=bounds.opts.alignment,
        ),
    )


def bounds_sample_points(
    bounds: Bounds,
    spacing=1.0,
    *,
    is_return_local: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if not isinstance(bounds, Bounds):
        raise TypeError("`bounds` must be a Bounds instance.")
    spacing = as_dimension_info(spacing, name="spacing").astype(float)
    if np.any(spacing <= 0):
        raise ValueError("`spacing` must contain only positive values.")
    axes = np.column_stack([bounds.opts.axis1, bounds.calc_axis2, bounds.calc_axis3])
    lengths = bounds.lengths
    if bounds.opts.alignment == "center":
        center = bounds.opts.origin
    else:
        center = bounds.opts.origin + 0.5 * (axes @ lengths)
    local_axes = [
        np.linspace(-0.5 * length, 0.5 * length, max(2, int(length // step) + 1))
        for length, step in zip(lengths, spacing)
    ]
    mesh = np.meshgrid(*local_axes, indexing="ij")
    local_points = np.column_stack([item.ravel() for item in mesh])
    points = apply_linear_transform(local_points, transform=axes.T, offset=center)
    return (points, local_points) if is_return_local else points


def obb_bounds_from_fit(fit: OBBFit, name: str | None = "seed bounds") -> Bounds:
    if not isinstance(fit, OBBFit):
        raise TypeError("`fit` must be an OBBFit returned by an OBB fitting helper.")
    return Bounds(
        name=name,
        opts=OptsBounds(
            origin=fit.center,
            axis1=fit.axes[:, 0],
            axis2=fit.axes[:, 1],
            length1=fit.lengths[0],
            length2=fit.lengths[1],
            length3=fit.lengths[2],
            alignment="center",
        ),
    )


__all__ = [
    "Bounds",
    "BoundsData",
    "OptsBounds",
    "as_bounds",
    "bounds_expanded",
    "bounds_minimal_wrapping_points",
    "bounds_sample_points",
    "obb_bounds_from_fit",
]
