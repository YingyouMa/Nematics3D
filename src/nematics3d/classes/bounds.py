"""Axis-aligned and oriented bounding-box host utilities."""

from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import weakref
from typing import Any, ClassVar, Literal, Mapping, TypeAlias

import numpy as np
import pyvista as pv

from nematics3d.geometry import OBBFit
from nematics3d.datatypes import (
    Number,
    UNSET,
    Unset,
    Vect,
    as_Number,
    as_Vect,
    as_dimension_info,
    as_points,
    as_str,
)
from nematics3d.field import apply_linear_transform
from nematics3d.logging_decorator import logging_and_warning_decorator
from nematics3d.general import get_box_corners, rotation_matrix_from_vectors
from .host_base import HostBase, OptsBase

BoundsData: TypeAlias = (
    "Bounds | Vect(6) | Tensor((4, 3)) | Tensor((8, 3)) | pv.PolyData"
)


@dataclass(slots=True)
class _BoundsSubscriberEntry:
    """Weak-reference record for one bounds-driven sync subscriber."""

    host_ref: weakref.ReferenceType
    sync_name: str
    kind: str

    @property
    def host(self):
        """Return the live host object, or ``None`` if it was garbage-collected."""
        return self.host_ref()


@dataclass(slots=True)
class _BoundsVisualEntry:
    """Weak-reference record for one bounds visualization frame inside a figure."""

    figure_ref: weakref.ReferenceType
    tube_ref: weakref.ReferenceType
    sync_name: str

    @property
    def figure(self):
        """Return the live figure object, or ``None`` if it was garbage-collected."""
        return self.figure_ref()

    @property
    def tube(self):
        """Return the live tube visual, or ``None`` if it was garbage-collected."""
        return self.tube_ref()


@dataclass(slots=True, repr=False)
class OptsBounds(OptsBase):
    """Opts object controlling bounds geometry and origin/axis interpretation."""

    origin: Vect(3) | Unset = UNSET
    axis1: Vect(3) | Unset = UNSET
    axis2: Vect(3) | None | Unset = UNSET

    length1: Number | Unset = UNSET
    length2: Number | None | Unset = UNSET
    length3: Number | None | Unset = UNSET

    alignment: Literal["min_corner", "center"] | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "origin": (
            "The anchor point of the bounds box. "
            "Its geometric meaning is determined by ``alignment``."
        ),
        "axis1": (
            "The first direction of the bounds box. "
            "It is typically normalized by the host before use."
        ),
        "axis2": (
            "The second direction of the bounds box. "
            "The third direction is derived from the cross product of ``axis1`` and ``axis2``. "
            "If None, the host may infer a default orthogonal direction."
        ),
        "length1": "The box length along ``axis1``.",
        "length2": "The box length along ``axis2``.",
        "length3": (
            "The box length along the third direction derived from "
            "``axis1 x axis2``."
        ),
        "alignment": (
            "How ``origin`` is interpreted relative to the box. "
            'Typical values include ``"min_corner"`` and ``"center"``.'
        ),
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase.impl_validators,
        "origin": lambda v, d: as_Vect(v, name=d, dim=3),
        "axis1": lambda v, d: as_Vect(v, name=d, dim=3, is_norm=True),
        "axis2": lambda v, d: (
            None if v is None else as_Vect(v, name=d, dim=3, is_norm=True)
        ),
        "length1": lambda v, d: as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "length2": lambda v, d: (
            None if v is None else as_Number(v, name=d, value_range=(1e-12, np.inf))
        ),
        "length3": lambda v, d: (
            None if v is None else as_Number(v, name=d, value_range=(1e-12, np.inf))
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
    """Host object representing one axis-aligned or oriented box bounds."""

    # fmt: off
    __attr_defs__ = {
        **dict(HostBase.__attr_defs__),
        "entity_corners": {
            "doc": "Corner coordinates of the bounds box in real space as an (8, 3) array.",
        },
        "entity_clip_geometry": {
            "doc": "PyVista PolyData surface used for clipping other meshes inside this bounds.",
        },
        "entity_visuals": {
            "doc": "Visual subscriptions of this bounds across figures.",
        },
        "entity_subscribers": {
            "doc": (
                "Weak subscriber records for hosts driven by this bounds, "
                "excluding its own visualization frames."
            ),
        },
        "calc_axis2": {
            "doc": "Resolved second axis used by the bounds box.",
        },
        "calc_axis3": {
            "doc": "Resolved third axis used by the bounds box.",
        },
        "corners": {
            "doc":  "Read-only: Alias of `entity_corners`.",
            "kind": "property",
        },
        "clip_geometry": {
            "doc":  "Read-only: Alias of `entity_clip_geometry`.",
            "kind": "property",
        },
        "subscribers": {
            "doc":  "Read-only: Live hosts currently subscribed to this bounds.",
            "kind": "property",
        },
        "glyph_subscribers": {
            "doc":  "Read-only: Live glyph hosts currently subscribed to this bounds.",
            "kind": "property",
        },
        "plane_grid_subscribers": {
            "doc":  "Read-only: Live plane-grid hosts currently subscribed to this bounds.",
            "kind": "property",
        },
    }
    # fmt: on

    _VISUAL_EDGES = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (4, 7),
        (3, 5),
        (3, 6),
        (5, 7),
        (6, 7),
    )
    _VISUAL_DEFAULTS = MappingProxyType(
        {
            "color": (0.0, 0.0, 0.0),
            "radius": 0.35,
            "is_pickable": True,
        }
    )

    __slots__ = (
        "entity_corners",
        "entity_clip_geometry",
        "entity_visuals",
        "entity_subscribers",
        "calc_axis2",
        "calc_axis3",
    )

    # ==================== OVERRIDE ====================
    # Bounds overrides HostBase.__init__ because it must initialize its derived
    # geometry stores and trigger the first geometry build immediately.
    # ==================================================
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

    # ==================== OVERRIDE ====================
    # Bounds overrides HostBase._helper_commit_apply_opts_main so finalized opts
    # are translated into concrete box geometry and clipping entities.
    # ==================================================
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
        length1 = self.opts.length1
        length2 = length1 if self.opts.length2 is None else self.opts.length2
        length3 = length1 if self.opts.length3 is None else self.opts.length3
        alignment = self.opts.alignment

        if axis2 is not None:
            dot_product = axis1 @ axis2
            if not np.isclose(dot_product, 0, atol=1e-8):
                old_axis2 = axis2.copy()
                axis2 = axis2 - dot_product * axis1
                axis2 /= np.linalg.norm(axis2)
                logger.warning(
                    f"Invalid geometry: axis2 is not perpendicular to axis1 "
                    f"(dot product: {dot_product:.4e}). Projecting original "
                    f"axis2 {old_axis2} onto the plane normal to axis1 {axis1}. "
                    f"New orthonormal axis2: {axis2}."
                )
        else:
            rotation_matrix = rotation_matrix_from_vectors((1, 0, 0), axis1)
            axis2 = rotation_matrix @ np.array([0.0, 1.0, 0.0])
            logger.debug(
                f"axis2 not provided. Automatically generated a reference axis2 {axis2} "
                f"from axis1 {axis1}."
            )

        axis3 = np.cross(axis1, axis2)
        corners_local = get_box_corners(length1, length2, length3)

        if alignment == "min_corner":
            origin_min_corner = origin
        elif alignment == "center":
            origin_min_corner = origin - 0.5 * (
                length1 * axis1 + length2 * axis2 + length3 * axis3
            )
        else:
            raise ValueError(f"Unsupported alignment {alignment!r}.")

        corners = (
            origin_min_corner
            + corners_local[:, [0]] * axis1
            + corners_local[:, [1]] * axis2
            + corners_local[:, [2]] * axis3
        )

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
            pv.PolyData(corners, faces)
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

        object.__setattr__(self, "calc_axis2", axis2)
        object.__setattr__(self, "calc_axis3", axis3)
        object.__setattr__(self, "entity_corners", corners)
        object.__setattr__(self, "entity_clip_geometry", clip_geometry)

    @property
    def corners(self):
        """Return the current box corners as an ``(8, 3)`` array."""
        return self.entity_corners

    @property
    def clip_geometry(self):
        """Return the current clipping ``PolyData`` for this bounds."""
        return self.entity_clip_geometry

    def act_copy(self, name: str | None = None):
        """
        Create a detached copy of this bounds geometry.

        The copied bounds keeps the same geometric options, but does not carry
        over figure visuals or subscriber relations from the original bounds.
        """
        if name is None:
            name_new = f"{self.name}_2"
        else:
            name_new = name

        opts_new = type(self.opts)(**self.opts.act_asdict())
        return type(self)(name=name_new, opts=opts_new)

    def _helper_is_subscriber_alive(self, entry: _BoundsSubscriberEntry) -> bool:
        return entry.host is not None

    def _helper_prune_subscribers(self):
        subscribers_alive = []
        sync_to_detach = []
        for entry in self.entity_subscribers:
            if self._helper_is_subscriber_alive(entry):
                subscribers_alive.append(entry)
            else:
                sync_to_detach.append(entry.sync_name)

        for sync_name in sync_to_detach:
            self.act_detach_sync_task(sync_name)

        if len(subscribers_alive) != len(self.entity_subscribers):
            object.__setattr__(self, "entity_subscribers", subscribers_alive)

    def _helper_find_subscriber(self, *, host=None, sync_name: str | None = None):
        for entry in self.entity_subscribers:
            if sync_name is not None and entry.sync_name == sync_name:
                return entry
            if host is not None and entry.host is host:
                return entry
        return None

    def act_register_subscriber(self, host, *, sync_name: str, kind: str):
        """Register one host as a sync subscriber of this bounds."""
        self._helper_prune_subscribers()
        entry_old = self._helper_find_subscriber(host=host, sync_name=sync_name)
        if entry_old is not None:
            return entry_old

        entry = _BoundsSubscriberEntry(
            host_ref=weakref.ref(host),
            sync_name=sync_name,
            kind=str(kind),
        )
        self.entity_subscribers.append(entry)
        return entry

    def act_unregister_subscriber(self, *, host=None, sync_name: str | None = None):
        """Unregister one subscriber by host object or sync-task name."""
        subscribers_alive = []
        sync_to_detach = []
        for entry in self.entity_subscribers:
            is_match = (sync_name is not None and entry.sync_name == sync_name) or (
                host is not None and entry.host is host
            )
            if is_match:
                sync_to_detach.append(entry.sync_name)
            else:
                subscribers_alive.append(entry)

        for name in sync_to_detach:
            self.act_detach_sync_task(name)

        if sync_to_detach:
            object.__setattr__(self, "entity_subscribers", subscribers_alive)

    @property
    def subscribers(self):
        """Return all live hosts currently subscribed to this bounds."""
        self._helper_prune_subscribers()
        return tuple(
            entry.host for entry in self.entity_subscribers if entry.host is not None
        )

    @property
    def glyph_subscribers(self):
        """Return the live glyph subscribers of this bounds."""
        self._helper_prune_subscribers()
        return tuple(
            entry.host
            for entry in self.entity_subscribers
            if entry.kind == "glyph" and entry.host is not None
        )

    @property
    def plane_grid_subscribers(self):
        """Return the live plane-grid subscribers of this bounds."""
        self._helper_prune_subscribers()
        return tuple(
            entry.host
            for entry in self.entity_subscribers
            if entry.kind == "plane_grid" and entry.host is not None
        )

    def _helper_build_visual_edges(self) -> tuple[np.ndarray, np.ndarray]:
        coords = []

        line_index = []
        for i, (a, b) in enumerate(self._VISUAL_EDGES):
            coords.append(self.corners[a])
            coords.append(self.corners[b])
            line_index.extend([i, i])

        return np.asarray(coords, dtype=float), np.asarray(line_index, dtype=int)

    def _helper_is_visual_entry_alive(self, entry: _BoundsVisualEntry) -> bool:
        figure = entry.figure
        tube = entry.tube
        return (
            figure is not None
            and tube is not None
            and figure.is_alive
            and tube in figure.glyphs
        )

    def _helper_find_visual_entry(
        self,
        *,
        figure=None,
        tube=None,
        sync_name: str | None = None,
    ) -> _BoundsVisualEntry | None:
        for entry in self.entity_visuals:
            if sync_name is not None and entry.sync_name == sync_name:
                return entry
            if figure is not None and entry.figure is figure:
                return entry
            if tube is not None and entry.tube is tube:
                return entry
        return None

    def _helper_prune_visuals(self):
        visuals_alive = []
        sync_to_detach = []
        for entry in self.entity_visuals:
            if self._helper_is_visual_entry_alive(entry):
                visuals_alive.append(entry)
            else:
                sync_to_detach.append(entry.sync_name)

        for sync_name in sync_to_detach:
            self.act_detach_sync_task(sync_name)

        if len(visuals_alive) != len(self.entity_visuals):
            object.__setattr__(self, "entity_visuals", visuals_alive)

    def _helper_unregister_visual_sync(
        self, sync_name: str | None = None, *, tube=None
    ):
        visuals_alive = []
        sync_to_detach = []
        for entry in self.entity_visuals:
            is_match = (sync_name is not None and entry.sync_name == sync_name) or (
                tube is not None and entry.tube is tube
            )
            if is_match:
                sync_to_detach.append(entry.sync_name)
            else:
                visuals_alive.append(entry)

        for name in sync_to_detach:
            self.act_detach_sync_task(name)

        if sync_to_detach:
            object.__setattr__(self, "entity_visuals", visuals_alive)

    def _helper_refresh_visual(self, sync_name: str):
        entry = self._helper_find_visual_entry(sync_name=sync_name)
        if entry is None or not self._helper_is_visual_entry_alive(entry):
            self._helper_unregister_visual_sync(sync_name)
            return

        coords, line_index = self._helper_build_visual_edges()
        entry.tube.act_commit(
            coords=coords, line_index=line_index, is_reapply_opts=True
        )

    def _helper_open_interact_panels(self, tube, figure):
        from .visual.qt.interact_bounds import InteractBounds
        from .visual.qt.interact_tube import InteractTube

        InteractTube(tube, figure).show()
        InteractBounds(self, figure).show()

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
        """Visualize this bounds as a tube frame inside one figure."""
        from .visual.plot_figure import PlotFigure
        from .visual.plot_tube import PlotTube

        self._helper_prune_visuals()
        if figure is None:
            figure = PlotFigure()
        elif not isinstance(figure, PlotFigure):
            try:
                figure = PlotFigure(plotter=figure)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                figure = PlotFigure()

        entry_old = self._helper_find_visual_entry(figure=figure)
        if entry_old is not None:
            tube_old = entry_old.tube
            if tube_old is not None and self._helper_is_visual_entry_alive(entry_old):
                if not is_replace:
                    if opts is not None:
                        tube_old.act_commit(opts=opts, **kwargs)
                    elif kwargs:
                        tube_old.act_commit(**kwargs)
                    return tube_old
                tube_old.act_remove()

        coords, line_index = self._helper_build_visual_edges()
        if opts_defaults_override is None:
            opts_defaults_override = dict(self._VISUAL_DEFAULTS)
        else:
            opts_defaults_override = dict(self._VISUAL_DEFAULTS) | dict(
                opts_defaults_override
            )

        tube = PlotTube(
            coords=coords,
            line_index=line_index,
            figure=figure,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            name=self.name if name is None else name,
            category=category,
            is_reset_camera=is_reset_camera,
            **kwargs,
        )

        sync_name = f"{tube.impl_name_pv}__bounds_sync"
        tube.act_bind_relation_base(
            "bounds_visual_source",
            self,
            doc="Bounds source driving this visualized frame.",
            is_weak=True,
        )
        tube.act_set_interact_func(
            lambda: self._helper_open_interact_panels(tube=tube, figure=figure)
        )

        self.act_attach_sync_task(
            sync_name,
            lambda **kwargs_sync: self._helper_refresh_visual(sync_name),
        )
        self.entity_visuals.append(
            _BoundsVisualEntry(
                figure_ref=weakref.ref(figure),
                tube_ref=weakref.ref(tube),
                sync_name=sync_name,
            )
        )
        return tube


_DEF_TOL = 1e-8


def _normalize_box_edge(edge: np.ndarray, *, name: str) -> tuple[np.ndarray, float]:
    edge = np.asarray(edge, dtype=float)
    length = float(np.linalg.norm(edge))
    if length <= _DEF_TOL:
        raise ValueError(f"{name} has near-zero length and cannot define a box axis.")
    return edge / length, length


def _is_orthogonal_triplet(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, tol: float = _DEF_TOL
) -> bool:
    return (
        abs(float(v1 @ v2)) <= tol
        and abs(float(v1 @ v3)) <= tol
        and abs(float(v2 @ v3)) <= tol
    )


def _match_points_unordered(
    points_a: np.ndarray, points_b: np.ndarray, tol: float = _DEF_TOL
) -> bool:
    used = np.zeros(len(points_b), dtype=bool)
    for pa in points_a:
        diff = np.linalg.norm(points_b - pa, axis=1)
        idx = int(np.argmin(diff))
        if used[idx] or diff[idx] > tol:
            return False
        used[idx] = True
    return True


def _build_bounds_from_corner_edges(
    origin: np.ndarray,
    edge1: np.ndarray,
    edge2: np.ndarray,
    edge3: np.ndarray,
    name: str | None = None,
    *,
    is_preserve_axis_order: bool = True,
) -> Bounds:
    axis1, length1 = _normalize_box_edge(edge1, name="edge1")
    axis2, length2 = _normalize_box_edge(edge2, name="edge2")
    axis3, length3 = _normalize_box_edge(edge3, name="edge3")

    if not _is_orthogonal_triplet(axis1, axis2, axis3):
        raise ValueError(
            "The input edges do not form an orthogonal box. "
            "Please convert this geometry to BoundsGeneral instead."
        )

    handedness = float(np.dot(np.cross(axis1, axis2), axis3))
    if handedness < 0:
        if is_preserve_axis_order:
            raise ValueError(
                "The input box edges form a left-handed frame under the given "
                "axis order. Please reorder the input points, or convert this "
                "geometry to BoundsGeneral instead."
            )
        axis2, axis3 = axis3, axis2
        length2, length3 = length3, length2

    return Bounds(
        name=name,
        opts=OptsBounds(
            origin=origin,
            axis1=axis1,
            axis2=axis2,
            length1=length1,
            length2=length2,
            length3=length3,
            alignment="min_corner",
        ),
    )


def _build_bounds_from_4_points(points: np.ndarray, name: str | None = None) -> Bounds:
    origin = points[0]
    edge1 = points[1] - origin
    edge2 = points[2] - origin
    edge3 = points[3] - origin
    return _build_bounds_from_corner_edges(
        origin,
        edge1,
        edge2,
        edge3,
        name=name,
        is_preserve_axis_order=True,
    )


def _build_bounds_from_bounds6(values: np.ndarray, name: str | None = None) -> Bounds:
    xmin, xmax, ymin, ymax, zmin, zmax = values.tolist()
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
            alignment="min_corner",
        ),
    )


def _build_bounds_from_8_points(points: np.ndarray, name: str | None = None) -> Bounds:
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 3):
        raise ValueError(f"Expected (8, 3) points for a box, got shape {points.shape}.")

    for i in range(8):
        origin = points[i]
        others = np.delete(points, i, axis=0)
        dist = np.linalg.norm(others - origin, axis=1)
        order = np.argsort(dist)

        for idx1, order_idx1 in enumerate(order):
            for order_idx2 in order[idx1 + 1 :]:
                edge1 = others[order_idx1] - origin
                edge2 = others[order_idx2] - origin

                try:
                    axis1, _ = _normalize_box_edge(edge1, name="edge1")
                    axis2, _ = _normalize_box_edge(edge2, name="edge2")
                except ValueError:
                    continue

                if abs(float(axis1 @ axis2)) > _DEF_TOL:
                    continue

                axis3_dir = np.cross(axis1, axis2)
                norm3 = float(np.linalg.norm(axis3_dir))
                if norm3 <= _DEF_TOL:
                    continue
                axis3_dir = axis3_dir / norm3

                for candidate in others:
                    edge3 = candidate - origin
                    try:
                        axis3, _ = _normalize_box_edge(edge3, name="edge3")
                    except ValueError:
                        continue

                    if abs(abs(float(axis3 @ axis3_dir)) - 1.0) > _DEF_TOL:
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
                        ],
                        dtype=float,
                    )
                    if _match_points_unordered(expected, points):
                        return _build_bounds_from_corner_edges(
                            origin,
                            edge1,
                            edge2,
                            edge3,
                            name=name,
                            is_preserve_axis_order=False,
                        )

    raise ValueError(
        "The input 8-point geometry does not describe an orthogonal box. "
        "Please convert this geometry to BoundsGeneral instead."
    )


def _polydata_to_box_points(polydata: pv.PolyData) -> np.ndarray:
    surface = polydata.extract_surface().triangulate().clean()
    points = np.asarray(surface.points, dtype=float)
    if points.size == 0:
        raise ValueError("clip_geometry PolyData is empty.")

    rounded = np.round(points, decimals=10)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_points = points[np.sort(unique_idx)]
    return unique_points


def bounds_minimal_wrapping_points(
    points,
    axes,
    origin=None,
    name: str | None = "minimal bounds",
    min_lengths=None,
) -> Bounds:
    """Build the smallest ``Bounds`` wrapping points in a supplied axes frame.

    ``Bounds`` requires positive side lengths, so degenerate dimensions are
    floored by ``min_lengths`` or the module tolerance.
    """
    points = as_points(
        points,
        name="points used to build minimal wrapping bounds",
        dim=3,
        min_num=1,
    )
    axes = np.asarray(axes, dtype=float).copy()
    if axes.shape != (3, 3):
        raise ValueError(f"Expected axes to have shape (3, 3), got {axes.shape}.")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=_DEF_TOL):
        raise ValueError("`axes` must be an orthonormal frame with columns as axes.")
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]

    if origin is None:
        origin = np.zeros(3, dtype=float)
    else:
        origin = as_Vect(
            origin,
            name="origin used to project points into the supplied axes",
            dim=3,
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
    """Return a copy of bounds with side lengths expanded about its center."""
    if not isinstance(bounds, Bounds):
        raise TypeError("`bounds` must be a Bounds instance.")

    expand_factors = np.asarray(expand_factors, dtype=float)
    if expand_factors.shape != (3,):
        raise ValueError(
            f"Expected expand_factors to have shape (3,), got {expand_factors.shape}."
        )
    if np.any(expand_factors <= 0):
        raise ValueError("`expand_factors` must contain only positive values.")

    if min_lengths is None:
        min_lengths = np.zeros(3, dtype=float)
    else:
        min_lengths = as_dimension_info(min_lengths, name="min_lengths").astype(float)
        if np.any(min_lengths < 0):
            raise ValueError("`min_lengths` cannot contain negative values.")

    base_lengths = np.array(
        [bounds.opts.length1, bounds.opts.length2, bounds.opts.length3],
        dtype=float,
    )
    expanded_lengths = np.maximum(base_lengths * expand_factors, min_lengths)

    return Bounds(
        name=name,
        opts=OptsBounds(
            origin=bounds.opts.origin,
            axis1=bounds.opts.axis1,
            axis2=bounds.opts.axis2,
            length1=expanded_lengths[0],
            length2=expanded_lengths[1],
            length3=expanded_lengths[2],
            alignment=bounds.opts.alignment,
        ),
    )


def bounds_sample_points(
    bounds: Bounds,
    spacing=1.0,
    *,
    is_return_local: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Generate fixed-spacing sample points inside a ``Bounds`` object."""
    if not isinstance(bounds, Bounds):
        raise TypeError("`bounds` must be a Bounds instance.")

    spacing = as_dimension_info(spacing, name="spacing").astype(float)
    if np.any(spacing <= 0):
        raise ValueError("`spacing` must contain only positive values.")

    axis1 = bounds.opts.axis1
    axis2 = bounds.calc_axis2
    axis3 = bounds.calc_axis3
    axes = np.column_stack([axis1, axis2, axis3])

    length1 = bounds.opts.length1
    length2 = length1 if bounds.opts.length2 is None else bounds.opts.length2
    length3 = length1 if bounds.opts.length3 is None else bounds.opts.length3
    lengths = np.asarray([length1, length2, length3], dtype=float)

    if bounds.opts.alignment == "center":
        center = bounds.opts.origin
    elif bounds.opts.alignment == "min_corner":
        center = bounds.opts.origin + 0.5 * (
            length1 * axis1 + length2 * axis2 + length3 * axis3
        )
    else:
        raise ValueError(f"Unsupported bounds alignment {bounds.opts.alignment!r}.")

    local_axes = []
    for length, step in zip(lengths, spacing):
        sample_count = max(2, int(np.floor(length / step)) + 1)
        local_axes.append(np.linspace(-0.5 * length, 0.5 * length, sample_count))

    mesh = np.meshgrid(*local_axes, indexing="ij")
    local_points = np.column_stack([axis_values.ravel() for axis_values in mesh])
    points = apply_linear_transform(
        local_points,
        transform=axes.T,
        offset=center,
    )

    if is_return_local:
        return points, local_points
    return points


def obb_bounds_from_fit(fit: OBBFit, name: str | None = "seed bounds") -> Bounds:
    """Convert a pure OBB fit result into a repository ``Bounds`` object."""
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


def as_bounds(input_data, name: str = "bounds") -> Bounds | None:
    """Convert supported box-like inputs to a ``Bounds`` instance."""
    if input_data is None:
        return None

    if isinstance(input_data, Bounds):
        return input_data

    if isinstance(input_data, pv.PolyData):
        unique_points = _polydata_to_box_points(input_data)
        if unique_points.shape != (8, 3):
            raise ValueError(
                f"{name!r} PolyData does not look like a box: it has "
                f"{len(unique_points)} unique points. Please convert this "
                "geometry to BoundsGeneral instead."
            )
        return _build_bounds_from_8_points(unique_points, name=name)

    arr = np.asarray(input_data, dtype=float)

    if arr.ndim == 1 and arr.shape == (6,):
        return _build_bounds_from_bounds6(arr, name=name)

    if arr.ndim == 2 and arr.shape == (4, 3):
        return _build_bounds_from_4_points(arr, name=name)

    if arr.ndim == 2 and arr.shape == (8, 3):
        return _build_bounds_from_8_points(arr, name=name)

    raise TypeError(
        f"{name!r} could not be converted to Bounds. Supported inputs are: None, Bounds, "
        "axis-aligned bounds with 6 numbers, four box-defining points, eight box corners, "
        "or a box-like PyVista PolyData."
    )
