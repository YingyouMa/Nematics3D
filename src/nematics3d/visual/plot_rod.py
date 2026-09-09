"""Rod glyph visuals built on the shared PlotGlyph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
import pyvista as pv

from nematics3d.analysis.bounds import BoundsData
from nematics3d.core.class_base import AttrDef
from nematics3d.core.host_base import HostBase
from nematics3d.datatypes import UNSET, Unset, as_points, as_str
from nematics3d.format import fmt_value
from nematics3d.logging_decorator import logging_and_warning_decorator
from nematics3d.visual.glyph import OptsGlyph, PlotGlyph, _as_resolver_source_or_none
from nematics3d.visual.plot_figure import FigureData
from nematics3d.visual.qt.interact_rod import InteractRod

LengthMode = float | Callable | Sequence


@dataclass(slots=True, repr=False)
class OptsRod(OptsGlyph):
    """
    Visual configuration object for `PlotRod`.

    `OptsRod` stores the settings that control how rod glyphs look after
    they are created. It does not define rod centers or orientations;
    those come from the `coords` and `orient` passed to `PlotRod`.
    Instead, this class controls appearance, coloring, shading, scalar
    mapping, thickness, rod length, and meshing details.

    Important readable attributes:

    - `host`: the PlotRod currently using this opts object, if any.
    - `length`, `radius`, `color`, `opacity`, `scalars`: the main per-rod
      visual controls.
    - `paint_by`: chooses direct RGBA painting or scalar-colormap rendering.
    - `resolver_source`: selects the input used by callable visual resolvers.
    - `sides`: cross-section resolution of the shared rod source geometry.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON, using sidecar `.npy`
      files when large arrays are present.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Common ways to use this object:

    - create `OptsRod(...)` first and pass it into `PlotRod`
    - modify fields on `rod.opts` after a rod glyph already exists
    - apply a prepared settings object with `rod.act_commit(opts=opts)`

    Most visual fields support the same three input styles:

    - one shared value applied to every rod
    - one value per rod, provided as an array
    - a callable resolver that computes values from the source selected by
      `resolver_source`

    The most useful fields for day-to-day work are usually:

    - `length`: rod length
    - `radius`: rod thickness
    - `color`: direct RGB coloring
    - `opacity`: transparency
    - `paint_by`: choose direct coloring or scalar-colormap rendering
    - `scalars`: numeric values used when `paint_by="scalars"`
    - `resolver_source`: choose what callable resolvers receive
    - `sides`: roundness of the rod cross-section

    `resolver_source` controls the input passed to callable visual resolvers:

    - `"coords"`: the callable receives the raw rod-center coordinates
    - `"u_percent"`: the callable receives point-index percentages from 0
      to 100 along the glyph ordering
    - `"orient"`: the callable receives the raw orientation vectors. This is
      the default setting for rods.

    A few useful relationships to keep in mind:

    - `color` and `scalars` belong to different rendering pipelines;
      `paint_by` decides which one is active
    - `scalars` are numeric data, not RGB colors
    - `resolver_source` matters only when a visual field is provided as a
      callable
    - `length` and `orient` determine the per-instance transform applied to
      the shared rod source geometry
    - lighting fields such as `ambient`, `diffuse`, `specular`, `metallic`,
      and `roughness` change appearance but not geometry

    If you want the full field list and their short descriptions, see
    `OptsRod.__attrs__`.
    For the shared glyph option model and lower-level commit/update rules,
    see the docstrings of `OptsGlyph` and `OptsBase`.

    Examples
    --------
    Create reusable rod options:

    >>> opts = OptsRod(length=3.0, radius=0.3, color=(0.9, 0.2, 0.2))
    >>> rods = PlotRod(coords, orient, opts=opts)

    Use one length for every rod:

    >>> opts = OptsRod(length=2.0)

    Use one length per rod:

    >>> opts = OptsRod(length=np.array([1.0, 2.0, 3.0])) # three rods

    Resolve values from coordinates:

    >>> opts = OptsRod(
    ...     resolver_source="coords",
    ...     length=lambda pts: 1.0 + np.abs(pts[:, 2]),
    ... )

    Resolve values from orientation vectors:

    >>> opts = OptsRod(
    ...     resolver_source="orient",
    ...     color=lambda n: np.abs(n),
    ...     length=lambda n: 1.0 + 2.0 * np.abs(n[:, 2]),
    ... )

    Use scalar coloring:

    >>> opts = OptsRod(
    ...     paint_by="scalars",
    ...     scalars=lambda n: n[:, 2],
    ...     resolver_source="orient",
    ...     scalars_cmap="viridis",
    ... )
    """

    length: LengthMode | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "length": "The length of rods",
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph.impl_validators),
        "resolver_source": lambda v, d: as_str(
            v,
            name=d,
            pool=("coords", "u_percent", "orient"),
        ),
        "resolver_source_color": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "u_percent", "orient")
        ),
        "resolver_source_opacity": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "u_percent", "orient")
        ),
        "resolver_source_radius": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "u_percent", "orient")
        ),
        "resolver_source_scalars": lambda v, d: _as_resolver_source_or_none(
            v, d, pool=("coords", "u_percent", "orient")
        ),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph.impl_defaults_frozen),
            "length": 3,
            "radius": 0.3,
            "resolver_source": "orient",
        }
    )


class PlotRod(PlotGlyph):
    """Render one oriented rod at each input point."""

    _use_instanced_glyph_mapper = True

    __attr_defs__ = {
        "raw_orient": AttrDef(
            doc="The orientation vectors of rods.",
            kind="raw",
            validator=lambda v, d: as_points(v, name=d),
            is_reapply_opts_after_raw=True,
        ),
        "calc_length": AttrDef(
            doc="The resolved per-rod length array used for rod geometry building.",
            kind="calc",
        ),
        "calc_keep_index": AttrDef(
            doc="Indices of raw rod centers kept after center-based clipping.",
            kind="calc",
        ),
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in HostBase.__slots__
    )

    _pending_resolution_attrs: Sequence[str] = PlotGlyph._pending_resolution_attrs + [
        "length"
    ]

    def __init__(
        self,
        coords: np.ndarray,
        orient: np.ndarray,
        name: str = "rod",
        name_replace: str = "rod",
        category: str = "rods",
        figure: FigureData | None = None,
        opts: OptsRod | None = None,
        bounds: BoundsData | None = None,
        clip_mode: str = "center",
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        orient = (
            type(self)
            .__attr_defs__["raw_orient"]
            .validator(
                orient,
                type(self).__attr_defs__["raw_orient"].doc,
            )
        )
        object.__setattr__(self, "raw_orient", orient)

        super().__init__(
            coords=coords,
            opts_type=OptsRod,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            bounds=bounds,
            clip_mode=clip_mode,
            is_clip_inside=is_clip_inside,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        if len(self.raw_orient) != len(self.raw_coords):
            raise ValueError(
                f"There are {len(self.raw_orient)} points for orientation, "
                f"while {len(self.raw_coords)} points for positions."
            )

        object.__setattr__(self, "calc_keep_index", None)
        self.act_set_interact_func(lambda: InteractRod.show_once(self, self.fig))
        self._helper_init_end()

    def _helper_get_resolver_source_name(self, attr_name=None):
        source_name = None
        if attr_name is not None:
            override_attr = self._resolver_source_override_attr_names.get(attr_name)
            if override_attr is not None:
                source_name = getattr(self.opts, override_attr, None)
        if source_name is None:
            source_name = self.opts.resolver_source
        return as_str(
            source_name,
            name="glyph resolver source",
            pool=("coords", "u_percent", "orient"),
        )

    def _helper_get_resolver_source(self, attr_name=None):
        source_name = self._helper_get_resolver_source_name(attr_name)
        if source_name == "orient":
            return self.raw_orient
        return super()._helper_get_resolver_source(attr_name)

    def _helper_expand_endpoint_values(self, values, keep_index=None):
        values = np.asarray(values)
        if keep_index is not None:
            keep_index = np.asarray(keep_index, dtype=int)
            values = values[keep_index]
        return np.repeat(values, 2, axis=0)

    def _helper_bound_coords(self):
        coords, keep_index = self._helper_filter_centers_by_bounds()
        object.__setattr__(self, "calc_keep_index", keep_index)
        return coords

    def _helper_prepare_rod_geometry(self):
        """Resolve kept centers, normalized directions, lengths, and radii."""
        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        points = self.calc_coords
        orient = self.raw_orient[keep_index].copy()
        orient_norm = np.linalg.norm(orient, axis=1, keepdims=True)
        mask = orient_norm.squeeze() > 1e-5
        orient[mask] /= orient_norm[mask]
        length = self.calc_length[keep_index]
        radius = self.calc_radius[keep_index]
        return keep_index, points, orient, orient_norm, mask, length, radius

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_build_poly(self, logger=None):
        keep_index, points, orient, orient_norm, mask, length, radius = (
            self._helper_prepare_rod_geometry()
        )
        if len(points) == 0:
            poly = pv.PolyData(np.empty((0, 3), dtype=float))
            object.__setattr__(self, "calc_poly", poly)
            self._helper_set_poly(poly)
            return
        if not np.all(mask):
            n_bad = np.count_nonzero(~mask)
            logger.warning(
                f"{n_bad} rod(s) have near-zero orientation norm (<= 1e-5). "
                "Their directions are left unnormalized, which may lead to "
                "degenerate or invisible rods."
            )
        poly = pv.PolyData(points)
        object.__setattr__(self, "calc_poly", poly)
        self._helper_set_poly(poly)
        poly.point_data["orient"] = orient

        scale = np.empty((len(points), 3), dtype=float)
        scale[:, 0] = length
        scale[:, 1] = radius
        scale[:, 2] = radius
        poly.point_data["scale"] = scale

    def _helper_set_poly(self, poly):
        if poly.n_points == 0:
            return

        keep_index = getattr(self, "calc_keep_index", None)
        if keep_index is None:
            keep_index = np.arange(len(self.raw_coords), dtype=int)

        color = self.calc_color[keep_index]
        opacity = self.calc_opacity[keep_index]
        radius = self.calc_radius[keep_index]
        scalars = self.calc_scalars[keep_index]

        poly.point_data["radius"] = radius
        poly.point_data["opacity"] = opacity
        poly.point_data["scalars"] = scalars
        rgba_values = np.hstack([color, opacity.reshape(-1, 1)])
        poly.point_data["rgba"] = rgba_values

    def _helper_materialize_mesh(self):
        """Materialize tube geometry for compatibility-only mesh operations."""
        keep_index, points, orient, _orient_norm, _mask, length, radius = (
            self._helper_prepare_rod_geometry()
        )
        if len(points) == 0:
            return pv.PolyData()
        half = 0.5 * length.reshape(-1, 1)
        p_minus = points - half * orient
        p_plus = points + half * orient
        endpoints = np.empty((2 * len(points), 3), dtype=float)
        endpoints[0::2] = p_minus
        endpoints[1::2] = p_plus
        lines = np.empty((len(points), 3), dtype=np.int64)
        lines[:, 0] = 2
        lines[:, 1] = 2 * np.arange(len(points))
        lines[:, 2] = 2 * np.arange(len(points)) + 1
        poly = pv.PolyData(endpoints, lines=lines.ravel())
        poly.point_data["radius"] = np.repeat(radius, 2)

        mesh = poly.tube(
            scalars="radius",
            n_sides=self.opts.sides,
            absolute=True,
        )

        object.__setattr__(self, "calc_poly", poly)
        return mesh

    def _helper_build_glyph_source(self):
        """Return one unit rod aligned with +x for GPU instancing."""
        return pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            radius=1.0,
            height=1.0,
            resolution=self.opts.sides,
            capping=True,
        )

    def _helper_configure_instanced_mapper_geometry(self, mapper):
        """Apply per-rod orientation, length, and radius without meshing tubes."""
        mapper.SetOrientationArray("orient")
        mapper.SetOrientationModeToDirection()
        mapper.OrientOn()
        mapper.SetScaleArray("scale")
        mapper.SetScaleModeToScaleByVectorComponents()
        mapper.ScalingOn()

    def _helper_resolve_pick(self, picked_point):
        pos, msg, idx = super()._helper_resolve_pick(picked_point)
        value = fmt_value(self.raw_orient[idx])
        msg = f"Local orientation: {value} \n" + msg
        return pos, msg, idx
