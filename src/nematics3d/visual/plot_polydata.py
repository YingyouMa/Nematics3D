"""PolyData-backed mesh visuals built on the shared PlotGlyph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from nematics3d.classes.bounds import BoundsData
from nematics3d.core.class_base import AttrDef
from nematics3d.core.host_base import HostBase
from nematics3d.datatypes import UNSET, Unset, as_bool, as_ColorRGB, as_number, as_str
from nematics3d.geometry.polydata import as_polydata_input, copy_polydata_geometry
from nematics3d.visual.glyph import OptsGlyph, PlotGlyph
from nematics3d.visual.plot_figure import FigureData
from nematics3d.visual.qt.interact_polydata import InteractPolyData


@dataclass(slots=True, repr=False)
class OptsPolyData(OptsGlyph):
    """Visual configuration object for ``PlotPolyData``."""

    is_show_edges: bool | Unset = UNSET
    edge_color: tuple[float, float, float] | Unset = UNSET
    edge_width: float | Unset = UNSET
    style: str | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "is_show_edges": "Whether polygon edges should be rendered on the mesh.",
        "edge_color": "Edge color used when is_show_edges is enabled.",
        "edge_width": "Displayed edge line width.",
        "style": "Mesh representation style: 'surface' or 'wireframe'.",
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsGlyph.impl_validators),
        "is_show_edges": lambda v, d: as_bool(v, name=d),
        "edge_color": lambda v, d: as_ColorRGB(v, name=d),
        "edge_width": lambda v, d: as_number(v, name=d, value_range=(0.0, np.inf)),
        "style": lambda v, d: as_str(v, name=d, pool=("surface", "wireframe")),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph.impl_defaults_frozen),
            "ambient": 0.5,
            "is_show_edges": False,
            "edge_color": (0.0, 0.0, 0.0),
            "edge_width": 1.0,
            "style": "surface",
        }
    )

    impl_actor_attr: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.impl_actor_attr),
        "is_show_edges": "prop.show_edges",
        "edge_color": "prop.edge_color",
        "edge_width": "prop.line_width",
        "style": "prop.style",
    }


class PlotPolyData(PlotGlyph):
    """Render an existing PolyData-like mesh through Nematics3D.

    Geometry and topology come from the caller. The input is normalized to
    ``pyvista.PolyData`` and copied into an internal geometry/topology-only
    template. Existing point, cell, and field arrays are intentionally
    stripped so Nematics3D exclusively owns the managed display arrays
    (``rgba``, ``opacity``, and ``scalars``).

    The caller-owned mesh is never mutated. Each render starts from a fresh
    deep copy of the template, attaches resolved pointwise display arrays,
    and then applies mesh clipping when bounds are active. The class is
    intentionally point-data-oriented; input cell arrays are not currently
    exposed as rendering resolver sources.
    """

    # fmt: off
    __attr_defs__ = {
        "raw_poly": AttrDef(
            doc=(
                "The normalized pyvista.PolyData geometry/topology template "
                "used to rebuild this visual. Input data arrays are stripped."
            ),
            kind="raw",
        ),
    }
    # fmt: on

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in HostBase.__slots__
    )

    _pending_resolution_attrs = ["color", "scalars", "opacity"]

    def __init__(
        self,
        polydata,
        name: str | None = None,
        name_replace: str = "polydata",
        category: str = "polydata",
        figure: FigureData | None = None,
        opts: OptsPolyData | None = None,
        bounds: BoundsData | None = None,
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        poly = as_polydata_input(polydata, name="polydata")
        coords = np.asarray(poly.points, dtype=float)

        super().__init__(
            coords=coords,
            opts_type=OptsPolyData,
            category=category,
            name=name,
            name_replace=name_replace,
            opts=opts,
            figure=figure,
            bounds=bounds,
            clip_mode="mesh",
            is_clip_inside=is_clip_inside,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

        object.__setattr__(self, "raw_poly", copy_polydata_geometry(poly))
        self.act_register_protected_attr(["coords", "raw_coords", "poly", "raw_poly"])
        self.act_set_interact_func(lambda: InteractPolyData.show_once(self, self.fig))
        self._helper_init_end()

    def _helper_materialize_mesh(self):
        mesh = self.raw_poly.copy(deep=True)
        mesh.points = np.asarray(self.calc_coords, dtype=float)
        mesh.point_data["opacity"] = np.asarray(self.calc_opacity, dtype=np.float32)
        mesh.point_data["scalars"] = np.asarray(self.calc_scalars, dtype=np.float32)
        mesh.point_data["rgba"] = np.hstack(
            [
                np.asarray(self.calc_color, dtype=np.float32),
                np.asarray(self.calc_opacity, dtype=np.float32).reshape(-1, 1),
            ]
        )
        return mesh


__all__ = ["OptsPolyData", "PlotPolyData"]
