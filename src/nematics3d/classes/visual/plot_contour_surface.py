"""Contour-surface mesh visuals built on the shared PlotGlyph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np
import pyvista as pv

from ...datatypes import UNSET, Unset, as_ColorRGB, as_Number, as_bool, as_str
from ..bounds import BoundsData
from ..contour_surface import ContourSurface
from .glyph import OptsGlyph, PlotGlyph
from .plot_figure import FigureData


@dataclass(slots=True, repr=False)
class OptsContourSurface(OptsGlyph):
    """Visual configuration object for ``PlotContourSurface``."""

    is_show_edges: bool | Unset = UNSET
    edge_color: tuple[float, float, float] | Unset = UNSET
    line_width: float | Unset = UNSET
    style: str | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__attrs__),
        "is_show_edges": "Whether triangle edges should be rendered on the contour mesh.",
        "edge_color": "Edge color used when is_show_edges is enabled.",
        "line_width": "Displayed edge line width.",
        "style": "Mesh representation style: 'surface' or 'wireframe'.",
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **dict(OptsGlyph.impl_validators),
        "is_show_edges": lambda v, d: as_bool(v, name=d),
        "edge_color": lambda v, d: as_ColorRGB(v, name=d),
        "line_width": lambda v, d: as_Number(
            v,
            name=d,
            value_range=(0.0, np.inf),
        ),
        "style": lambda v, d: as_str(v, name=d, pool=("surface", "wireframe")),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsGlyph.impl_defaults_frozen),
            "ambient": 0.5,
            "is_show_edges": False,
            "edge_color": (0.0, 0.0, 0.0),
            "line_width": 1.0,
            "style": "surface",
        }
    )

    impl_actor_attr: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.impl_actor_attr),
        "is_show_edges": "prop.show_edges",
        "edge_color": "prop.edge_color",
        "line_width": "prop.line_width",
        "style": "prop.style",
    }


class PlotContourSurface(PlotGlyph):
    """Render one extracted ``ContourSurface`` mesh in a figure."""

    __attr_defs__ = {
        **dict(PlotGlyph.__attr_defs__),
        "calc_level": {
            "doc": "Current contour level copied from the owning ContourSurface.",
        },
        "impl_owner_sync_name": {
            "doc": "Internal sync-task key used to auto-refresh from the owner surface.",
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
    )

    _pending_resolution_attrs = ["color", "scalars", "opacity"]

    def __init__(
        self,
        surface: ContourSurface,
        name: str | None = None,
        name_replace: str = "contour-surface-plot",
        category: str = "contour_surface",
        figure: FigureData | None = None,
        opts: OptsContourSurface | None = None,
        bounds: BoundsData | None = None,
        clip_mode: str = "mesh",
        is_clip_inside: bool = True,
        opts_defaults_override: Mapping[str, Any] | None = None,
        is_extract: bool = True,
        **kwargs,
    ):
        if not isinstance(surface, ContourSurface):
            raise TypeError(
                "`surface` must be a ContourSurface instance for PlotContourSurface."
            )

        if surface.mesh is None:
            if not is_extract:
                raise ValueError(
                    "The input ContourSurface has no extracted mesh. "
                    "Either extract it first or pass is_extract=True."
                )
            surface.act_extract()

        mesh = surface.mesh
        if mesh is None:
            raise RuntimeError("ContourSurface mesh extraction failed.")
        coords = np.asarray(mesh.points, dtype=float)

        super().__init__(
            coords=coords,
            opts_type=OptsContourSurface,
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

        self.act_bind_relation_base(
            "owner",
            surface,
            doc="The ContourSurface that owns this plot wrapper.",
            is_weak=True,
        )
        object.__setattr__(self, "calc_level", float(surface.raw_level))
        object.__setattr__(
            self,
            "impl_owner_sync_name",
            f"{self.impl_name_pv}__owner_refresh",
        )
        surface.act_attach_sync_task(
            self.impl_owner_sync_name,
            self._sync_from_owner_surface,
        )

        self._helper_init_end()

    def _helper_bound_coords(self):
        """Contour mesh visuals use mesh clipping instead of center-point clipping."""
        return self.raw_coords.copy()

    def _helper_build_mesh(self):
        """Return the current contour mesh with resolved display arrays attached."""
        surface = self.owner
        if surface is None:
            raise RuntimeError("PlotContourSurface lost its ContourSurface owner.")

        mesh_source = surface.mesh
        if mesh_source is None:
            mesh_source = surface.act_extract()

        mesh = mesh_source.copy(deep=True)
        mesh.points = np.asarray(self.calc_coords, dtype=float)
        mesh.point_data["opacity"] = np.asarray(self.calc_opacity, dtype=np.float32)
        mesh.point_data["scalars"] = np.asarray(self.calc_scalars, dtype=np.float32)
        rgba_values = np.hstack(
            [
                np.asarray(self.calc_color, dtype=np.float32),
                np.asarray(self.calc_opacity, dtype=np.float32).reshape(-1, 1),
            ]
        )
        mesh.point_data["rgba"] = rgba_values
        return mesh

    def act_refresh_mesh(self, *, is_extract: bool = True):
        """Refresh this plot wrapper from the current owner mesh and level."""
        surface = self.owner
        if surface is None:
            raise RuntimeError("PlotContourSurface lost its ContourSurface owner.")
        if surface.mesh is None:
            if not is_extract:
                raise ValueError(
                    "The owning ContourSurface has no mesh to refresh from."
                )
            surface.act_extract()

        object.__setattr__(self, "calc_level", float(surface.raw_level))
        self.act_commit(raw_coords=np.asarray(surface.mesh.points, dtype=float))

    def _sync_from_owner_surface(self, **kwargs):
        """Refresh this plot wrapper when the owner contour surface updates its mesh."""
        del kwargs
        self.act_refresh_mesh(is_extract=False)

    def act_remove(self):
        """Detach the owner sync callback, then remove the glyph from the figure."""
        owner = self.owner
        if owner is not None:
            owner.act_detach_sync_task(self.impl_owner_sync_name)
            if getattr(owner, "visual", None) is self:
                owner.act_unbind_relation_base("visual")
        super().act_remove()
