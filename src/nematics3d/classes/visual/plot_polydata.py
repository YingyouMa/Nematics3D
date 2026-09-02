"""PolyData-backed mesh visuals built on the shared PlotGlyph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np
import pyvista as pv
import vtk

from ...datatypes import UNSET, Unset, as_ColorRGB, as_number, as_bool, as_str
from ..bounds import BoundsData
from ...core.class_base import AttrDef
from ...core.host_base import HostBase
from .glyph import OptsGlyph, PlotGlyph
from .plot_figure import FigureData
from .qt.interact_polydata import InteractPolyData


def as_polydata_input(data, name: str = "polydata input") -> pv.PolyData:
    """
    Normalize supported mesh-like inputs to ``pyvista.PolyData``.

    Supported inputs currently include:

    - ``pyvista.PolyData``
    - ``vtk.vtkPolyData``
    - surface-like ``pyvista.DataSet`` / ``vtk.vtkDataSet`` objects that can
      be converted through ``extract_surface()`` or ``cast_to_polydata()``
    """

    if isinstance(data, pv.PolyData):
        return data

    wrapped = None
    if isinstance(data, vtk.vtkPolyData):
        wrapped = pv.wrap(data)
    elif isinstance(data, pv.DataSet):
        wrapped = data
    elif isinstance(data, vtk.vtkDataSet):
        wrapped = pv.wrap(data)
    else:
        raise TypeError(
            f"{name} must be a pyvista.PolyData, vtkPolyData, or another "
            f"surface-like PyVista/VTK dataset. Got {type(data).__name__}."
        )

    if isinstance(wrapped, pv.PolyData):
        return wrapped

    for method_name in ("extract_surface", "cast_to_polydata"):
        method = getattr(wrapped, method_name, None)
        if method is None:
            continue
        try:
            candidate = method()
        except Exception:
            continue
        if isinstance(candidate, pv.PolyData):
            return candidate
        if candidate is not None:
            candidate = pv.wrap(candidate)
            if isinstance(candidate, pv.PolyData):
                return candidate

    raise TypeError(
        f"{name} with type {type(data).__name__} could not be converted to "
        "pyvista.PolyData."
    )


def make_clean_polydata(poly: pv.PolyData) -> pv.PolyData:
    """
    Return a deep-copied PolyData that keeps only geometry/topology.

    All attached point, cell, and field arrays are removed from the returned
    copy so the plot wrapper starts from a clean internal template.
    """

    poly_clean = poly.copy(deep=True)
    for attr_name in ("point_data", "cell_data", "field_data"):
        data_attr = getattr(poly_clean, attr_name, None)
        if data_attr is None:
            continue
        clear_method = getattr(data_attr, "clear", None)
        if callable(clear_method):
            clear_method()
    return poly_clean


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
        "edge_width": lambda v, d: as_number(
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
    """
    Render one existing ``pyvista.PolyData``-like mesh in a figure.

    `PlotPolyData` is the direct-mesh concrete glyph class. It takes an
    already defined surface/line/vertex-style mesh, normalizes it to
    `pyvista.PolyData`, and renders that mesh through the shared glyph
    display pipeline. This makes it useful when geometry is already available
    as a mesh and should be displayed without reconstructing topology from
    higher-level primitives such as points, centerlines, or vectors.

    Visual appearance is controlled through `opts`, explicit keyword
    arguments, or later updates with `act_commit(...)`. The most relevant
    pointwise visual fields for this class are `color`, `opacity`, and
    `scalars`; these can be provided as shared constants, per-point arrays,
    or callable resolvers. Callable resolvers use the source selected by
    `resolver_source`.

    Important readable attributes:

    - `opts`: the paired OptsPolyData controlling mesh appearance.
    - `fig`: the PlotFigure currently hosting this glyph, if any.
    - `bounds`: the currently bound clipping object, if any.
    - `raw_coords`: the raw point coordinates copied from the normalized input
      PolyData.
    - `raw_poly`: the normalized `pyvista.PolyData` template retained by this
      wrapper.

    Parameters
    ----------
    polydata
        Input mesh-like object. Supported inputs currently include
        `pyvista.PolyData`, `vtk.vtkPolyData`, and selected surface-like
        PyVista/VTK datasets that can be converted to `pyvista.PolyData`.
        The input is normalized immediately, and this wrapper stores its own
        deep-copied, data-stripped `raw_poly` template instead of mutating the
        caller's original object in place.
    name
        Optional readable object name.
    name_replace
        Fallback name used when `name` is not provided.
    category
        Category label used when the object is registered in a figure.
        The default is `"polydata"`.
    figure
        Optional figure/container for this glyph. You may pass an existing
        `PlotFigure`, a `pyvistaqt.BackgroundPlotter`, or a `pyvista.Plotter`.
        Non-`PlotFigure` inputs are wrapped into a `PlotFigure` internally so
        this glyph can join an existing scene without extra setup. If `None`,
        a new figure is created automatically.
    opts
        Optional `OptsPolyData` instance holding the visual configuration.
        You can also reuse an existing options object later with
        `mesh.act_commit(opts=other_mesh.opts)` to apply another object's
        current option settings directly. If both `opts` and explicit option
        keyword arguments are provided, the explicit keyword arguments are
        merged in and take precedence.
    is_clip_inside
        Controls whether clipping keeps the region inside the active bounds
        (`True`) or outside it (`False`). This is a glyph/host setting, not
        an `OptsPolyData` field.
    bounds
        Optional clipping object forwarded through the underlying `PlotGlyph`
        interface.
    opts_defaults_override and other advanced keyword arguments
        These mostly affect default resolution and higher-level host/glyph
        behavior. New users can usually ignore them at first; see the
        docstring of `PlotGlyph` if you want the full forwarding model.
    **kwargs
        Additional option values forwarded into the glyph configuration
        pipeline. For the full list of supported visual options, see the
        docstring of `OptsPolyData` and its base option classes.

    Data Preservation
    -----------------
    `PlotPolyData` does not directly mutate the caller-provided input mesh.
    During initialization it stores a deep-copied `raw_poly` with attached
    point/cell/field arrays removed, and each render rebuild starts from
    `raw_poly.copy(deep=True)`. The wrapper then writes its resolved display
    arrays such as `opacity`, `scalars`, and `rgba` onto that copied render
    mesh.

    Notes
    -----
    At the moment this class is point-data-oriented. Resolved visual arrays
    are attached to mesh points, not cells. Input mesh arrays are intentionally
    stripped from the internal `raw_poly` template during initialization, so
    plotting starts from clean geometry/topology rather than mixing external
    data arrays with Nematics3D-managed display arrays.
    """

    # fmt: off
    __attr_defs__ = {
        "raw_poly": AttrDef(
            doc=(
                "The normalized pyvista.PolyData source used as the topology "
                "and point-data template for this plot wrapper."
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

        object.__setattr__(self, "raw_poly", make_clean_polydata(poly))
        self.act_register_protected_attr(["coords", "raw_coords", "poly", "raw_poly"])
        self.act_set_interact_func(lambda: InteractPolyData.show_once(self, self.fig))
        self._helper_init_end()

    def _helper_bound_coords(self):
        return self.raw_coords.copy()

    def _helper_build_mesh(self):
        mesh = self.raw_poly.copy(deep=True)
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
