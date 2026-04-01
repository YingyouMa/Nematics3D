"""Glyph visuals, opts resolution, bounds binding, and live figure updates."""

from __future__ import annotations
import datetime
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, List, Literal, Mapping, Sequence, Type

import numpy as np
import pyvista as pv

from nematics3d.datatypes import (
    UNSET,
    Unset,
    ColorRGB,
    Vect,
    as_Number,
    as_str,
    as_bool,
    as_ColorRGB,
    as_ColorRGB_array,
    as_Vect,
    as_points,
)
from nematics3d.format import save_opts_json
from nematics3d.general import find_nearest_point, fmt_value
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..bounds import BoundsData, as_bounds
from ..host_base import OptsBase, HostBase
from .plot_figure import FigureData, as_plotfigure

#!!! colorbar name args manager
#!!! is_reset_camera commit

# --- Type aliases ---
ColorMode = ColorRGB | Callable | Sequence
OpacityMode = float | Callable | Sequence
RadiusMode = float | Callable | Sequence
ScalarsMode = Callable | Sequence | None
# fmt: off
@dataclass(slots=True, repr=False)
class OptsGlyph(OptsBase):
    """
    Glyph-specific opts object for visual style, resolver inputs, and scalar
    display behavior.

    For most users, an ``OptsGlyph`` instance is accessed through ``glyph.opts``
    rather than created in isolation.

    Important readable attributes on ``OptsGlyph`` include:
    - ``host`` to access the owning glyph object when attached

    User-facing `act_*` methods on `OptsGlyph` include the inherited `OptsBase`
    helpers, with glyph-aware JSON export behavior:

    - ``act_finalize()`` to fill defaults and enter the functioning state
    - ``act_asdict()`` to export the current glyph opts payload
    - ``act_save_json()`` to serialize glyph opts, replacing callable visual
      resolvers with the current resolved ``calc_*`` arrays when available
    - ``act_load_json()`` to load a saved JSON payload back into this instance

    Representation behavior is split intentionally:
    - ``str(opts)`` gives a compact one-line identity like ``OptsGlyph``
    - ``repr(opts)`` prints the full field-by-field summary that is meant for
      interactive inspection
    """
    # --- Visibility & Global ---
    is_visible:                 bool | Unset                        = UNSET
    is_pickable:                bool | Unset                        = UNSET
    shading_type:               Literal["phong", "pbr"] | Unset     = UNSET
    is_reset_camera:            bool | Unset                        = UNSET

    # --- Phong Lighting ---
    ambient:                    float | Unset                       = UNSET
    diffuse:                    float | Unset                       = UNSET
    specular:                   float | Unset                       = UNSET
    specular_power:             float | Unset                       = UNSET
    specular_color:             ColorRGB | Unset                    = UNSET

    # --- PBR Lighting ---
    metallic:                   float | Unset                       = UNSET
    roughness:                  float | Unset                       = UNSET

    # --- Shape & Color ---
    paint_by:                   Literal["color", "scalars"] | Unset = UNSET
    color:                      ColorMode | Unset                   = UNSET
    opacity:                    OpacityMode | Unset                 = UNSET
    scalars:                    ScalarsMode | Unset                 = UNSET
    radius:                     RadiusMode | Unset                  = UNSET
    resolver_source:             str | Unset                         = UNSET

    # --- Scalars (used if color == "scalars") ---
    scalars_cmap:               str | Unset                         = UNSET
    scalars_clim:               Vect(2) | None | Unset              = UNSET
    is_scalar_bar:              bool | Unset                        = UNSET
    scalar_bar_title:           str | Unset                         = UNSET

    # --- Geometry ---
    sides:                      int | Unset                         = UNSET
    __attrs__: ClassVar[Mapping[str, str]] = {
        **(OptsBase.__attrs__),
        
        # === Visibility & Global Settings ===
        "is_visible":           "Whether the glyph is visible in the scene.",
        "is_pickable":          "Whether the glyph could be picked by mouse in the scene.",
        "shading_type":         "'phong', 'pbr' (Physical)",
        "is_reset_camera":      "Whether to reset the camera settings for each (re-)plot.",
        
        # === Lighting - Phong ===
        "ambient":              "Reflected light from environment (0-1).",
        "diffuse":              "Standard matte reflection (0-1).",
        "specular":             "Glossy highlight strength (0-1).",
        "specular_power":       "Focus of gloss (1-100). Higher = shinier/smaller spot.",
        "specular_color":       "The color of the glossy highlight (RGB). Usually white [1,1,1].",
        
        # === Lighting - PBR ===
        "metallic":             "PBR metallic effect (0-1). Needs PBR enabled.",
        "roughness":            "PBR surface roughness (0-1). Needs PBR enabled.",
        
        # === Shape and Color Control ===
        "paint_by":             "Select rendering pipeline: direct RGBA vs scalar colormap.",
        "color": (
            "Determines point colors. Options: "
            "1) ColorRGB for entire glyph (e.g. (1,0,0)) "
            "2) Function (mapping function), "
            "3) color data set manually, "
        ),
        "opacity": (
            "Determines point transparency. Options: "
            "1) float 0-1 for entire glyph, "
            "2) Function (mapping function), "
            "3) opacity data set manually."
        ),
        "scalars": (
            "Determines point scalars. Options: "
            "1) Function (mapping function), "
            "2) scalars data set manually, "
            "3) None (No scalars)"
        ),
        "radius": (
            "Determines glyph thickness. Options: "
            "1) float for entire glyph, "
            "2) Function (mapping function), "
            "3) radius data set manually."
        ),
        "resolver_source": (
            "Defines the input passed to callable visual resolvers. "
            "Use 'coords' for raw coordinates or 'u_percent' for the point-index percentage along the glyph."
        ),
        
        # === Scalars Control (Needs color_rule='scalars') ===
        "scalars_cmap":         "Colormap name (e.g., 'viridis') used if color is set to scalar.",
        "scalars_clim":         "Color limits [min, max] for scalar mapping.",
        "is_scalar_bar":        "Whether to display the color legend (scalar bar).",
        "scalar_bar_title":     "Title for the scalar bar (e.g., 'Stress (MPa)').",
        
        # --- Geometry ---
        "sides":                "Number of facets around the glyph (higher = smoother).",
    }

    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **(OptsBase.impl_validators),
        "is_visible":           lambda v, d: as_bool(v, name=d),
        "is_pickable":          lambda v, d: as_bool(v, name=d),
        "shading_type":         lambda v, d: as_str(v, name=d, pool=("phong", "pbr")),
        "is_reset_camera":      lambda v, d: as_bool(v, name=d),
        "ambient":              lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "diffuse":              lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "specular":             lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "specular_power":       lambda v, d: as_Number(v, name=d, value_range=(1, 100), bounded=True),
        "specular_color":       lambda v, d: as_ColorRGB(v, name=d),
        "metallic":             lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "roughness":            lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "paint_by":             lambda v, d: as_str(v, name=d, pool=("color", "scalars")),
        "resolver_source":      lambda v, d: as_str(v, name=d, pool=("coords", "u_percent")),
        "scalars_cmap":         lambda v, d: as_str(v, name=d),
        "scalars_clim":         lambda v, d: (v if v is None else as_Vect(v, name=d, dim=2)),
        "is_scalar_bar":        lambda v, d: as_bool(v, name=d),
        "scalar_bar_title":     lambda v, d: as_str(v, name=d),
        "sides":                lambda v, d: as_Number(v, name=d, is_int=True, value_range=(3, 128), bounded=True),
        }


    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType({
        "tag":                  "glyph options",
        "is_visible":           True,
        "is_pickable":          True,
        "shading_type":         "phong",
        "is_reset_camera":      True,
        "ambient":              0.2,
        "diffuse":              0.7,
        "specular":             0.2,
        "specular_power":       20.0,
        "specular_color":       (1.0, 1.0, 1.0),
        "metallic":             0.0,
        "roughness":            0.5,
        "paint_by":             "color",
        "color":                (0.5, 0.5, 0.5),
        "opacity":              1.0,
        "scalars":              lambda x: np.arange(len(x)),
        "radius":               0.5,
        "resolver_source":      "coords",
        "scalars_cmap":         "viridis",
        "scalars_clim":         None,
        "is_scalar_bar":        True,
        "scalar_bar_title":     "scalar",
        "sides":                12,
    })

    _actor_attr: ClassVar[Mapping[str, str]] = {
        "is_visible":           "visibility",
        "is_pickable":          "pickable",
        "shading_type":         "prop.interpolation",
        "ambient":              "prop.ambient",
        "diffuse":              "prop.diffuse",
        "specular":             "prop.specular",
        "specular_power":       "prop.specular_power",
        "specular_color":       "prop.specular_color",
        "metallic":             "prop.metallic",
        "roughness":            "prop.roughness",
        }
    
    # ==================== OVERRIDE ====================
    # OptsGlyph overrides OptsBase._helper_host_apply so opts updates are
    # forwarded through the owning glyph commit pipeline.
    # ==================================================

    def _helper_host_apply(self, key, value, *, host=None):
        host = self.host if host is None else host
        if host:
            host.act_commit(**{key: value})

    # ==================== OVERRIDE ====================
    # OptsGlyph overrides OptsBase.act_save_json so callable visual resolvers
    # are saved as the current resolved `calc_*` arrays when the owning glyph
    # is available.
    # ==================================================
    @logging_and_warning_decorator(start_finish_level=5)
    def act_save_json(
        self,
        path: str | Path,
        *,
        max_inline_array_size: int = 64,
        is_include_unset: bool = False,
        logger=None,
    ) -> Path:
        """Save glyph opts to JSON, materializing callable fields as current calc arrays."""
        opts_dict = self.act_asdict(is_include_unset=is_include_unset)
        host = self.host
        if host is not None:
            for key, value in list(opts_dict.items()):
                if callable(value):
                    calc_name = f"calc_{key}"
                    calc_value = getattr(host, calc_name, None)
                    if isinstance(calc_value, np.ndarray):
                        opts_dict[key] = calc_value.copy()

        path = save_opts_json(
            opts_dict,
            path,
            opts_class_name=type(self).__name__,
            max_inline_array_size=max_inline_array_size,
        )
        logger.info(f"Saved opts JSON to {path}.")
        return path
# fmt: on


# Subclassing rules:
# - PlotGlyph subclasses must preserve the HostBase commit contract while also
#   keeping the glyph-specific render pipeline consistent: resolved data ->
#   polydata -> mesh -> actor.
# - Subclasses should override `_helper_build_mesh()` to define geometry and
#   should override `_helper_build_poly()` only when the default point-data
#   preparation is not sufficient.
# - Keep figure registration, bounds binding, and pick registration aligned so
#   actor lifecycle stays synchronized with PlotFigure lifecycle.
# - Preserve the distinction between raw inputs (`raw_*`), resolved arrays
#   (`calc_*`), and live plotter entities (`entity_*`).


class PlotGlyph(HostBase):
    """
    Base class for drawable glyph-style objects attached to a PlotFigure.

    For most users, concrete subclasses of ``PlotGlyph`` are created by
    higher-level visualization helpers rather than instantiated directly.

    A ``PlotGlyph`` combines the normal ``HostBase`` object/opts model with a
    glyph-specific render pipeline:

    - raw geometric inputs such as ``raw_coords``
    - resolved visual arrays such as ``calc_color`` and ``calc_radius``
    - live plotter entities such as the actor and optional silhouette

    Important readable attributes on ``PlotGlyph`` include:
    - ``opts`` to access the paired ``OptsGlyph`` object controlling rendering
    - ``fig`` to access the containing ``PlotFigure``
    - ``bounds`` to inspect the currently bound clipping ``Bounds`` object

    User-facing `show_*` methods on `PlotGlyph` are inherited from `HostBase`:

    - ``show_readable_attrs()`` to list readable glyph, host, and opts surfaces
    - ``show_attr_desc()`` to explain one glyph attr, relation, alias, or opts attr
    - ``show_modifiable_attrs()`` to separate host attrs, opts attrs, extra attrs,
      and writable properties
    - ``show_relations()`` / ``show_relation_tree()`` to inspect figure, bounds,
      wrapper, and other object links
    - ``show_saved_opts()`` to list named snapshots stored in ``opts_backup``

    User-facing `act_*` methods on `PlotGlyph` include both inherited host
    actions and glyph-specific rendering helpers. Common ones are:

    - ``act_commit()`` to apply host and opts updates through the managed glyph
      update pipeline
    - ``act_bind_bounds()`` / ``act_unbind_bounds()`` to manage clipping bounds
    - ``act_save_opts()`` to snapshot current opts into ``opts_backup``
    - ``act_highlight()`` / ``act_dehighlight()`` to control silhouette emphasis
    - ``act_interact()`` / ``act_set_interact_func()`` to manage glyph-side
      interaction hooks
    - ``act_remove()`` to detach the glyph from its figure and live plotter state

    PlotGlyph manages both the resolved visual data of the glyph and the live
    actor created in the plotter.
    """

    # fmt: off
    __attr_defs__ = {
        **dict(HostBase.__attr_defs__),
        "raw_name": {
            **dict(HostBase.__attr_defs__["raw_name"]),
            "doc": "The name identifier of the glyph.",
        },
        "raw_category": {
            "doc": "The category of the glyph, used in the classification of PlotFigure.",
            "validator": lambda v, d: as_str(v, name=d),
        },
        "raw_coords": {
            "doc": "The N x 3 input coordinates of each glyph.",
            "validator": lambda v, d: as_points(v, name=d),
            "is_reapply_opts_after_raw": True,
        },
        "state_clip_mode": {
            "doc": (
                "Clip strategy for bounds application. 'mesh' clips the built "
                "mesh while 'center' clips the center representation before meshing."
            ),
            "validator": lambda v, d: as_str(v, name=d, pool=("mesh", "center")),
            "is_reapply_opts_after_raw": True,
        },
        "state_is_clip_inside": {
            "doc": "Whether bounds clipping keeps the region inside the bounds (True) or outside (False).",
            "validator": lambda v, d: as_bool(v, name=d),
            "is_reapply_opts_after_raw": True,
        },
        "state_is_silhouette": {
            "doc": "Whether silhouette actors should be rebuilt during glyph updates.",
            "validator": lambda v, d: as_bool(v, name=d),
        },
        "state_is_interactable": {
            "doc": "Whether to create a control window when the instance is double right-clicked.",
            "validator": lambda v, d: as_bool(v, name=d),
        },
        "calc_coords": {
            "doc": "The effective coordinates used for the current glyph build after clip-mode preprocessing.",
        },
        "calc_poly": {
            "doc": "The generated PyVista PolyData.",
        },
        "calc_color": {
            "doc": "The resolved per-point RGB color array of the glyph.",
        },
        "calc_opacity": {
            "doc": "The resolved per-point opacity array of the glyph.",
        },
        "calc_radius": {
            "doc": "The resolved per-point radius array used for glyph thickness.",
        },
        "calc_scalars": {
            "doc": "The resolved per-point scalar array used for scalar coloring.",
        },
        "calc_is_empty": {
            "doc": "Whether the glyph currently has no drawable geometry and should skip render-side updates.",
        },
        "entity_actor": {
            "doc": "The PyVista Actor corresponding to this object in the plotter.",
        },
        "entity_silhouette": {
            "doc": "The PyVista Actor used as the silhouette highlight for this object.",
        },
        "fig": {
            "doc": "The PlotFigure instance containing this glyph.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "bounds": {
            "doc": "The Bounds instance clipping this glyph.",
            "kind": "relation",
            "is_weak_by_default": True,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "impl_name_pv": {
            "doc": "The unique identifier of this glyph stored in the PyVista plotter.",
        },
        "impl_interact_func": {
            "doc": "The function to trigger the control window when the instance is double right-clicked.",
        },
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
    )

    _pending_resolution_attrs: List[str] = ["radius", "opacity", "color", "scalars"]
    # fmt: on

    # ==================== OVERRIDE ====================
    # PlotGlyph overrides HostBase.__init__ because it must validate glyph
    # inputs, choose or create a figure, and initialize render-specific state
    # before the generic host lifecycle is completed.
    # ==================================================

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        opts_type: Type[OptsBase],
        category: str,
        name: str | None,
        name_replace: str,
        opts: OptsGlyph | None = None,
        figure: FigureData | None = None,
        bounds: BoundsData | None = None,
        clip_mode: str = "center",
        is_clip_inside: bool = True,
        is_subscribe_bounds: bool = True,
        is_passive_bounds_sync: bool = False,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):

        coords = type(self).__attr_defs__["raw_coords"]["validator"](
            coords,
            type(self).__attr_defs__["raw_coords"]["doc"],
        )
        object.__setattr__(self, "raw_coords", coords)
        clip_mode = type(self).__attr_defs__["state_clip_mode"]["validator"](
            clip_mode,
            type(self).__attr_defs__["state_clip_mode"]["doc"],
        )
        object.__setattr__(self, "state_clip_mode", clip_mode)
        is_clip_inside = type(self).__attr_defs__["state_is_clip_inside"]["validator"](
            is_clip_inside,
            type(self).__attr_defs__["state_is_clip_inside"]["doc"],
        )
        object.__setattr__(self, "state_is_clip_inside", is_clip_inside)
        object.__setattr__(self, "calc_coords", coords.copy())
        category = type(self).__attr_defs__["raw_category"]["validator"](
            category,
            type(self).__attr_defs__["raw_category"]["doc"],
        )
        object.__setattr__(self, "raw_category", category)
        if name is None:
            name = name_replace
        else:
            name = as_str(
                name,
                name=type(self).__attr_defs__["raw_name"]["doc"],
                replace=name_replace,
            )
        object.__setattr__(self, "opts_backup", {})
        object.__setattr__(self, "state_is_silhouette", True)
        object.__setattr__(self, "calc_is_empty", False)
        object.__setattr__(self, "state_is_interactable", True)

        super().__init__(
            opts_type,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs,
        )

        if self.opts.paint_by is UNSET:
            if self.opts.color is UNSET and self.opts.scalars is not UNSET:
                self.opts.paint_by = "scalars"
            elif self.opts.color is not UNSET and self.opts.scalars is UNSET:
                self.opts.paint_by = "color"
            elif self.opts.color is not UNSET and self.opts.scalars is not UNSET:
                logger.warning(
                    "Both 'color' and 'scalars' are provided, but 'paint_by' is not explicitly specified."
                    "The default paint_by strategy will be applied."
                )

        figure = as_plotfigure(figure)
        self.act_bind_relation_base("fig", figure, is_weak=True)

        self.opts.act_finalize(self.opts_defaults)
        str_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        unique_id = self.name + str_now
        object.__setattr__(self, "impl_name_pv", unique_id)

        object.__setattr__(self, "impl_interact_func", None)
        self.act_bind_bounds(
            bounds,
            is_apply=False,
            is_subscribe=is_subscribe_bounds,
            is_passive_sync=is_passive_bounds_sync,
        )

    def _helper_init_end(self):

        for attr in self._pending_resolution_attrs:
            self._helper_resolver_spec(attr)
        if self.state_clip_mode == "center":
            object.__setattr__(self, "calc_coords", self._helper_bound_coords())
        else:
            object.__setattr__(self, "calc_coords", self.raw_coords.copy())
        self._helper_make_figure()

        figure = self.fig
        figure.pl.render()
        figure.act_register(self)


    def act_unbind_bounds(self, is_apply=True):
        """Detach the current bounds object and optionally reapply the glyph state."""
        bounds_old = self.bounds
        if bounds_old is None:
            return
        bounds_old.act_unregister_subscriber(sync_name=self.impl_name_pv, host=self)
        self.act_unbind_relation_base("bounds")
        if is_apply:
            self.act_commit(is_reapply_opts=True)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_bind_bounds(
        self,
        bounds: BoundsData | None,
        is_apply=True,
        is_replace=True,
        is_subscribe=True,
        is_passive_sync=False,
        logger=None,
    ):
        """Bind one bounds object to this glyph and optionally subscribe for sync updates."""
        if bounds is None:
            self.act_unbind_bounds(is_apply=is_apply)
            return

        try:
            bounds = as_bounds(bounds, name="The bounds controlling this glyph")
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
                raise RuntimeError("This glyph is already bound to a Bounds object.")
            self.act_unbind_bounds(is_apply=False)

        self.act_bind_relation_base("bounds", bounds, is_weak=True)
        if is_subscribe:
            if not is_passive_sync:
                bounds.act_attach_sync_task(
                    self.impl_name_pv,
                    lambda **kwargs: self.act_commit(is_reapply_opts=True),
                )
            bounds.act_register_subscriber(
                self,
                sync_name=self.impl_name_pv,
                kind="glyph",
            )
        if is_apply:
            self.act_commit(is_reapply_opts=True)

    def _helper_apply_bounds_mesh(self, mesh):
        bounds = self.bounds
        if bounds is None:
            return mesh
        return mesh.clip_surface(
            bounds.clip_geometry,
            invert=self.state_is_clip_inside,
        )

    def _helper_bound_coords(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement center-based bounds clipping yet."
        )

    # ----------------------------------------------------------------------------------------------------
    # Resolver function: to resolve point-wise properties (color, opacity, etc) for each inidividual glyph
    # ----------------------------------------------------------------------------------------------------

    def _helper_get_resolver_source(self):
        source_name = as_str(
            self.opts.resolver_source,
            name="glyph resolver source",
            pool=("coords", "u_percent"),
        )
        if source_name == "coords":
            return self.raw_coords
        n_points = len(self.raw_coords)
        if n_points == 0:
            return np.empty((0,), dtype=np.float32)
        return np.arange(n_points, dtype=np.float32) / float(n_points) * 100.0

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_generic(
        self, attr_name, attr_input, default_val, is_recover=False, logger=None
    ):

        target_shape = (
            (len(self.raw_coords), 3)
            if attr_name == "color"
            else (len(self.raw_coords),)
        )
        source = self._helper_get_resolver_source()

        try:
            if attr_input is None:
                raise TypeError(f"Require input for {attr_name!r}. Got None instead.")
            elif callable(attr_input):
                resolved = np.asarray(attr_input(source), dtype=np.float32)
            else:
                arr = np.asarray(attr_input, dtype=float)
                if arr.shape == () and attr_name == "color":
                    raise TypeError(
                        f"To provide a single value for color, the input should be expressed by (R, G, B). Got {attr_input} instead."
                    )
                resolved = np.full(target_shape, arr, dtype=np.float32)

            if resolved.shape != target_shape:
                raise ValueError(
                    f"Shape mismatch for {attr_name!r}: got {resolved.shape}, expected {target_shape}."
                )

            if attr_name == "color":
                resolved = as_ColorRGB_array(
                    resolved, name="The pairwise color data of glyph"
                )

            object.__setattr__(self, "calc_" + attr_name, resolved)
            object.__setattr__(self.opts, attr_name, attr_input)

        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
            if is_recover:
                raise ValueError(f"The default value is not valid for {attr_name!r}!")
            else:
                logger.exception(f"Failed to resolve {attr_name!r}")
                if getattr(self, "entity_actor", None):
                    logger.recovery("Automatically ignore this modification.")
                else:
                    logger.recovery(
                        f"Reset {attr_name!r} to default."
                        f"To find it, check self.opts_defaults['{attr_name}']."
                    )
                    self._helper_resolver_generic(
                        attr_name, default_val, default_val, is_recover=True
                    )


    def _helper_resolver_spec(self, attr_name, attr_value=None):

        if attr_value is None:
            attr_value = getattr(self.opts, attr_name)

        return self._helper_resolver_generic(
            attr_name, attr_value, self.opts_defaults[attr_name]
        )

    # ----------------------------------------------------------------------------------------------------
    # Create the polydata, mesh and actor.
    # ----------------------------------------------------------------------------------------------------

    def _helper_build_poly(self):
        poly = pv.PolyData(self.calc_coords)
        object.__setattr__(self, "calc_poly", poly)
        self._helper_set_poly(poly)

    def _helper_set_poly(self, poly):
        if hasattr(self, "calc_radius"):
            poly.point_data["radius"] = self.calc_radius
        poly.point_data["opacity"] = self.calc_opacity
        poly.point_data["scalars"] = self.calc_scalars
        rgba_values = np.hstack([self.calc_color, self.calc_opacity.reshape(-1, 1)])
        poly.point_data["rgba"] = rgba_values

    def _helper_build_mesh(self):
        raise NotImplementedError(...)

    def _helper_is_empty_mesh(self, mesh):
        if mesh is None:
            return True
        if getattr(mesh, "n_points", 0) == 0:
            return True
        if getattr(mesh, "n_cells", 0) == 0:
            return True
        return False

    def _helper_clear_live_actor(self):
        fig = self.fig
        if fig is None:
            object.__setattr__(self, "entity_actor", None)
            self._helper_clear_silhouette()
            return

        plotter = fig.pl
        actor = getattr(self, "entity_actor", None)

        def _safe_remove_actor(target):
            try:
                plotter.remove_actor(target, render=False)
            except AttributeError:
                # The Qt panel may outlive the PyVista renderer during shutdown.
                # In that case, the renderer bookkeeping is already gone and
                # there is nothing left for us to remove cleanly.
                return

        if actor is not None:
            pm = getattr(fig, "_entity_pick_manager", None)
            if pm is not None:
                try:
                    pm.act_unregister(actor)
                except (KeyError, AttributeError, RuntimeError):
                    pm._impl_registry.pop(actor, None)
            _safe_remove_actor(actor)
        _safe_remove_actor(self.impl_name_pv)
        self._helper_clear_silhouette()
        object.__setattr__(self, "entity_actor", None)

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_make_figure(self, logger=None):
        """
        Creates or updates the rendering in a PyVista Plotter.
        """

        is_scalars = self.opts.paint_by == "scalars"
        unique_id = self.impl_name_pv

        input_dir = {
            "name": unique_id,
            "pbr": self.opts.shading_type == "pbr",
            "rgb": not is_scalars,
            "scalars": "scalars" if is_scalars else "rgba",
            "reset_camera": self.opts.is_reset_camera,
        }
        if is_scalars:
            input_dir["opacity"] = "opacity"
            input_dir["cmap"] = self.opts.scalars_cmap
            input_dir["show_scalar_bar"] = self.opts.is_scalar_bar
            input_dir["clim"] = self.opts.scalars_clim
            input_dir["scalar_bar_args"] = {"title": self.opts.scalar_bar_title}

        if self.state_clip_mode == "center":
            object.__setattr__(self, "calc_coords", self._helper_bound_coords())
            self._helper_build_poly()
            mesh = self._helper_build_mesh()
        else:
            object.__setattr__(self, "calc_coords", self.raw_coords.copy())
            self._helper_build_poly()
            mesh = self._helper_build_mesh()
            mesh = self._helper_apply_bounds_mesh(mesh)

        if self._helper_is_empty_mesh(mesh):
            object.__setattr__(self, "calc_is_empty", True)
            self._helper_clear_live_actor()
            self.fig.pl.render()
            return

        object.__setattr__(self, "calc_is_empty", False)
        self._helper_clear_live_actor()
        plotter = self.fig.pl
        actor = plotter.add_mesh(mesh, **input_dir)

        prop = actor.prop

        shading = self.opts.shading_type.lower()
        if shading not in ("pbr", "phong"):
            try:
                raise ValueError("shading type must either be `pbr` or `phong`")
            except ValueError:
                logger.exception("Please check input")
                logger.recovery("Use `phong` in the following.")
            shading = "phong"
        prop.interpolation = shading
        object.__setattr__(self.opts, "shading_type", shading)

        prop.ambient = self.opts.ambient
        prop.diffuse = self.opts.diffuse
        prop.specular = self.opts.specular
        prop.specular_power = self.opts.specular_power
        prop.specular_color = self.opts.specular_color

        if shading == "pbr":
            prop.metallic = self.opts.metallic
            prop.roughness = self.opts.roughness

        actor.visibility = self.opts.is_visible
        actor.pickable = self.opts.is_pickable

        object.__setattr__(self, "entity_actor", actor)
        self._helper_register_pick(actor)

        if self.state_is_silhouette:
            self._helper_add_silhouette()
        else:
            self._helper_clear_silhouette()

    def _helper_add_silhouette(self):

        plotter = self.fig.pl

        self._helper_clear_silhouette()

        mesh = self.entity_actor.mapper.dataset
        surf = mesh.extract_surface().triangulate().clean()

        actor_silhouette = plotter.add_silhouette(
            surf,
            color=(0, 0, 0),
            line_width=6,
            opacity=0.8,
        )

        actor_silhouette.visibility = False
        actor_silhouette.pickable = False

        object.__setattr__(self, "entity_silhouette", actor_silhouette)

    def _helper_clear_silhouette(self):
        fig = self.fig
        actor_silhouette = getattr(self, "entity_silhouette", None)
        if fig is None or actor_silhouette is None:
            object.__setattr__(self, "entity_silhouette", None)
            return

        fig.pl.remove_actor(actor_silhouette)
        object.__setattr__(self, "entity_silhouette", None)

    # ----------------------------------------------------------------------------------------------------
    # The functios to update the given point-wise data values
    # ----------------------------------------------------------------------------------------------------

    def _helper_update_rgba(self):
        mapper = self.entity_actor.mapper
        mapper.scalar_visibility = True
        mapper.color_mode = "direct"
        mapper.lookup_table = None
        mapper.dataset.set_active_scalars("rgba")
        mapper.SetArrayName("rgba")
        if self.opts.scalar_bar_title in self.fig.pl.scalar_bars.keys():
            self.fig.pl.remove_scalar_bar(title=self.opts.scalar_bar_title)


    def _helper_update_scalars(self):

        mapper = self.entity_actor.mapper
        mesh_data = mapper.dataset.point_data

        if "__custom_rgba" in mesh_data.keys():
            mesh_data.remove("__custom_rgba")

        if not isinstance(mapper.lookup_table, pv.LookupTable):
            mapper.lookup_table = pv.LookupTable()

        mapper.set_scalars(
            mesh_data["scalars"],
            "scalars",
            cmap=self.opts.scalars_cmap,
            clim=self.opts.scalars_clim,
            custom_opac=True,
            opacity=mesh_data["opacity"],
        )

        if not self.opts.is_scalar_bar:
            if self.opts.scalar_bar_title in self.fig.pl.scalar_bars.keys():
                self.fig.pl.remove_scalar_bar(title=self.opts.scalar_bar_title)
            return

        if self.opts.scalar_bar_title not in self.fig.pl.scalar_bars.keys():
            self.fig.pl.add_scalar_bar(
                title=self.opts.scalar_bar_title, mapper=mapper, render=False
            )

    # ----------------------------------------------------------------------------------------------------
    # The register and un-register of glyphs onto PlotFigure instance
    # ----------------------------------------------------------------------------------------------------

    def _helper_register_pick(self, actor):

        fig = self.fig
        if fig is None:
            return
        pm = getattr(fig, "_entity_pick_manager", None)
        if pm is None:
            return
        pm.act_register(actor=actor, owner=self)

    def act_remove(self):
        """Remove this glyph from its figure, bounds subscriptions, and live actors."""
        bounds_visual_source = getattr(self, "_impl_bounds_visual_source", None)
        bounds_visual_sync_name = getattr(self, "_impl_bounds_visual_sync_name", None)
        if bounds_visual_source is not None:
            bounds_visual_source._helper_unregister_visual_sync(
                bounds_visual_sync_name,
                tube=self,
            )
        figure = self.fig
        self.act_unbind_bounds(is_apply=False)
        self._helper_clear_live_actor()
        if figure is not None:
            figure.act_unregister(self, is_missing_ok=True)

    # ==================== OVERRIDE ====================
    # PlotGlyph overrides HostBase._helper_commit_apply_opts_main to resolve
    # visual data, rebuild mesh state when needed, and push updates into the
    # live actor inside the figure plotter.
    # ==================================================

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):
        if not is_reapply_opts and not kwargs:
            return

        paint_method = kwargs.pop("paint_by", None)
        if paint_method is None:
            has_color, has_scalars = "color" in kwargs, "scalars" in kwargs
            if has_color ^ has_scalars:
                paint_method = "scalars" if has_scalars else "color"
        if paint_method is None:
            paint_method = self.opts.paint_by
        else:
            object.__setattr__(self.opts, "paint_by", paint_method)

        resolver_source = kwargs.pop("resolver_source", None)
        is_reresolve = is_reapply_opts
        if resolver_source is not None:
            object.__setattr__(self.opts, "resolver_source", resolver_source)
            is_reresolve = True

        is_needs_remesh = is_reresolve
        for attr in self._pending_resolution_attrs:
            if attr not in kwargs:
                if is_reresolve:
                    self._helper_resolver_spec(attr)
            else:
                self._helper_resolver_spec(attr, attr_value=kwargs.pop(attr))
                is_needs_remesh = True

        if "sides" in kwargs:
            object.__setattr__(self.opts, "sides", kwargs["sides"])
            is_needs_remesh = True

        if is_needs_remesh:
            if self.state_clip_mode == "center":
                object.__setattr__(self, "calc_coords", self._helper_bound_coords())
                self._helper_build_poly()
                mesh = self._helper_build_mesh()
            else:
                object.__setattr__(self, "calc_coords", self.raw_coords.copy())
                self._helper_build_poly()
                mesh = self._helper_build_mesh()
                mesh = self._helper_apply_bounds_mesh(mesh)
            if self._helper_is_empty_mesh(mesh):
                object.__setattr__(self, "calc_is_empty", True)
                self._helper_clear_live_actor()
            else:
                object.__setattr__(self, "calc_is_empty", False)
                if getattr(self, "entity_actor", None) is None:
                    self._helper_make_figure()
                else:
                    self.entity_actor.mapper.SetInputData(mesh)
                    self.entity_actor.mapper.Update()
                    if self.state_is_silhouette:
                        self._helper_add_silhouette()
                    else:
                        self._helper_clear_silhouette()

        for key, value in kwargs.items():
            try:
                attr_path_actor = self.opts._actor_attr.get(key, None)
                if attr_path_actor and getattr(self, "entity_actor", None) is not None:
                    parts = attr_path_actor.split(".")
                    obj = self.entity_actor
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)

                object.__setattr__(self.opts, key, value)
            except (AttributeError, TypeError, ValueError, KeyError):
                logger.exception(f"Failed to reset value of {key!r}")
                logger.recovery("Ignore this modification")

        if getattr(self, "entity_actor", None) is not None and not self.calc_is_empty:
            if paint_method == "color":
                self._helper_update_rgba()
            else:
                self._helper_update_scalars()

        self.fig.pl.render()

    def act_highlight(
        self,
        color: ColorRGB | None = None,
        opacity: float | None = None,
        width: float | None = None,
    ):
        """Show the silhouette highlight and optionally update its visual style."""

        silhouette = getattr(self, "entity_silhouette", None)
        if not silhouette:
            return

        silhouette.visibility = True

        color = (
            as_ColorRGB(color, name="The color of silhouette", replace=None)
            if color is not None
            else None
        )
        opacity = (
            as_Number(
                opacity,
                name="The opacity of silhouette",
                value_range=(0, 1),
                replace=None,
            )
            if opacity is not None
            else None
        )
        width = (
            as_Number(
                width,
                name="The line width of silhouette",
                value_range=(0, np.inf),
                replace=None,
            )
            if width is not None
            else None
        )

        sil_prop = silhouette.prop
        if color is not None:
            sil_prop.color = color
        if opacity is not None:
            sil_prop.opacity = opacity
        if width is not None:
            sil_prop.line_width = width

    def act_dehighlight(self):
        """Hide the current silhouette highlight if one exists."""
        silhouette = getattr(self, "entity_silhouette", None)
        if silhouette:
            silhouette.visibility = False

    def _helper_resolve_pick(self, picked_point):
        pos, idx = find_nearest_point(picked_point, self.raw_coords, is_return_idx=True)
        with np.printoptions(precision=2, suppress=True):
            if self.opts.paint_by == "color":
                msg = f"Local color: {fmt_value(self.calc_color[idx])} \n"
            else:
                msg = f"Local scalar: {fmt_value(self.calc_scalars[idx])} \n"
            for attr in self._pending_resolution_attrs:
                if attr in ("color", "scalars"):
                    pass
                else:
                    attr_name = "calc_" + attr
                    value = object.__getattribute__(self, attr_name)[idx]
                    value = fmt_value(value)
                    msg += f"Local {attr}: {value} \n"
        return pos, msg, idx

    def act_interact(self):
        """Open or trigger the configured interaction callback for this glyph."""
        if getattr(self, "state_is_interactable", False):
            func = getattr(self, "impl_interact_func", None)
            if func is None:
                raise RuntimeError(
                    f"{type(self).__name__} is interactable but no interact function has been set."
                )
            if callable(func):
                func()
            else:
                raise RuntimeError("impl_interact_func is not callable.")

    @logging_and_warning_decorator(start_finish_level=5)
    def act_set_interact_func(self, func, logger=None):
        """Register the callable used when this glyph enters its interaction flow."""
        if callable(func):
            object.__setattr__(self, "impl_interact_func", func)
        else:
            try:
                raise RuntimeError("impl_interact_func is not callable.")
            except RuntimeError:
                logger.exception("Check input.")
                logger.recovery("Automatically ignore this modification")

    # ==================== OVERRIDE ====================
    # PlotGlyph overrides ClassBase/HostBase.__repr__ to keep the glyph string
    # form compact and focused on its class and public name.
    # ==================================================

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})"
        return msg
