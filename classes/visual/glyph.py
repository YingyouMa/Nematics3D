from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Literal, Mapping, Sequence
import pyvista as pv
import weakref
import numpy as np
import datetime

from Nematics3D.datatypes import (
    UNSET,
    Unset,
    ColorRGB,
    Vect,
    as_Number,
    as_str,
    as_bool,
    as_ColorRGB,
    as_Vect,
)
from ..opts_base import OptsBase
from .plot_figure import PlotFigure
from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET, Unset, as_bool, as_Number, as_points, as_ColorRGB_array, as_str
from ..opts import merge_opts_all, build_defaults_with_override



LEVEL_ACTOR = 0  # Only changes GPU/Rendering state. (Fastest)
LEVEL_RECALC = 1  # Needs to re-calculate data arrays (colors, etc.) but keeps geometry.
LEVEL_REMESH = 2  # Needs to re-run the tube filter to rebuild the 3D mesh. (Heaviest)

# --- Type aliases ---
ColorMode = ColorRGB | Callable | Sequence | Literal["scalars"]
OpacityMode = float | Callable | Sequence
RadiusMode = float | Callable | Sequence
ScalarsMode = Callable | Sequence | None
ClipGeometryLike = list[float] | pv.PolyData | None


@dataclass(slots=True)
class OptsGlyph(OptsBase):
    # --- Visibility & Global ---
    name:                   str | Unset                         = UNSET
    category:               str | Unset                         = UNSET
    is_visible:             bool | Unset                        = UNSET
    shading_type:           Literal["phong", "pbr"] | Unset     = UNSET
    is_reset_camera:        bool | Unset                        = UNSET

    # --- Phong Lighting ---
    ambient:                float | Unset                       = UNSET
    diffuse:                float | Unset                       = UNSET
    specular:               float | Unset                       = UNSET
    specular_pow:           float | Unset                       = UNSET
    specular_color:         ColorRGB | Unset                    = UNSET

    # --- PBR Lighting ---
    metallic:               float | Unset                       = UNSET
    roughness:              float | Unset                       = UNSET

    # --- Shape & Color ---
    color:                  ColorMode | Unset                   = UNSET
    opacity:                OpacityMode | Unset                 = UNSET
    scalars:                ScalarsMode | Unset                 = UNSET
    radius:                 RadiusMode | Unset                  = UNSET

    # --- Scalars (used if color == "scalars") ---
    scalars_cmap:           str | Unset                         = UNSET
    scalars_clim:           Vect(2) | None | Unset              = UNSET
    is_scalar_bar:          bool | Unset                        = UNSET
    scalar_bar_title:       str | Unset                         = UNSET

    # --- Geometry & Clipping ---
    sides:                  int | Unset                         = UNSET
    clip_geometry:          ClipGeometryLike | Unset            = UNSET
    
    _state_is_category_locked: bool = field(default=False, init=False, repr=False)

    __descriptions__: ClassVar[Mapping[str, str]] = {
        # === Visibility & Global Settings ===
        "name":             "Identifier for the actor in the plotter.",
        "category":         "The semantic category of this plotting entity.",
        "is_visible":       "Whether the tube is visible in the scene.",
        "shading_type":     "'phong', 'pbr' (Physical)",
        "is_reset_camera":  "Whether to reset the camera settings for each (re-)plot.",
        
        # === Lighting - Phong ===
        "ambient":          "Reflected light from environment (0-1).",
        "diffuse":          "Standard matte reflection (0-1).",
        "specular":         "Glossy highlight strength (0-1).",
        "specular_pow":     "Focus of gloss (1-100). Higher = shinier/smaller spot.",
        "specular_color":   "The color of the glossy highlight (RGB). Usually white [1,1,1].",
        
        # === Lighting - PBR ===
        "metallic":         "PBR metallic effect (0-1). Needs PBR enabled.",
        "roughness":        "PBR surface roughness (0-1). Needs PBR enabled.",
        
        # === Shape and Color Control ===
        "color": (
            "Determines point colors. Options: "
            "1) ColorRGB for entire tube (e.g. (1,0,0)) "
            "2) Function (mapping function), "
            "3) color data set manually, "
            "4) 'scalars' (maps 1D data to colors using scalars_cmap/scalars_clim)."
        ),
        "opacity": (
            "Determines point transparency. Options: "
            "1) float 0-1 for entire tube, "
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
            "Determines tube thickness. Options: "
            "1) float for entire tube, "
            "2) Function (mapping function), "
            "3) radius data set manually."
        ),
        
        # === Scalars Control (Needs color_rule='scalars') ===
        "scalars_cmap":     "Colormap name (e.g., 'viridis') used if color is set to scalar.",
        "scalars_clim":     "Color limits [min, max] for scalar mapping.",
        "is_scalar_bar":    "Whether to display the color legend (scalar bar).",
        "scalar_bar_title": "Title for the scalar bar (e.g., 'Stress (MPa)').",
        
        # --- Geometry & Clipping ---
        "sides":            "Number of facets around the tube (higher = smoother).",
        "clip_geometry": (
            "(INVALID FOR NOW!!!) Define clipping boundary. Can be: "
            "1) List of 6 floats [xmin, xmax...] for axis-aligned box, "
            "2) A Mesh/PolyData representing any closed shape (e.g. 8-point box)."
        ),
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        "name":                 lambda v, d: as_str(v, name=d),
        "category":             lambda v, d: as_str(v, name=d),
        "is_visible":           lambda v, d: as_bool(v, name=d),
        "shading_type":         lambda v, d: as_str(v, name=d, pool=("phong", "pbr")),
        "is_reset_camera":      lambda v, d: as_bool(v, name=d),
        "ambient":              lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "diffuse":              lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "specular":             lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "specular_pow":         lambda v, d: as_Number(v, name=d, value_range=(1, 100), bounded=True),
        "specular_color":       lambda v, d: as_ColorRGB(v, name=d),
        "metallic":             lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "roughness":            lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
        "scalars_cmap":         lambda v, d: as_str(v, name=d),
        "scalars_clim":         lambda v, d: (v if v is None else as_Vect(v, name=d, dim=2)),
        "is_scalar_bar":        lambda v, d: as_bool(v, name=d),
        "scalar_bar_title":     lambda v, d: as_str(v, name=d),
        "sides":                lambda v, d: as_Number(v, name=d, is_int=True, value_range=(3, 128), bounded=True),
        }


    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        "name":                 "glyph",
        "category":             "glyph",
        "is_visible":           True,
        "shading_type":         "phong",
        "is_reset_camera":      True,
        "ambient":              0.2,
        "diffuse":              0.7,
        "specular":             0.2,
        "specular_pow":         20.0,
        "specular_color":       (1.0, 1.0, 1.0),
        "metallic":             0.0,
        "roughness":            0.5,
        "color":                (0.5, 0.5, 0.5),
        "opacity":              1.0,
        "scalars":              None,
        "radius":               0.5,
        "scalars_cmap":         "viridis",
        "scalars_clim":         None,
        "is_scalar_bar":        True,
        "scalar_bar_title":     "scalar",
        "sides":                12,
        "clip_geometry":        None,
    })


    _commit_level: ClassVar[Mapping[str, Any]] = {
        "color":                LEVEL_RECALC,
        "opacity":              LEVEL_RECALC,
        "scalars":              LEVEL_RECALC,
        "radius":               LEVEL_REMESH,
        "scalars_cmap":         LEVEL_RECALC,
        "scalars_clim":         LEVEL_RECALC,
        "sides":                LEVEL_REMESH,
        "clip_geometry":        LEVEL_REMESH,
    }

    _actor_attr: ClassVar[Mapping[str, str]] = {
        "is_visible":           "visibility",
        "shading_type":         "prop.interpolation",
        "ambient":              "prop.ambient",
        "diffuse":              "prop.diffuse",
        "specular":             "prop.specular",
        "specular_pow":         "prop.specular_power",
        "specular_color":       "prop.specular_color",
        "metallic":             "prop.metallic",
        "roughness":            "prop.roughness",
        }

    # ---------------------------------------------------------------------
    # Public API entrypoints (thin wrappers around OptsBase basics)
    # ---------------------------------------------------------------------
    def __setattr__(self, key: str, value: Any, logger=None):
        if getattr(self, "_state_is_category_locked", False) and key == "category":
            raise AttributeError("Modification of 'category' is not allowed, because it is used as the key in dir: PlotFigure._entity")
        self._helper_setattr_basic(key, value)

    def act_finalize(self, defaults: Mapping[str, Any] | None = None) -> None:
        self._helper_finalize_basic(defaults)

    def act_asdict(self, is_include_UNSET: bool = False) -> dict[str, Any]:
        return self._helper_asdict_basic(is_include_UNSET=is_include_UNSET)
    
    
class PlotGlyph:
    
    _internal_owner_ref: weakref.ReferenceType | None = field(
        default=None, repr=False, init=False
    )
    __descriptions__: ClassVar[Mapping[str, str]] = {
        "raw_coords": "The N x 3 input coordinates of each glyph",
        
        "_calc_poly": "The generated PyVista PolyData",
        "_calc_mesh": "The generated PyVista surface mesh",
        
        "_calc_color": "The resolved per-point RGB color array of the tube.",
        "_calc_opacity": "The resolved per-point opacity array of the tube.",
        "_calc_radius": "The resolved per-point radius array used for tube thickness.",
        "_calc_scalars": "The resolved per-point scalar array used for scalar coloring.",
        
        "_entity": "The PyVista Actor corresponding to this object in the plotter.",
        
        "opts": "The Opts instance controlling rendering and geometry options.",
        "opts_defaults": "The default option settings for tube visualization",
        
        "_internal_owner_ref": ("A weak reference to the PlotFigure object associated with this object."
                                "To access it, use .owner or ._internal_owner."),
        "_internal_name_pv": "The unique identifier of this tube stored in the PyVista plotter.",
        
        "_internal_extra_attrs": (
            "A dict storing user-registered extra attributes. "
            "These are accessed via `tube.<name>` after calling `act_add_attr(name, doc)`."
        ),
        "_internal_extra_attrs_docs": (
            "A dict storing docstrings for user-registered extra attributes."
        ),
        }
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        opts: OptsGlyph,   
        figure: PlotFigure | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
        **kwargs
            ):
        
        object.__setattr__(self, "_internal_extra_attrs", {})
        object.__setattr__(self, "_internal_extra_attrs_docs", {})
        
        logger.detail("Building default option values ...")
        opts_defaults = build_defaults_with_override(
                            opts._DEFAULTS_FROZEN,
                            opts_defaults_override,
                            name=type(opts).__name__,
                        )
        object.__setattr__(self, "opts_defaults", opts_defaults)
        
        object.__setattr__(self, "raw_coords", as_points(coords))
        
        logger.detail('Handling explicit kwargs overrides ...')
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_internal_owner_ref", weakref.ref(self))
        object.__setattr__(self, "opts", opts)
        
        logger.detail("Establishing PlotFigure object ...")
        if figure is not None:
            try:
                if not isinstance(figure, PlotFigure):
                    raise TypeError('`figure` for plotting must be PlotFigure object!')
                else:
                    if not figure:
                        raise RuntimeError("The plotting window has been closed. Cannot update an inactive plotter.") 
            except (TypeError, RuntimeError):
                logger.exception("Check input")
                logger.recovery("Create a new PlotFigure object and store it in self._owner")
                figure = PlotFigure()
        elif figure is None:
            figure = PlotFigure()
        object.__setattr__(self, "_internal_owner_ref", weakref.ref(figure))
        
        
        logger.detail('Checking if name already exists ...')
        name_set = set(figure.act_get_entity_names())
        name_input = self.opts.name
        if name_input in name_set:
            new_name = name_input
            index = 1
            while new_name in name_set:
                new_name = f"{name_input}_{index}"
                index += 1
            object.__setattr__(opts, 'name', new_name)
            logger.warning(f"{name_input!r} already exists in PlotFigure object! Renamed to {opts.name!r}.")
            

        logger.detail("Examining the options before plotting ...")
        self.opts.act_finalize(self.opts_defaults)
        str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        unique_id = opts.name + str_now
        object.__setattr__(self, "_internal_name_pv", unique_id)
        
        if not (isinstance(opts.color, str) and opts.color == 'scalars') and opts.scalars not in (None, UNSET):
            msg = "Color input of PlotTube is not set to 'scalars'. However, scalars is provided.\n"
            msg += "The scalars data will be ignored unless color='scalars' is explicitly specified."
            logger.warning(msg)
            
    def _helper_init_end(self):
        figure = self.owner
        figure.pl.render()
        figure.pl.show(interactive_update=True)
        object.__setattr__(self.opts, '_state_is_category_locked', True)
        figure._helper_register_entity(self, self.opts.category, self.opts.is_reset_camera)
