from dataclasses import dataclass, field, fields
from typing import Callable, Sequence, Literal, Any, Mapping
import numpy as np
import pyvista as pv
import datetime
import weakref
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    as_ColorRGB_array,
    as_Number,
    as_str,
    as_bool,
    Vect,
    as_Vect,
    as_points,
    UNSET,
    Unset
)
from .plot_figure import PlotFigure
from ..opts import merge_opts_all
from Nematics3D.general import pop_exclusive



LEVEL_ACTOR  = 0  # Only changes GPU/Rendering state. (Fastest)
LEVEL_RECALC = 1  # Needs to re-calculate data arrays (colors, etc.) but keeps geometry.
LEVEL_REMESH = 2  # Needs to re-run the points filter to rebuild the 3D mesh. (Heaviest)

ATTR_MAP = {
    # === Visibility & Global Settings ===
    "name":                 (LEVEL_ACTOR,  None,                    "Identifier for the actor in the plotter."),
    "category":             (LEVEL_ACTOR,  None,                    "The semantic category of this plotting entity."),
    "is_visible":           (LEVEL_ACTOR,  "visibility",            "Whether the points is visible in the scene."),
    "shading_type":         (LEVEL_ACTOR,  "prop.interpolation",    "'phong', 'pbr' (Physical)"),
    "is_reset_camera":      (LEVEL_ACTOR,  None,                    "Whether to reset the camera settings for each (re-)plot."),

    # === Lighting - Phong ===
    "ambient":              (LEVEL_ACTOR,  "prop.ambient",          "Reflected light from environment (0-1)."),
    "diffuse":              (LEVEL_ACTOR,  "prop.diffuse",          "Standard matte reflection (0-1)."),
    "specular":             (LEVEL_ACTOR,  "prop.specular",         "Glossy highlight strength (0-1)."),
    "specular_pow":         (LEVEL_ACTOR,  "prop.specular_power",   "Focus of gloss (1-100). Higher = shinier/smaller spot."),
    "specular_color":       (LEVEL_ACTOR,  "prop.specular_color",   "The color of the glossy highlight (RGB). Usually white [1,1,1]."),
    
    # === Lighting - PBR ===
    "metallic":             (LEVEL_ACTOR,  "prop.metallic",         "PBR metallic effect (0-1). Needs PBR enabled."),
    "roughness":            (LEVEL_ACTOR,  "prop.roughness",        "PBR surface roughness (0-1). Needs PBR enabled."),

    # === Shape and Color Control ===
    "color":                (LEVEL_RECALC, None,                    ("Determines point colors. Options: "
                                                                    "1) ColorRGB for all points (e.g. (1,0,0))"
                                                                    "2) Function (mapping function), "
                                                                    "3) color data set manually, "
                                                                    "4) 'scalars' (maps 1D data to colors using scalars_cmap/scalars_clim).")),
    
    "opacity":              (LEVEL_RECALC, None,                    ("Determines point transparency. Options: "
                                                                    "1) float 0-1 for all points, "
                                                                    "2) Function (mapping function), "
                                                                    "3) opacity data set manually.")),
    
    "scalars":              (LEVEL_RECALC, None,                    ("Determines point scalars. Options: "
                                                                    "1) Function (mapping function), "
                                                                    "2) scalars data set manually, "
                                                                    "3) None (No scalars)")),
    
    "radius":               (LEVEL_REMESH, None,                    ("Determines tube thickness. Options: "
                                                                    "1) float for all points, "
                                                                    "2) Function (mapping function), "
                                                                    "3) radius data set manually.")),
    
    # === Scalars Control (Needs color_rule='scalars') ===
    "scalars_cmap":         (LEVEL_RECALC, None,                    "Colormap name (e.g., 'viridis') used if color is set to scalar."),
    "scalars_clim":         (LEVEL_RECALC, None,                    "Color limits [min, max] for scalar mapping."),
    "is_scalar_bar":        (LEVEL_ACTOR,  None,                    "Whether to display the color legend (scalar bar)."),
    "scalar_bar_title":     (LEVEL_ACTOR,  None,                    "Title for the scalar bar (e.g., 'Stress (MPa)')."),

    # === Geometry & Topology (LEVEL_REMESH) ===
    "resolution":           (LEVEL_REMESH, None,                    "The subdivision level of the sphere mesh. "
                                                                    "A higher value produces a smoother surface by increasing the number of polygonal faces"),
    
    # === Advanced Spatial Clipping ===
    "clip_geometry":        (LEVEL_REMESH, None,                    "Define clipping boundary. Can be: "
                                                                    "1) List of 6 floats [xmin, xmax...] for axis-aligned box, "
                                                                    "2) A Mesh/PolyData representing any closed shape (e.g. 8-point box).")
}

# --- Type aliases ---
ColorMode = ColorRGB | Callable | Sequence | Literal["scalars"]
OpacityMode = float | Callable | Sequence
RadiusMode = float | Callable | Sequence
ScalarsMode = Callable | Sequence | None
ClipGeometryLike = list[float] | pv.PolyData | None

@dataclass(slots=True)
class OptsSphere:
    """
    Options for rendering a sphere object.

    This class supports a two-phase lifecycle:
      (1) Configuration phase: many fields may remain UNSET.
      (2) Finalization phase: act_finalize() replaces UNSET fields using defaults,
          validates them, and freezes the opts for use by an owner.
    """

    # -------------------------------------------------------------------------
    # Frozen defaults (read-only, global baseline)
    #
    # - Must contain ALL public fields that are expected to be finalized.
    # - Should be treated as immutable. MappingProxyType prevents accidental edits.
    # - act_finalize() will fill UNSET fields from the provided defaults mapping
    #   first, then fall back to this frozen table.
    # -------------------------------------------------------------------------
    _DEFAULTS_FROZEN = MappingProxyType({
        # --- Visibility & Global ---
        "name":                 "sphere",
        "category":             "point",
        "is_visible":           True,
        "shading_type":         "phong",
        "is_reset_camera":      True,

        # --- Phong Lighting ---
        "ambient":              0.0,
        "diffuse":              1.0,
        "specular":             1.0,
        "specular_pow":         10.0,
        "specular_color":       (1.0, 1.0, 1.0),

        # --- PBR Lighting ---
        "metallic":             0.0,
        "roughness":            0.5,

        # --- Shape & Color ---
        "color":                (0.5, 0.5, 0.5),
        "opacity":              1.0,
        "radius":               0.5,
        "scalars":              None,

        # --- Scalars (used if color == "scalars") ---
        "scalars_cmap":         "viridis",
        "scalars_clim":         None,
        "is_scalar_bar":        True,
        "scalar_bar_title":     "scalars",

        # --- Geometry & Clipping ---
        "resolution":           50,
        "clip_geometry":        None,
    })
    
    # --- Visibility & Global ---
    name: str | Unset = UNSET
    category: str | Unset = UNSET
    is_visible: bool | Unset = UNSET
    shading_type: Literal["phong", "pbr"] | Unset = UNSET
    is_reset_camera: bool | Unset = UNSET

    # --- Phong Lighting ---
    ambient: float | Unset = UNSET
    diffuse: float | Unset = UNSET
    specular: float | Unset = UNSET
    specular_pow: float | Unset = UNSET
    specular_color: ColorRGB | Unset = UNSET

    # --- PBR Lighting ---
    metallic: float | Unset = UNSET
    roughness: float | Unset = UNSET

    # --- Shape & Color ---
    color: ColorMode | Unset = UNSET
    opacity: OpacityMode | Unset = UNSET
    scalars: ScalarsMode | Unset = UNSET
    radius: RadiusMode | Unset = UNSET

    # --- Scalars (used if color == "scalars") ---
    scalars_cmap: str | Unset = UNSET
    scalars_clim: Vect(2) | None | Unset = UNSET
    is_scalar_bar: bool | Unset = UNSET
    scalar_bar_title: str | Unset = UNSET

    # --- Geometry & Clipping ---
    resolution: int | Unset = UNSET
    clip_geometry: ClipGeometryLike | Unset = UNSET
    
    # --- Internal State (not part of defaults/finalization) ---
    _state_is_category_locked: bool = field(default=False, init=False, repr=False)
    _state_functioning: bool = field(default=False, init=False, repr=False)
    _defaults: dict[str, Any] = field(init=False, repr=False)

    _internal_owner: object | None = field(default=None, repr=False, init=False)
    
    _validators = {
        "name": lambda self, v, d: as_str(v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["name"]),
        "category": lambda self, v, d: as_str(v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["category"]),
        "is_visible": lambda self, v, d: as_bool(v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["is_visible"]),
        "shading_type": lambda self, v, d: as_str(
            v, name=d,
            replace=OptsSphere._DEFAULTS_FROZEN["shading_type"],
            pool=("phong", "pbr"),
        ),
        "is_reset_camera": lambda self, v, d: as_bool(v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["is_reset_camera"]),
    
        "ambient": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsSphere._DEFAULTS_FROZEN["ambient"]
        ),
        "diffuse": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsSphere._DEFAULTS_FROZEN["diffuse"]
        ),
        "specular": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsSphere._DEFAULTS_FROZEN["specular"]
        ),
        "specular_pow": lambda self, v, d: as_Number(
            v, name=d, value_range=(1, 100), bounded=True, replace=OptsSphere._DEFAULTS_FROZEN["specular_pow"]
        ),
        "specular_color": lambda self, v, d: as_ColorRGB(
            v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["specular_color"]
        ),
    
        "metallic": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsSphere._DEFAULTS_FROZEN["metallic"]
        ),
        "roughness": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsSphere._DEFAULTS_FROZEN["roughness"]
        ),
    
        "scalars_cmap": lambda self, v, d: as_str(v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["scalars_cmap"]),
        "scalars_clim": lambda self, v, d: (
            v if v is None else as_Vect(v, name=d, dim=2, replace=OptsSphere._DEFAULTS_FROZEN["scalars_clim"])
        ),
        "is_scalar_bar": lambda self, v, d: as_bool(v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["is_scalar_bar"]),
        "scalar_bar_title": lambda self, v, d: as_str(v, name=d, replace=OptsSphere._DEFAULTS_FROZEN["scalar_bar_title"]),
    
        "resolution": lambda self, v, d: as_Number(
            v, name=d, is_int=True, value_range=(3, np.inf), bounded=True, replace=OptsSphere._DEFAULTS_FROZEN["resolution"]),
    }

    def __post_init__(self):
        # Instance-level copy (mutable), useful for debugging or transitional logic.
        # The canonical baseline remains _DEFAULTS_FROZEN.
        object.__setattr__(self, "_defaults", dict(self._DEFAULTS_FROZEN))
        
    def __setattr__(self, key, value):

        if value is not UNSET and key in self._validators:
            desc = f'{key!r}: {ATTR_MAP.get(key)[2]}'
            value = self._validators[key](self, value, desc)
            
        if getattr(self, "_state_is_category_locked", False) and key == "category":
            raise AttributeError("Modification of 'category' is not allowed, because it is used as the key in dir: PlotFigure._entities")

        object.__setattr__(self, key, value)
        
        if key != "_internal_owner" and getattr(self, "_state_functioning", False) and self._internal_owner is not None:
            self._internal_owner.act_commit(**{key: value}, is_setattr=False)
            
    def act_finalize(self, defaults: Mapping[str, Any] | None = None):
        """
        Resolve all UNSET fields using:
          1) the provided `defaults` mapping (higher priority), then
          2) the class-level `_DEFAULTS_FROZEN` mapping.

        This must be called before visualization. After finalization, the opts
        should be treated as ready-to-use (no more defaults resolution).
        """
        if getattr(self, "_state_functioning", False):
            raise RuntimeError("OptsSphere has already been finalized.")

        defaults = {} if defaults is None else dict(defaults)

        for f in fields(self):
            k = f.name
            if k.startswith("_"):
                continue  # internal fields are not finalized

            if getattr(self, k) is UNSET:
                v = defaults.get(k, self._DEFAULTS_FROZEN.get(k, UNSET))
                if v is UNSET:
                    raise KeyError(f"Missing default for field {k!r}.")
                setattr(self, k, v)  # runs validators

        object.__setattr__(self, "_state_functioning", True)
        

class PlotSphere:
    """
    Wraps PyVista sphere filtering and rendering with integrated option management.
    """
    __descriptions__ = {
        "raw_coords": "The N x 3 input coordinates. ",
    
        "_calc_point_cloud": (
            "The generated PyVista PolyData representing the points "
            "before applying the sphere filter."
        ),
        
        "_calc_glyph_mesh": (
            "The generated mesh where each point is represented by a 3D shpere (glyph) "
            "before applying the sphere filter."
        ),
    
        "_calc_color": "The resolved per-point RGB color array of the points.",
        "_calc_opacity": "The resolved per-point opacity array of the points.",
        "_calc_radius": "The resolved per-point radius array used for points radii.",
        "_calc_scalars": "The resolved per-point scalar array used for scalar coloring.",
    
        "_entities": "The PyVista Actor corresponding to these oints in the plotter.",
        
        
        "opts": "The OptsSphere instance controlling rendering and geometry options.",
        "opts_defaults": "The default option settings for tube visualization",
        
        
        "_internal_owner_ref": ("A weak reference to the PlotFigure object associated with these points."
                                "To access it, use .owner or ._internal_owner."),
        "_internal_name_pv": "The unique identifier of these points stored in the PyVista plotter.",
        
        # --- user-defined attributes (extension mechanism) ---
        "_internal_extra_attrs": (
            "A dict storing user-registered extra attributes. "
            "These are accessed via `points.<name>` after calling `act_add_attr(name, doc)`."
        ),
        "_internal_extra_attrs_docs": (
            "A dict storing docstrings for user-registered extra attributes."
        ),
    }
    
    __slots__ = tuple(__descriptions__.keys())
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        figure: PlotFigure | None = None,
        opts: OptsTube | None = None,
        line_index: Sequence | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
        **kwargs
    ):
        
        if opts_defaults_override is None:
            opts_defaults_override = {}
        opts_defaults = dict(OptsTube._DEFAULTS_FROZEN)
        for k, v in opts_defaults_override.items():
            if k not in opts_defaults:
                raise KeyError(
                    f"Invalid key {k!r} in opts_defaults_override; "
                    f"not a valid OptsTube option."
                )
            opts_defaults[k] = v
        object.__setattr__(self, "opts_defaults", opts_defaults)
        
        
        object.__setattr__(self, "raw_coords", as_points(coords))
    
    










import pyvista as pv
import numpy as np

# 1. 准备点数据
points = np.random.rand(10, 3)
pd = pv.PolyData(points)

# 2. 准备属性数组
pd["my_colors"] = np.arange(10)      # 控制颜色
pd["my_sizes"] = np.random.rand(10)  # 控制球的大小
pd["my_opac"] = np.linspace(0.2, 1, 10) # 控制透明度

# 3. 创建 Glyph (把球体模板按照 my_sizes 缩放并放置在 pd 的点上)
# scale="my_sizes" 告诉 PyVista：请用这个数组来决定每个球的大小
glyph_mesh = pd.glyph(geom=pv.Sphere(theta_resolution=50, phi_resolution=50), scale="my_sizes", orient=False)

# 4. 绘图
pl = pv.Plotter()
actor = pl.add_mesh(glyph_mesh, 
            scalars="my_colors", 
            opacity="my_opac", 
            cmap="viridis")
pl.show(interactive_update=True)