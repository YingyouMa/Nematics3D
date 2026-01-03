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

#! scalars_limit
#! scalars_bar
#! clip_geometry
#! light dark pbr

#! only change cmap

#! info log extra attr
#1 del
#! orphan figure

#! @coords


LEVEL_ACTOR  = 0  # Only changes GPU/Rendering state. (Fastest)
LEVEL_RECALC = 1  # Needs to re-calculate data arrays (colors, etc.) but keeps geometry.
LEVEL_REMESH = 2  # Needs to re-run the tube filter to rebuild the 3D mesh. (Heaviest)

ATTR_MAP = {
    # === Visibility & Global Settings ===
    "name":                 (LEVEL_ACTOR,  None,                    "Identifier for the actor in the plotter."),
    "category":             (LEVEL_ACTOR,  None,                    "The semantic category of this plotting entity."),
    "is_visible":           (LEVEL_ACTOR,  "visibility",            "Whether the tube is visible in the scene."),
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
                                                                    "1) ColorRGB for entire tube (e.g. (1,0,0))"
                                                                    "2) Function (mapping function), "
                                                                    "3) color data set manually, "
                                                                    "4) 'scalars' (maps 1D data to colors using scalars_cmap/scalars_clim).")),
    
    "opacity":              (LEVEL_RECALC, None,                    ("Determines point transparency. Options: "
                                                                    "1) float 0-1 for entire tube, "
                                                                    "2) Function (mapping function), "
                                                                    "3) opacity data set manually.")),
    
    "scalars":              (LEVEL_RECALC, None,                    ("Determines point scalars. Options: "
                                                                    "1) Function (mapping function), "
                                                                    "2) scalars data set manually, "
                                                                    "3) None (No scalars)")),
    
    "radius":               (LEVEL_REMESH, None,                    ("Determines tube thickness. Options: "
                                                                    "1) float for entire tube, "
                                                                    "2) Function (mapping function), "
                                                                    "3) radius data set manually.")),
    
    # === Scalars Control (Needs color_rule='scalars') ===
    "scalars_cmap":         (LEVEL_RECALC, None,                    "Colormap name (e.g., 'viridis') used if color is set to scalar."),
    "scalars_clim":         (LEVEL_RECALC, None,                    "Color limits [min, max] for scalar mapping."),
    "is_scalar_bar":        (LEVEL_ACTOR,  None,                    "Whether to display the color legend (scalar bar)."),
    "scalar_bar_title":     (LEVEL_ACTOR,  None,                    "Title for the scalar bar (e.g., 'Stress (MPa)')."),

    # === Geometry & Topology (LEVEL_REMESH) ===
    "sides":                (LEVEL_REMESH, None,                    "Number of facets around the tube (higher = smoother)."),
    "is_capping":           (LEVEL_REMESH, None,                    "Whether to close the ends of the tube."),
    "smooth_iter":          (LEVEL_REMESH, None,                    "Path smoothing iterations to remove jagged edges."),
    
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
class OptsTube:
    """
    Options for rendering a tube-like polyline object.

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
        "name":                 "tube",
        "category":             "line",
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
        "sides":                6,
        "is_capping":           True,
        "smooth_iter":          0,
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
    sides: int | Unset = UNSET
    is_capping: bool | Unset = UNSET
    smooth_iter: int | Unset = UNSET
    clip_geometry: ClipGeometryLike | Unset = UNSET

    # --- Internal State (not part of defaults/finalization) ---
    _state_is_category_locked: bool = field(default=False, init=False, repr=False)
    _state_functioning: bool = field(default=False, init=False, repr=False)
    _defaults: dict[str, Any] = field(init=False, repr=False)

    _internal_owner: object | None = field(default=None, repr=False, init=False)
    
    
    _validators = {
        "name": lambda self, v, d: as_str(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["name"]),
        "category": lambda self, v, d: as_str(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["category"]),
        "is_visible": lambda self, v, d: as_bool(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["is_visible"]),
        "shading_type": lambda self, v, d: as_str(
            v, name=d,
            replace=OptsTube._DEFAULTS_FROZEN["shading_type"],
            pool=("phong", "pbr"),
        ),
        "is_reset_camera": lambda self, v, d: as_bool(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["is_reset_camera"]),
    
        "ambient": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["ambient"]
        ),
        "diffuse": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["diffuse"]
        ),
        "specular": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["specular"]
        ),
        "specular_pow": lambda self, v, d: as_Number(
            v, name=d, value_range=(1, 100), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["specular_pow"]
        ),
        "specular_color": lambda self, v, d: as_ColorRGB(
            v, name=d, replace=OptsTube._DEFAULTS_FROZEN["specular_color"]
        ),
    
        "metallic": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["metallic"]
        ),
        "roughness": lambda self, v, d: as_Number(
            v, name=d, value_range=(0, 1), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["roughness"]
        ),
    
        "scalars_cmap": lambda self, v, d: as_str(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["scalars_cmap"]),
        "scalars_clim": lambda self, v, d: (
            v if v is None else as_Vect(v, name=d, dim=2, replace=OptsTube._DEFAULTS_FROZEN["scalars_clim"])
        ),
        "is_scalar_bar": lambda self, v, d: as_bool(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["is_scalar_bar"]),
        "scalar_bar_title": lambda self, v, d: as_str(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["scalar_bar_title"]),
    
        "sides": lambda self, v, d: as_Number(
            v, name=d, is_int=True, value_range=(3, 128), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["sides"]
        ),
        "is_capping": lambda self, v, d: as_bool(v, name=d, replace=OptsTube._DEFAULTS_FROZEN["is_capping"]),
        "smooth_iter": lambda self, v, d: as_Number(
            v, name=d, is_int=True, value_range=(0, 1000), bounded=True, replace=OptsTube._DEFAULTS_FROZEN["smooth_iter"]
    )}
    
    
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
            raise RuntimeError("OptsTube has already been finalized.")

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

        
        
class PlotTube:
    """
    Wraps PyVista tube filtering and rendering with integrated option management.
    """
    __descriptions__ = {
        "raw_coords": (
            "The N x 3 input coordinates. "
            "Points are ordered along polylines but may belong to multiple disconnected lines."
        ),
    
        "raw_line_index": (
            "Optional integer array of length N specifying polyline membership for each point. "
            "Points with the same index and appearing consecutively form a single connected polyline. "
            "If None or constant, the input is treated as a single continuous line."
        ),
    
        "_calc_poly": (
            "The generated PyVista PolyData representing the polyline(s) "
            "before applying the tube filter."
        ),
    
        "_calc_color": "The resolved per-point RGB color array of the tube.",
        "_calc_opacity": "The resolved per-point opacity array of the tube.",
        "_calc_radius": "The resolved per-point radius array used for tube thickness.",
        "_calc_scalars": "The resolved per-point scalar array used for scalar coloring.",
    
        "_entities": "The PyVista Actor corresponding to this tube in the plotter.",
        
        
        "opts": "The OptsTube instance controlling rendering and geometry options.",
        "opts_defaults": "The default option settings for tube visualization",
        
        
        "_internal_owner_ref": ("A weak reference to the PlotFigure object associated with this tube."
                                "To access it, use .owner or ._internal_owner."),
        "_internal_name_pv": "The unique identifier of this tube stored in the PyVista plotter.",
        
        # --- user-defined attributes (extension mechanism) ---
        "_internal_extra_attrs": (
            "A dict storing user-registered extra attributes. "
            "These are accessed via `tube.<name>` after calling `act_add_attr(name, doc)`."
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
        opts: OptsTube = OptsTube(),
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
        
        if line_index is not None:
            try:
                line_index = self._helper_check_index(line_index)
            except:
                logger.exception("Invalid `line_index` input")
                logger.recovery("Set line_index=None in the following (no stop points within the tube)")
                line_index = None
        object.__setattr__(self, "raw_line_index", line_index)
    
        if figure is not None and not isinstance(figure, PlotFigure):
            try:
                raise TypeError('`figure` for PlotTube must be PlotFigure object!')
            except:
                logger.exception("Check input")
                logger.recovery("Create a new PlotFigure object and store it in self._owner")
                figure = PlotFigure()
        elif figure is None:
            figure = PlotFigure()
        object.__setattr__(self, "_internal_owner_ref", weakref.ref(figure))

        logger.detail('Handling explicit kwargs overrides')
        
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_internal_owner", self)
        
        object.__setattr__(self, "opts", opts)
        
        logger.detail('Checking if name already exists')
        name_set = set(figure.act_get_entities_names())
        name_input = self.opts.name
        if name_input in name_set:
            new_name = name_input
            index = 1
            while new_name in name_set:
                new_name = f"{name_input}_{index}"
                index += 1
            object.__setattr__(opts, 'name', new_name)
            logger.warning(f"{name_input!r} already exists in PlotFigure object! Renamed to {opts.name!r}.")
        

        logger.detail("Executing initial plot")
        self.opts.act_finalize(self.opts_defaults)
        str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        unique_id = opts.name + str_now
        object.__setattr__(self, "_internal_name_pv", unique_id)
        
        if not (isinstance(opts.color, str) and opts.color == 'scalars') and opts.scalars not in (None, UNSET):
            msg = "Color input of PlotTube is not set to 'scalars'. However, scalars is provided.\n"
            msg += "The scalars data will be ignored unless color='scalars' is explicitly specified."
            logger.warning(msg)
        self._helper_resolver_init()
        self._helper_make_figure()
        
        figure._obj_plotter.render()
        figure._obj_plotter.show(interactive_update=True)
        object.__setattr__(self.opts, '_state_is_category_locked', True)
        figure._helper_register_entity(self, self.opts.category, self.opts.is_reset_camera)
        
        object.__setattr__(self, "_internal_extra_attrs", {})
        object.__setattr__(self, "_internal_extra_attrs_docs", {})
        
    @property
    def _internal_owner(self):
        return self._internal_owner_ref()
    
    @property
    def _owner(self):
        return self._internal_owner_ref()
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_check_index(self, line_index, is_keep=False, logger=None):
        try:
            line_index = np.asarray(line_index, dtype=int)
            if line_index.ndim != 1 or len(line_index) != self.raw_coords.shape[0]:
                raise ValueError(
                    f"`line_index` must be a ({self.raw_coords.shape[0]},) array. "
                    f"Got shape {line_index.shape} instead."
                )
            return line_index
        except (ValueError, TypeError):
            raise
            
            
    def __setattr__(self, key, value):
    
        extra = object.__getattribute__(self, "_internal_extra_attrs")
        docs = object.__getattribute__(self, "_internal_extra_attrs_docs")
        if key in docs:
            extra[key] = value
            return
    
        allowed_core = ("raw_coords", "raw_line_index")
        if key not in allowed_core:
            raise AttributeError(
                f"Invalid attribute assignment: {key!r}. Only {allowed_core} can be modified directly, "
                f"or a registered extra attribute."
            )
        self.act_commit(**{key: value})
            
            
    def __getattr__(self, key):
        extra = object.__getattribute__(self, "_internal_extra_attrs")
        if key in extra:
            return extra[key]
        else:
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {key!r}.")
            
        

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_generic(self, attr_name, attr_input, default_val, logger=None):
        
        target_shape = (len(self.raw_coords),3) if attr_name=='color' else (len(self.raw_coords),)
        
        try:
            if attr_input is None:
                raise TypeError(f"Require input for {attr_name!r}. Got None instead.")
            elif callable(attr_input):
                resolved = np.asarray(attr_input(self.raw_coords), dtype=np.float32)
            else:
                arr = np.asarray(attr_input, dtype=float)
                if arr.shape == () and attr_name == 'color':
                    raise TypeError(f"To provide a single value for color, the input should be expressed by (R, G, B). Got {attr_input} instead.")
                resolved = np.full(target_shape, arr, dtype=np.float32)
    
            if resolved.shape != target_shape:
                raise ValueError(
                    f"Shape mismatch for {attr_name!r}: got {resolved.shape}, expected {target_shape}."
                )
                
            if attr_name == 'color':
                resolved = as_ColorRGB_array(resolved, name='The pairwise color data of tube', replace=default_val)
    
    
        except:
            logger.exception(f"Failed to resolve {attr_name!r}")
            resolved = np.full(target_shape, default_val, dtype=np.float32)
            logger.recovery(f"Reset {attr_name!r} to default: {default_val} everywhere.")
            object.__setattr__(self.opts, attr_name, default_val)
              
        object.__setattr__(self, '_calc_'+attr_name, resolved)
            
            
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_init(self, logger=None):
        logger.detail("Resolving data for color, opacity and radius")
        self._helper_resolver_spec('opacity')
        self._helper_resolver_spec('radius')
        
        if isinstance(self.opts.color, str) and self.opts.color == 'scalars':
            self._helper_resolver_spec('scalars')
        else:
            self._helper_resolver_spec('color')

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_spec(self, attr_name, logger=None):
        
        if attr_name not in ['color', 'radius', 'scalars', 'opacity']:
            raise ValueError(f"Attribute resolved by `_helper_resolver_spec()` must be in ['color', 'radius', 'scalars', 'opacity']. Got {attr_name} instead.")
        
        self._helper_resolver_generic(attr_name, getattr(self.opts, attr_name), self.opts_defaults[attr_name])
        
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_build_tube_mesh(self, logger=None):
        """
        Internal: Create the PyVista PolyData, apply smoothing/clipping, 
        and generate tube with dynamic or static radius.
        """
        points = self.raw_coords
        idx = getattr(self, "raw_line_index", None)
        
        # Decide whether to treat the input as a single continuous polyline
        is_use_multi = (idx is None) or (len(np.unique(idx)) == 1)
        if is_use_multi:
            poly = pv.MultipleLines(points)
        else:
            logger.detail('Searching run boundaries: each run corresponds to one disconnected polyline')
            breaks = np.nonzero(idx[1:] != idx[:-1])[0] + 1
            starts = np.r_[0, breaks]
            ends   = np.r_[breaks, len(idx)]
        
            chunks = []
            for s, e in zip(starts, ends):
                k = e - s
                if k < 2:
                    msg = 'Detect one invalid line segment with only one point. Ignore it in the following.'
                    logger.warning(msg)
                chunks.append(np.r_[k, np.arange(s, e, dtype=np.int64)])
        
            if len(chunks) == 0:
                raise ValueError("line_index produced no valid line segments (each segment needs >=2 points).")
        
            lines = np.concatenate(chunks).astype(np.int64)
            poly = pv.PolyData(points, lines=lines)
        
        if self.opts.smooth_iter > 0:
            logger.detail(f"Smoothing path with {self.opts.smooth_iter} iterations")
            poly = poly.smooth(n_iter=self.opts.smooth_iter)
        
        poly.point_data['radius'] = self._calc_radius 
        if isinstance(self.opts.color, str) and self.opts.color == 'scalars':
            poly.point_data['opacity'] = self._calc_opacity
            poly.point_data['scalars'] = self._calc_scalars
        else:
            rgba_values = np.hstack([self._calc_color, self._calc_opacity.reshape(-1, 1)])
            poly.point_data['rgba'] = rgba_values 
            

        logger.detail("Applying tube filter with dynamic radius scaling")
        mesh = poly.tube(
            scalars='radius', 
            n_sides=self.opts.sides, 
            capping=self.opts.is_capping,
            absolute=True 
        )

        if self.opts.clip_geometry is not None:
            logger.detail("Applying spatial clipping to tube mesh")
            if isinstance(self.opts.clip_geometry, (list, tuple)) and len(self.opts.clip_geometry) == 6:
                mesh = mesh.clip_box(bounds=self.opts.clip_geometry, invert=False)
            elif hasattr(self.opts.clip_geometry, "points"):
                mesh = mesh.clip_surface(self.opts.clip_geometry, invert=False)

        object.__setattr__(self, "_calc_poly", poly)
        return mesh
    
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_make_figure(self, logger=None):
        """
        Creates or updates the rendering in a PyVista Plotter.
        """
        
        is_scalars = (isinstance(self.opts.color, str) and self.opts.color == 'scalars')
        unique_id = self._internal_name_pv
        
        input_dir = {
            "name":         unique_id,
            "pbr":          self.opts.shading_type == 'pbr',
            "rgb":          not is_scalars,
            "scalars":      'scalars' if is_scalars else 'rgba',
            "reset_camera": self.opts.is_reset_camera
            }
        if is_scalars:
            input_dir["opacity"] = "opacity"
            input_dir["cmap"] = self.opts.scalars_cmap
            input_dir["show_scalar_bar"] = self.opts.is_scalar_bar
            input_dir["scalar_bar_args"] = {"title": self.opts.scalar_bar_title}
            input_dir["clim"] = self.opts.scalars_clim
            
        logger.detail("Creating tube mesh")
        mesh = self._helper_build_tube_mesh()
            
        logger.detail("Visualizing the tube")
        plotter = self._internal_owner._obj_plotter
        if unique_id in plotter.actors:
            plotter.remove_actor(unique_id)
        actor = plotter.add_mesh(mesh, **input_dir)
        
        logger.detail("Applying detailed rendering properties directly to the Actor's property object")
        
        prop = actor.prop
        
        shading = self.opts.shading_type.lower()
        if shading not in ('pbr', 'phong'):
            try:
                raise ValueError("shading type must either be `pbr` or `phong`")
            except ValueError:
                logger.exception("Please check input")
                logger.recovery("Use `phong` in the following.")
                shading = 'phong'
        prop.interpolation = shading
        object.__setattr__(self.opts, 'shading_type', shading)
            
        prop.ambient = self.opts.ambient
        prop.diffuse = self.opts.diffuse
        prop.specular = self.opts.specular
        prop.specular_power = self.opts.specular_pow
        prop.specular_color = self.opts.specular_color
        
        if shading == 'pbr':
            prop.metallic = self.opts.metallic
            prop.roughness = self.opts.roughness
            
        actor.visibility = self.opts.is_visible

        object.__setattr__(self, "_entities", actor)
        
        
    def _helper_replace_data_pv(self, attr: str, data: np.ndarray):
        mesh = self._entities.mapper.dataset
        mesh_data = mesh.point_data
        if attr in self._calc_poly.point_data:
            del self._calc_poly.point_data[attr]
        if attr in mesh_data:
            del mesh_data[attr]
        self._calc_poly.point_data[attr] = data
        mesh_data[attr] = mesh.interpolate(self._calc_poly).point_data[attr]
    
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_update_rgba(self, logger=None):
        rgba = np.hstack([self._calc_color, self._calc_opacity.reshape(-1, 1)])
        self._helper_replace_data_pv('rgba', rgba)
        mapper = self._entities.mapper
        mapper.scalar_visibility = True
        mapper.color_mode = 'direct'
        mapper.lookup_table = None
        mapper.dataset.set_active_scalars('rgba')
        mapper.SetArrayName('rgba')
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_update_scalars(self, logger=None):
        logger.detail("Update scalar coloring, which may involve switching from a direct color-based scheme to scalar-based coloring.")

        self._helper_replace_data_pv('scalars', self._calc_scalars)
        self._helper_replace_data_pv('opacity', self._calc_opacity)
        
        mapper = self._entities.mapper
        mesh_data = mapper.dataset.point_data

        if "__custom_rgba" in mesh_data.keys():
            mesh_data.remove("__custom_rgba")
        
        mapper.set_scalars(
            mesh_data['scalars'], 
            'scalars',
            cmap = self.opts.scalars_cmap,
            clim = self.opts.scalars_clim,
            custom_opac=True,
            opacity=mesh_data['opacity'])
        
        object.__setattr__(self.opts, 'color', 'scalars')
        
        
    @logging_and_warning_decorator()
    def act_commit(self, is_setattr=True, logger=None, **kwargs):
        
        if not kwargs:
            return
    
        is_needs_remesh = False
        
        found, coords = pop_exclusive(kwargs, "coords", "raw_coords")
        if found:
            try:
                object.__setattr__(self, "raw_coords", as_points(coords))
                is_needs_remesh = True
            except:
                logger.exception("Invalid input of coords for PlotTube.")
                logger.recovery("Ignore this modification in the following")
        
        found, line_index = pop_exclusive(kwargs, "line_index", "raw_line_index")
        if found:
            if line_index is None:
                object.__setattr__(self, "raw_line_index", line_index)
                is_needs_remesh = True
            else:
                try:
                    line_index = self._helper_check_index(line_index)
                    object.__setattr__(self, "raw_line_index", line_index)
                    is_needs_remesh = True
                except:
                    logger.exception("Invalid `line_index` input")
                    logger.recovery("Ignore this modification in the following")
                
        if is_needs_remesh:
            for attr in ['radius', 'color', 'opacity']:
                if attr not in kwargs.keys():
                    if attr == 'color' and isinstance(self.opts.color, str):
                        self._helper_resolver_spec('scalars')
                    else:
                        self._helper_resolver_spec(attr)
        
        current_shading = kwargs.get("shading_type", getattr(self.opts, "shading_type"))
        current_shading = as_str(current_shading, name='shading_type', replace=getattr(self.opts, "shading_type"), pool=('phong', 'pbr'))
        
        color_method = None
        if 'scalars' in kwargs.keys():
            if 'color' in kwargs.keys():
                msg = ("You are attempting to modify both 'color' and 'scalars' simultaneously."
                       "This is a potentially confusing operation."
                       "The values will be updated accordingly, but rendering will use 'scalars' for coloring.")
                logger.warning(msg)
            color_method = 'scalars'
        elif 'color' in kwargs.keys():
            color_method = 'color'
        elif 'opacity' in kwargs.keys():
            color_method = 'scalars' if self.opts.color == 'scalars' else 'color'


        for key, value in kwargs.items():
            
            try:
                if key not in ATTR_MAP:
                        raise ValueError(f"Unknown attribute: {key} in class: PlotTube.opts")
                        
                if is_setattr and key != "category":
                    object.__setattr__(self.opts, key, value)
    
                level, attr_path_actor, doc = ATTR_MAP[key]
    
                # Dealing with LEVEL ACTOR (simply resetting values)
                if level == LEVEL_ACTOR:
                    
                    if key == "category":
                        raise AttributeError("Modification of 'category' is not allowed, because it is used as the key in dir: PlotFigure._entities")
                    
                    if key == "name":
                        msg = "Changing 'name' of PlotTube object is not recommended because: \n"
                        msg += "1) There is no guarantee that name collisions will be avoided in PlotFigure._entities; and\n"
                        msg += "2) The corresponding actor name stored in the PyVista renderer cannot be updated accordingly."
                        logger.warning(msg)
                    
                    # if key in "is_reset_camera":
                    # if key in ["is_visible", "shading_type"]:
    
                    pbr_params = ["metallic", "roughness"]
                    phong_params = ["ambient", "diffuse", "specular", "specular_pow", "specular_color"]
                    
                    if key in pbr_params and current_shading != "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is '{current_shading}'. PBR effects may not show.")
                    elif key in phong_params and current_shading == "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is 'pbr'. Phong lighting parameters may be ignored.")
    
                    if attr_path_actor and not is_needs_remesh:
                        parts = attr_path_actor.split('.')
                        obj = self._entities
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value)
                
                # Dealing with LEVEL_RECALC (resolver for color, opacity and scalars)
                elif level == LEVEL_RECALC:
                    self._helper_resolver_spec(key)
    
                # Dealing with LEVEL_REMESH (Geometry)
                elif level == LEVEL_REMESH:
                    is_needs_remesh = True
                    if key == 'radius':
                        self._helper_resolver_spec('radius')
        
            except:
                logger.exception(f"Failed to reset value of {key!r}")
                logger.recovery("Ignore this modification")
                
        if is_needs_remesh:
            self._helper_make_figure()
        else:
            if color_method == 'scalars':
                self._helper_update_scalars()
            elif color_method == 'color':
                self._helper_update_rgba()
                
        self._internal_owner._obj_plotter.render()
        
    @logging_and_warning_decorator(start_finish_level=5)
    def act_add_attr(
        self,
        name: str,
        doc: str,
        default=None,
        overwrite: bool = False,
        logger=None,
    ):

        name = as_str(name, name='Extra attribute name for PlotTube')
        doc = as_str(doc, name='Extra attribute doc for PlotTube')


        if not name.isidentifier():
            raise ValueError(f"Invalid extra attribute name {name!r}: must be a valid Python identifier.")

        if hasattr(type(self), name) or (name in getattr(type(self), "__slots__", ())):
            raise AttributeError(
                f"Cannot register extra attribute {name!r}: it conflicts with an existing attribute of {type(self).__name__}."
            )

        docs = self._internal_extra_attrs_docs
        data = self._internal_extra_attrs

        if (name in docs) and (not overwrite):
            raise KeyError(
                f"Extra attribute {name!r} is already registered. Use overwrite=True to override."
            )

        docs[name] = doc
        if overwrite or (name not in data):
            data[name] = default

            
                
        
