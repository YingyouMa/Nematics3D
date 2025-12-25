from dataclasses import dataclass, field, asdict
from typing import Union, Optional, Callable, List
import numpy as np
import pyvista as pv

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    as_ColorRGB_array,
    Number,
    as_Number,
    as_str,
    as_bool
)
from ..opts import merge_opts_all

#! scalars_limit
#! scalars_bar
#! clip_geometry
#! light dark pbr


LEVEL_ACTOR  = 0  # Only changes GPU/Rendering state. (Fastest)
LEVEL_RECALC = 1  # Needs to re-calculate data arrays (colors, etc.) but keeps geometry.
LEVEL_REMESH = 2  # Needs to re-run the tube filter to rebuild the 3D mesh. (Heaviest)

ATTR_MAP = {
    # === Visibility & Global Settings ===
    "name":                 (LEVEL_ACTOR,  None,                    "Unique identifier for the actor in the plotter."),
    "is_visible":           (LEVEL_ACTOR,  "visibility",            "Whether the tube is visible in the scene."),
    "shading_type":         (LEVEL_ACTOR,  "prop.interpolation",     "'phong', 'pbr' (Physical)"),

    # === Legacy Lighting - Phong ===
    "ambient":              (LEVEL_ACTOR,  "prop.ambient",          "Reflected light from environment (0-1)."),
    "diffuse":              (LEVEL_ACTOR,  "prop.diffuse",          "Standard matte reflection (0-1)."),
    "specular":             (LEVEL_ACTOR,  "prop.specular",         "Glossy highlight strength (0-1)."),
    "specular_pow":         (LEVEL_ACTOR,  "prop.specular_power",   "Focus of gloss (1-100). Higher = shinier/smaller spot."),
    "specular_color":       (LEVEL_ACTOR,  "prop.specular_color",   "The color of the glossy highlight (RGB). Usually white [1,1,1]."),
    
    # === Physically Based Rendering - PBR ===
    "metallic":             (LEVEL_ACTOR,  "prop.metallic",         "PBR metallic effect (0-1). Needs PBR enabled."),
    "roughness":            (LEVEL_ACTOR,  "prop.roughness",        "PBR surface roughness (0-1). Needs PBR enabled."),

    # === Color Control ===
    "color_rule":           (LEVEL_RECALC, None,                    ("Determines point colors. Options: "
                                                                     "1) Uniform (e.g. (1,0,0)), "
                                                                     "2) Function (mapping function), "
                                                                     "3) 'manual' (uses color_values), "
                                                                     "4) 'scalars' (maps 1D data to colors using cmap/clim).")),
    "color_values":         (LEVEL_RECALC, None,                    "The resolved RGB/RGBA array."),

    # === Opacity Control ===
    "opacity_rule":         (LEVEL_RECALC, None,                    "Determines point transparency. Options: 1) Uniform (float 0-1), 2) Function (mapping function), 3) 'manual' (uses opacity_values)."),
    "opacity_values":       (LEVEL_RECALC, None,                    "The resolved Alpha array."),

    # === Radius Control ===
    "radius_rule":          (LEVEL_REMESH, None,                    "Determines tube thickness. Options: 1) Uniform (float), 2) Function (mapping function), 3) 'manual' (uses radius_values)."),
    "radius_values":        (LEVEL_REMESH, None,                    "The resolved radius array."),
    
    # === Scalars Control (Needs color_rule='scalars') ===
    "scalars_rule":         (LEVEL_RECALC, None,                    "Determines point scalars. Options: 1) Function (mapping function), 2) 'manual' (uses scalars_values), 3) None (No scalars)"),
    "scalars_values":       (LEVEL_RECALC, None,                    "The resolved scalars array."),
    "cmap":                 (LEVEL_RECALC, None,                    "Colormap name (e.g., 'viridis') used if color_rule is scalar."),
    # "clim":                 (LEVEL_RECALC, None,                    "Color limits [min, max] for scalar mapping."),
    # "is_scalar_bar":        (LEVEL_ACTOR,  None,                    "Whether to display the color legend (scalar bar)."),
    # "scalar_bar_title":     (LEVEL_ACTOR,  None,                    "Title for the scalar bar (e.g., 'Stress (MPa)')."),

    # === Geometry & Topology (LEVEL_REMESH) ===
    "sides":                (LEVEL_REMESH, None,                    "Number of facets around the tube (higher = smoother)."),
    "is_capping":           (LEVEL_REMESH, None,                    "Whether to close the ends of the tube."),
    "smooth_iter":          (LEVEL_REMESH, None,                    "Path smoothing iterations to remove jagged edges."),
    
    # === Advanced Spatial Clipping ===
    "clip_geometry":        (LEVEL_REMESH, None,                    "Define clipping boundary. Can be: "
                                                                    "1) List of 6 floats [xmin, xmax...] for axis-aligned box, "
                                                                    "2) A Mesh/PolyData representing any closed shape (e.g. 8-point box).")
}


@dataclass(slots=True)
class OptsTube:
    # --- Visibility & Global ---
    name: str = "tube"
    is_visible: bool = True
    shading_type: str = "phong"

    # --- Phong Lighting ---
    ambient: float = 0.0
    diffuse: float = 1.0
    specular: float = 0.0
    specular_pow: float = 10.0
    specular_color: ColorRGB = (1.0, 1.0, 1.0)

    # --- PBR Lighting ---
    metallic: float = 0.0
    roughness: float = 0.5

    # --- Rules & Values ---
    color_rule: ColorRGB | str | Callable = (0,0,0)
    color_values: Optional[np.ndarray] = field(default=None, repr=False, metadata={"is_data": True})
    
    opacity_rule: float | str | Callable = 1.0
    opacity_values: Optional[np.ndarray] = field(default=None, repr=False, metadata={"is_data": True})
    
    radius_rule: float | str | Callable = 0.1
    radius_values: Optional[np.ndarray] = field(default=None, repr=False, metadata={"is_data": True})
    
    # --- Scalars (Used if color_rule='scalars') ---
    scalars_rule: str | Callable | None = None
    scalars_values: Optional[np.ndarray] = field(default=None, repr=False, metadata={"is_data": True})
    cmap: str = "viridis"
    clim: list | None = None

    # --- Geometry & Clipping ---
    sides: int = 6
    is_capping: bool = True
    smooth_iter: int = 0
    clip_geometry: list[float] | pv.PolyData | None = None

    # --- Internal State ---
    _owner: object | None = field(default=None, repr=False, init=False)

    _validators = {
        
        "name": lambda self, v, d: as_str(v, name=d, replace='tube'),
        "is_visible": lambda self, v, d: as_bool(v, name=d, replace=True),
        # "shading_type": lambda self, v, d: v if v.lower() in ['phong', 'pbr'] else 'standard',
        
        "ambient": lambda self, v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True, replace=0.0),
        "diffuse": lambda self, v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True, replace=1.0),
        "specular": lambda self, v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True, replace=0.0),
        "specular_pow": lambda self, v, d: as_Number(v, name=d, value_range=(1, 100), bounded=True, replace=10.0),
        "specular_color": lambda self, v, d: as_ColorRGB(v, name=d, replace=(1.0, 1.0, 1.0)),
        
        "metallic": lambda self, v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True, replace=0.0),
        "roughness": lambda self, v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True, replace=0.5),
        
        "color_rule": lambda self, v, d: (
            v if isinstance(v, (str, Callable)) 
            else as_ColorRGB(v, name=d, replace=(0.0, 0.0, 0.0))
        ),
        "color_values": lambda self, v, d: None if v is None else as_ColorRGB_array(v, name="color_values"),
        
        "opacity_rule": lambda self, v, d: (
            v if isinstance(v, (str, Callable)) 
            else as_Number(v, name=d, value_range=(0, 1), bounded=True, replace=1.0)
        ),
        #"opacity_values": lambda self, v, d: v if isinstance(v, np.ndarray) or v is None else None,
        
        "radius_rule": lambda self, v, d: (
            v if isinstance(v, (str, Callable)) 
            else as_Number(v, name=d, value_range=(0, np.inf), bounded=True, replace=0.1)
        ),
        #"radius_values": lambda self, v, d: v if isinstance(v, np.ndarray) or v is None else None,
        
        #"scalars_rule": lambda self, v, d: v if (isinstance(v, Callable) or v in ['manual', None]) else None,
        #"scalars_values": lambda self, v, d: v if isinstance(v, np.ndarray) or v is None else None,
        #"cmap": lambda self, v, d: as_str(v, name=d, replace='viridis'),
        #"clim": lambda self, v, d: v if (isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2) else None,
        
        "sides": lambda self, v, d: as_Number(v, name=d, is_int=True, value_range=(3, 128), bounded=True, replace=6),
        "is_capping": lambda self, v, d: as_bool(v, name=d, replace=True),
        "smooth_iter": lambda self, v, d: as_Number(v, name=d, is_int=True, value_range=(0, 1000), bounded=True, replace=0),
    }

    def __setattr__(self, key, value):

        if key in self._validators:
            desc = ATTR_MAP.get(key, (None, None, key))[2]
            value = self._validators[key](self, value, desc)

        object.__setattr__(self, key, value)
        
        # if key != "_owner" and hasattr(self, "_owner") and self._owner is not None:
        #     self._owner.act_commit(**{key: value})
        
        
class PlotTube:
    """
    Wraps PyVista tube filtering and rendering with integrated option management.
    """
    __descriptions__ = {
        "_raw_coords": "The N x 3 input coordinates",
        "_calc_mesh": "The generated PyVista PolyData mesh of the tube",
        "_entities": "The PyVista Actor in the plotter",
        "opts": "The OptsTube instance for configuration"
    }
    
    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        plotter: pv.Plotter,
        opts: OptsTube = OptsTube(),
        logger = None,
        **kwargs
    ):
        
        # Initializing internal states
        object.__setattr__(self, "_raw_coords", np.asarray(coords))
        object.__setattr__(self, "_entities", None)
        object.__setattr__(self, "_calc_mesh", None)

        # Handle explicit kwargs overrides
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_owner", self)
        object.__setattr__(self, "opts", opts)

        logger.detail("Executing initial plot")
        self._helper_make_figure(plotter=plotter)
      
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_generic(self, rule_attr, values_attr, target_shape, default_val, logger=None):

        rule = getattr(self.opts, rule_attr)
        
        try:
            # 1. Dispatching Logic
            if callable(rule):
                resolved = np.asarray(rule(self._raw_coords))
            elif rule == "manual":
                resolved = getattr(self.opts, values_attr)
                if resolved is None:
                    raise TypeError(f"{values_attr} is None in manual mode.")
            else:
                resolved = np.full(target_shape, rule)
    
            # 2. Integrity Check
            resolved = np.asarray(resolved, dtype=np.float32)
            if resolved.shape != target_shape:
                raise ValueError(f"Shape mismatch: {resolved.shape} vs {target_shape} in {values_attr}")
    
            setattr(self.opts, values_attr, resolved)
    
        except Exception as e:
            logger.exception(f"Failed to resolve {rule_attr}: {str(e)}")
            fallback = np.full(target_shape, default_val, dtype=np.float32)
            setattr(self.opts, values_attr, fallback)
            setattr(self.opts, rule_attr, default_val)
            logger.recovery(f"Reset {values_attr} to default: {default_val} everywhere.")

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_spec(self, attr, logger=None):
        
        if attr not in {'color', 'radius', 'scalars', 'opacity'}:
            raise ValueError(f"Attribute resolved by `_helper_resolver_init()` must be in ['color', 'radius', 'scalars', 'opacity']. Got {attr} instead.")
            
        default_val_dir = {
            'color':    (0,0,0),
            'radius':   0.1,
            'scalars':  0.0,
            'opacity':  1
            }
            
        input_dir = {
            'rule_attr':        attr + "_rule",
            'values_attr':       attr + "_values",
            'target_shape':     (len(self._raw_coords),3) if attr=='color' else (len(self._raw_coords),),
            'default_val':      default_val_dir[attr]
            }
        
        self._helper_resolver_generic(**input_dir)
        
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_build_tube_mesh(self, logger=None):
        """
        Internal: Create the PyVista PolyData, apply smoothing/clipping, 
        and generate tube with dynamic or static radius.
        """
        logger.detail("Creating a line from coordinates")
        poly = pv.MultipleLines(self._raw_coords)
        
        if self.opts.smooth_iter > 0:
            logger.detail(f"Smoothing path with {self.opts.smooth_iter} iterations")
            poly = poly.smooth(n_iter=self.opts.smooth_iter)
        
        poly.point_data['radius'] = self.opts.radius_values 
        if self.opts.color_rule == 'scalars':
            poly.point_data['opacity'] = self.opts.opacity_values
            poly.point_data['scalars'] = self.opts.scalars_values
        else:
            rgba_values = np.hstack([self.opts.color_values, self.opts.opacity_values.reshape(-1, 1)])
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

        object.__setattr__(self, "_calc_mesh", mesh)
        return mesh
    
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_make_figure(self, plotter: pv.Plotter, logger=None):
        """
        Creates or updates the rendering in a PyVista Plotter.
        """
        
        is_scalars = (self.opts.color_rule == 'scalars')
        
        input_dir = {
            "name":     self.opts.name,
            "pbr":      self.opts.shading_type == 'pbr',
            "rgb":      not is_scalars,
            "scalars":  'scalars' if is_scalars else 'rgba',
            }
        
        logger.detail("Resolving data for color, opacity and radius")
        self._helper_resolver_spec('opacity')
        self._helper_resolver_spec('radius')
        
        if is_scalars:
            self._helper_resolver_spec('scalars')
            input_dir["opacity"] = "opacity"
        else:
            self._helper_resolver_spec('color')
        
        logger.detail("Creating tube mesh")
        mesh = self._helper_build_tube_mesh(logger=logger)
            
        logger.detail("Visualizing the tube")
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
        self.opts.shading_type = shading
            

        # Set specific lighting values
        prop.ambient = self.opts.ambient
        prop.diffuse = self.opts.diffuse
        prop.specular = self.opts.specular
        prop.specular_power = self.opts.specular_pow
        prop.specular_color = self.opts.specular_color
        
        # PBR specific
        if shading == 'pbr':
            prop.metallic = self.opts.metallic
            prop.roughness = self.opts.roughness
            
        # Global visibility
        actor.visibility = self.opts.is_visible

        # 4. Store the actor reference for future LEVEL_ACTOR updates
        object.__setattr__(self, "_entities", actor)
        
        # 5. Final render trigger
        plotter.render()
        
    @logging_and_warning_decorator()
    def act_commit(self, logger=None, **updates):
        """
        Unified entry point for property updates. 
        Decides the minimal pipeline stage required to reflect changes.
        """
        max_level = -1
        changed_actor_attrs = []
    
        # 1. Analyze updates and find the highest required level
        for key, value in updates.items():
            if key not in ATTR_MAP:
                continue
                
            level, path = ATTR_MAP[key]
            max_level = max(max_level, level)
            
            if level == LEVEL_ACTOR and path:
                changed_actor_attrs.append((path, value))
    
        # 2. Execute pipeline stages based on max_level
        if max_level == LEVEL_REMESH:
            # Full rebuild: Re-generate mesh -> Re-resolve color -> Re-add actor
            # Note: act_replot must call _helper_resolver_color internally 
            # to ensure the new mesh has the correct color arrays.
            self.act_rereplot(self._plotter)
            
        elif max_level == LEVEL_RECALC:
            # Data update: Re-resolve color values and inject into existing mesh
            self._helper_resolver_color()
            if self._calc_mesh:
                self._calc_mesh.point_data['colors'] = self.opts.color_values
                # Manual update of scalars usually requires a render call
                self._plotter.render()
                
        elif max_level == LEVEL_ACTOR:
            # Property update: Direct modification of VTK actor properties
            for path, val in changed_actor_attrs:
                target = self._entities[0]
                attrs = path.split(".")
                for attr in attrs[:-1]:
                    target = getattr(target, attr)
                setattr(target, attrs[-1], val)
            self._plotter.render()
