from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, List
import numpy as np
import pyvista as pv

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    as_ColorRGB_array,
    as_Number,
    as_str,
    as_bool,
    Vect,
    as_Vect
)
from .plot_figure import PlotFigure
from ..opts import merge_opts_all

#! scalars_limit
#! scalars_bar
#! clip_geometry
#! light dark pbr

#! only change cmap


LEVEL_ACTOR  = 0  # Only changes GPU/Rendering state. (Fastest)
LEVEL_RECALC = 1  # Needs to re-calculate data arrays (colors, etc.) but keeps geometry.
LEVEL_REMESH = 2  # Needs to re-run the tube filter to rebuild the 3D mesh. (Heaviest)

ATTR_MAP = {
    # === Visibility & Global Settings ===
    "name":                 (LEVEL_ACTOR,  None,                    "Unique identifier for the actor in the plotter."),
    "category":             (LEVEL_ACTOR,  None,                    "The semantic category of this plotting entity."),
    "is_visible":           (LEVEL_ACTOR,  "visibility",            "Whether the tube is visible in the scene."),
    "shading_type":         (LEVEL_ACTOR,  "prop.interpolation",    "'phong', 'pbr' (Physical)"),
    "is_reset_camera":      (LEVEL_ACTOR,  None,                    "Whether to reset the camera settings for each (re-)plot."),

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
                                                                     "4) 'scalars' (maps 1D data to colors using scalars_cmap/scalars_clim).")),
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
    "scalars_cmap":         (LEVEL_RECALC, None,                    "Colormap name (e.g., 'viridis') used if color_rule is scalar."),
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


@dataclass(slots=True)
class OptsTube:
    # --- Visibility & Global ---
    name: str = "tube"
    category: str = 'line'
    is_visible: bool = True
    shading_type: str = "phong"
    is_reset_camera: bool = True

    # --- Phong Lighting ---
    ambient: float = 0.0
    diffuse: float = 1.0
    specular: float = 1.0
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
    scalars_cmap: str = "viridis"
    scalars_clim: Vect(2) | None = None
    is_scalar_bar: bool = True
    scalar_bar_title: str = 'scalars'
    

    # --- Geometry & Clipping ---
    sides: int = 6
    is_capping: bool = True
    smooth_iter: int = 0
    clip_geometry: list[float] | pv.PolyData | None = None

    # --- Internal State ---
    _owner: object | None = field(default=None, repr=False, init=False)
    _restricted: bool = field(default=False, init=False, repr=False)
        

    _validators = {
        
        "name": lambda self, v, d: as_str(v, name=d, replace='tube'),
        "category": lambda self, v, d: as_str(v, name=d, replace='line'),
        "is_visible": lambda self, v, d: as_bool(v, name=d, replace=True),
        "shading_type": lambda self, v, d: as_str(v, name=d, replace='phong', pool=('phong', 'pbr')),
        "is_reset_camera": lambda self, v, d: as_bool(v, name=d, replace=True),
        
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
        
        "scalars_cmap": lambda self, v, d: as_str(v, name=d, replace='viridis'),
        "scalars_clim": lambda self, v, d: v if v is None else as_Vect(v, name=d, dim=2, replace=None),
        "is_scalar_bar": lambda self, v, d: as_bool(v, name=d, replace=True),
        "scalar_bar_title": lambda self, v, d: as_str(v, name=d, replace='scalars'),
        
        "sides": lambda self, v, d: as_Number(v, name=d, is_int=True, value_range=(3, 128), bounded=True, replace=6),
        "is_capping": lambda self, v, d: as_bool(v, name=d, replace=True),
        "smooth_iter": lambda self, v, d: as_Number(v, name=d, is_int=True, value_range=(0, 1000), bounded=True, replace=0),
    }

    def __setattr__(self, key, value):

        if key in self._validators:
            desc = f'{key!r}: {ATTR_MAP.get(key)[2]}'
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
        "_calc_poly": "The generated PyVista PolyData",
        "_entities": "The PyVista Actor in the plotter",
        "opts": "The OptsTube instance for configuration",
    }
    
    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        Figure: PlotFigure,
        opts: OptsTube = OptsTube(),
        logger = None,
        **kwargs
    ):

        
        # Initializing internal states
        object.__setattr__(self, "_raw_coords", np.asarray(coords))
        object.__setattr__(self, "_entities", None)
        object.__setattr__(self, "_calc_mesh", None)
        object.__setattr__(self, "_calc_poly", None)

        logger.detail('Handling explicit kwargs overrides')
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_owner", self)
        object.__setattr__(self, "opts", opts)
        
        logger.detail('Checking if name already exists')
        name_set = set(Figure.act_get_entities_names())
        name_input = opts.name
        if name_input in name_set:
            index = 1
            new_name = f"{name_input}_{index}"
            while opts.name in name_set:
                index += 1
                new_name = f"{name_input}_{index}"
                
            opts.name = new_name
            logger.warning(f"{name_input!r} already exists in PlotFigure object! Renamed to {opts.name!r}.")
        

        logger.detail("Executing initial plot")
        self._helper_resolver_init()
        self._helper_make_figure(Figure=Figure, is_reset_camera=opts.is_reset_camera)
        
        object.__setattr__(self.opts, "_restricted", True)
      
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_generic(self, attr_rule, attr_values, target_shape, default_val, logger=None):

        rule = getattr(self.opts, attr_rule)
        
        try:
            # 1. Dispatching Logic
            if rule == "manual":
                resolved = getattr(self.opts, attr_values)
                if resolved is None:
                    raise TypeError(f"`manual` mode of {attr_rule!r} requires {attr_values!r} to provide raw data, but only got `None`.")
            else:
                if getattr(self.opts, attr_values) is not None:
                    msg = f"{attr_rule!r} is set to {rule}. {attr_values!r} will be ignored.\n "
                    msg += f"To enable {attr_values!r}, set {attr_rule!r} to `manual`."
                    logger.warning(msg)
            
                if callable(rule):
                    resolved = np.asarray(rule(self._raw_coords))
                else:
                    resolved = np.full(target_shape, rule)
    
            # 2. Integrity Check
            resolved = np.asarray(resolved, dtype=np.float32)
            if resolved.shape != target_shape:
                raise ValueError(f"Shape mismatch: {resolved.shape} vs {target_shape} in {attr_values}")
    
            setattr(self.opts, attr_values, resolved)
    
        except Exception as e:
            logger.exception(f"Failed to resolve {attr_rule}: {str(e)}")
            fallback = np.full(target_shape, default_val, dtype=np.float32)
            setattr(self.opts, attr_values, fallback)
            setattr(self.opts, attr_rule, default_val)
            logger.recovery(f"Reset {attr_values} to default: {default_val} everywhere.")
            
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_init(self, logger=None):
        logger.detail("Resolving data for color, opacity and radius")
        self._helper_resolver_spec('opacity')
        self._helper_resolver_spec('radius')
        
        if self.opts.color_rule == 'scalars':
            self._helper_resolver_spec('scalars')
        else:
            self._helper_resolver_spec('color')

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_spec(self, attr, logger=None):
        
        if attr not in ['color', 'radius', 'scalars', 'opacity']:
            raise ValueError(f"Attribute resolved by `_helper_resolver_init()` must be in ['color', 'radius', 'scalars', 'opacity']. Got {attr} instead.")
            
        default_val_dir = {
            'color':    (0,0,0),
            'radius':   0.1,
            'scalars':  0.0,
            'opacity':  1
            }
            
        input_dir = {
            'attr_rule':        attr + "_rule",
            'attr_values':       attr + "_values",
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
        object.__setattr__(self, "_calc_poly", poly)
        return mesh
    
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_make_figure(self, Figure: PlotFigure, is_reset_camera: bool = True, logger=None):
        """
        Creates or updates the rendering in a PyVista Plotter.
        """
        
        is_scalars = (self.opts.color_rule == 'scalars')
        
        input_dir = {
            "name":         self.opts.name,
            "pbr":          self.opts.shading_type == 'pbr',
            "rgb":          not is_scalars,
            "scalars":      'scalars' if is_scalars else 'rgba',
            "reset_camera": is_reset_camera
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
        actor = Figure.obj_plotter.add_mesh(mesh, **input_dir)
        
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
        
        Figure.obj_plotter.render()
        Figure.obj_plotter.show(interactive_update=True)
        
        Figure._helper_register_entity(self, self.opts.category, self.opts.is_reset_camera)
        
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_imperative(self, attr, values, logger=None):

        object.__setattr__(self.opts, "_restricted", False)        

        valid_attrs = ['color', 'radius', 'scalars', 'opacity']
        if attr not in valid_attrs:
            raise ValueError(f"Imperative attribute must be in {valid_attrs}. Got '{attr}' instead.")

        is_array_data = (isinstance(values, (np.ndarray, list, tuple)) and 
                         not (attr == 'color' and not isinstance(values[0], (list, np.ndarray))))

        if is_array_data:
            setattr(self.opts, f"{attr}_values", np.asarray(values))
            setattr(self.opts, f"{attr}_rule", "manual")
            
        else:
            setattr(self.opts, f"{attr}_rule", values)
            
        if attr == 'scalars':
            self.opts.color_rule = 'scalars'
        
        self._helper_resolver_spec(attr)
        
        object.__setattr__(self.opts, "_restricted", True)
        
        
    def _helper_replace_data_pv(self, attr: str, data: np.ndarray):
        if attr in self._calc_poly.point_data:
            del self._calc_poly.point_data[attr]
        if attr in self._calc_mesh.point_data:
            del self._calc_mesh.point_data[attr]
        self._calc_poly.point_data[attr] = data
        self._calc_mesh.point_data[attr] = self._calc_mesh.interpolate(self._calc_poly).point_data[attr]
        
    # @logging_and_warning_decorator(start_finish_level=5)
    # def _helper_update_rgba(self, logger=None):
    #     rgba = np.hstack([self.opts.color_values, self.opts.opacity_values.reshape(-1, 1)])
    #     if 'rgba' in self._calc_poly.point_data:
    #         del self._calc_poly.point_data['rgba']
    #     if 'rgba' in self._calc_mesh.point_data:
    #         del self._calc_mesh.point_data['rgba']
    #     self._calc_poly.point_data['rgba'] = rgba
    #     self._calc_mesh.point_data['rgba'] = self._calc_mesh.interpolate(self._calc_poly).point_data['rgba']   
    
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_update_rgba(self, logger=None):
        rgba = np.hstack([self.opts.color_values, self.opts.opacity_values.reshape(-1, 1)])
        self._helper_replace_data_pv('rgba', rgba)
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_switch_scalars_to_rgba(self, logger=None):
        logger.detail("Change the color_rule from 'scalars' to current color settings")
        mapper = self._entities.mapper
        mapper.scalar_visibility = True
        mapper.color_mode = 'direct'
        mapper.SetColorModeToDirectScalars()
        mapper.lookup_table = None
        self._calc_mesh.set_active_scalars('rgba')
        mapper.SetArrayName('rgba')
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_update_scalars(self, is_update_opacity=False, logger=None):
        logger.detail("Update scalar coloring, which may involve switching from a direct color-based scheme to scalar-based coloring.")

        self._helper_replace_data_pv('scalars', self.opts.scalars_values)
        
        mapper = self._entities.mapper
        mesh = mapper.dataset.point_data
        
        if not is_update_opacity:
            mapper.set_scalars(
                mesh['scalars'], 
                'scalars',
                cmap = self.opts.scalars_cmap,
                clim = self.opts.scalars_clim)
        else:
            if "__custom_rgba" in mesh.keys():
                mesh.remove("__custom_rgba")
            self._helper_replace_data_pv('opacity', self.opts.opacity_values)
            mapper.set_scalars(
                mesh['scalars'], 
                'scalars',
                cmap = self.opts.scalars_cmap,
                clim = self.opts.scalars_clim,
                custom_opac=True,
                opacity=mesh['opacity'])
        
        # self._calc_mesh.set_active_scalars('scalars')
        # mapper = self._entities.mapper
        # mapper.scalar_visibility = True
        # mapper.color_mode = "map" 
        # mapper.SetColorModeToMapScalars()
        # mapper.SetArrayName('scalars')
        
        # if self.opts.scalars_clim is None:
        #     s1 = self._calc_mesh.point_data['scalars']
        #     s2 = self._calc_poly.point_data['scalars']
        #     vmin = np.nanmin([np.nanmin(s1), np.nanmin(s2)])
        #     vmax = np.nanmax([np.nanmax(s1), np.nanmax(s2)])
        #     scalars_clim = (vmin, vmax)
        # else:
        #     scalars_clim = self.opts.scalars_clim
        
        # mapper.SetLookupTable(pv.LookupTable(cmap=self.opts.scalars_cmap))
        # mapper.GetLookupTable().SetRange(*scalars_clim)
        # mapper.GetLookupTable().Build()
        # mapper.SetUseLookupTableScalarRange(True)
        
        # if is_update_opacity:
        #     self._helper_replace_data_pv('opacity', self.opts.opacity_values)
        #     mapper.SetScalarOpacityArrayName('opacity')
        # # mapper.SetUseLookupTableScalarRange(True)
        
        
        
        
    ''' 
    @logging_and_warning_decorator()
    def act_commit(self, logger=None, **kwargs):
        
        is_needs_remesh = False
        
        current_shading = kwargs.get("shading_type", self.opts.get("shading_type", "phong"))
        previous_color_rule = self.opts.get("color_rule")

        for key, value in kwargs.items():
            
            try:
                if key not in ATTR_MAP:
                        raise ValueError(f"Unknown attribute: {key} in class: PlotTube.opts")
    
                level, attr_path_actor, doc = self.ATTR_MAP[key]
    
                # Dealing with LEVEL ACTOR (simply resetting values)
                if level == self.LEVEL_ACTOR:
                    
                    if key == "category":
                        raise AttributeError("Modification of 'category' is not allowed, because it is used as the key in dir: PlotFigure._entities")
                    
                    if key == "name":
                        msg = "Changing 'name' of PlotTube object is not recommended because: \n"
                        msg += "1) There is no guarantee that name collisions will be avoided in PlotFigure._entities; and\n"
                        msg += "2) The corresponding actor name stored in the PyVista renderer cannot be updated accordingly."
                        logging.warning(msg)
                        self.opts[key] = value
                        continue
                    
                    if key in "is_reset_camera":
                        self.opts[key] = value
                        continue
    
                    if key in ["is_visible", "shading_type"]:
                        self.opts[key] = value
                        parts = attr_path_actor.split('.')
                        obj = self._entities
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value)
                        continue
    
                    pbr_params = ["metallic", "roughness"]
                    phong_params = ["ambient", "diffuse", "specular", "specular_pow", "specular_color"]
                    
                    if key in pbr_params and current_shading != "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is '{current_shading}'. PBR effects may not show.")
                    elif key in phong_params and current_shading == "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is 'pbr'. Phong lighting parameters may be ignored.")
    
                    self.opts[key] = value
    
                    if entity_path:
                        parts = attr_path_actor.split('.')
                        obj = self._entities
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value)
    
                
                 # Dealing with LEVEL_RECALC (resolver for color, opacity and scalars)
                
                elif level == self.LEVEL_RECALC:
                    scalar_bar_keys = ["scalars_cmap", "scalars_clim", "is_scalar_bar", "scalar_bar_title"]
                    if key in scalar_bar_keys:
                        raise NotImplementedError(f"Directly modifying '{key}' is not supported in this version.")
    
                    if key in ["color_rule", "color_values", "opacity_rule", "opacity_values", "scalars_rule", "scalars_values"]:

                        target_prop = "color"
                        if "opacity" in key: target_prop = "opacity"
                        if "scalars" in key: target_prop = "scalars"
                        
                        raise ValueError(f"Cannot set '{key}' directly. Please use the '.{target_prop}' property instead.")
    
                    if key in ["color", "scalars", "opacity"]:
                        self._helper_resolver_imperative(key, value)
    
                # ==========================================
                # 3. 处理 LEVEL_REMESH (Geometry)
                # ==========================================
                elif level == self.LEVEL_REMESH:
                    if key == "clip_geometry":
                        logging.error("Modification of 'clip_geometry' is currently not supported.")
                        continue
                    
                    if key in ["radius_rule", "radius_values"]:
                        logging.error("Cannot set radius details directly. Please use the '.radius' property.")
                        continue
    
                    # 正常修改 sides, is_capping, smooth_iter
                    self.opts[key] = value
                    needs_remesh = True
    
            # 最后处理：如果改了 remesh 级别的参数，触发重画
            if is_needs_remesh:
                self.replot()
    '''

