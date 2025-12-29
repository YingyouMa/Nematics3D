from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Sequence, Literal
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

DEFAULT_VAL_DIR = {
    'color':    (0,0,0),
    'radius':   0.1,
    'scalars':  0.0,
    'opacity':  1
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

    # --- Shape and Color ---
    color: ColorRGB | Callable | Sequence | Literal['scalars'] = (0,0,0)
    opacity: float | Callable | Sequence = 1.0
    scalars: Callable | Sequence | None = None
    radius: float | Callable | Sequence = 0.1
    
    # --- Scalars (Used if color='scalars') ---
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
    _is_category_locked: bool = field(default=False, repr=False, init=False)
        

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
        
        # color, opacity, scalars, radius
        
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
            
        if self._is_category_locked and key == 'category':
            raise AttributeError("Modification of 'category' is not allowed, because it is used as the key in dir: PlotFigure._entities")

        object.__setattr__(self, key, value)
        
        # if key != "_owner" and hasattr(self, "_owner") and self._owner is not None:
        #     self._owner.act_commit(**{key: value})
        
        
class PlotTube:
    """
    Wraps PyVista tube filtering and rendering with integrated option management.
    """
    __descriptions__ = {
        "_raw_coords": "The N x 3 input coordinates",
        "_calc_poly": "The generated PyVista PolyData",
        "_calc_color": "The pointwise data of color of tube",
        "_calc_opacity": "The pointwise data of color of tube",
        "_calc_radius": "The pointwise data of opacity of tube",
        "_calc_scalars": "The pointwise data of scalars of tube",
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
        object.__setattr__(self, "_calc_poly", None)
        object.__setattr__(self, "_calc_color", None)
        object.__setattr__(self, "_calc_opacity", None)
        object.__setattr__(self, "_calc_radius", None)
        object.__setattr__(self, "_calc_scalars", None)

        logger.detail('Handling explicit kwargs overrides')
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_owner", self)
        
        if not (isinstance(opts.color, str) and opts.color == 'scalars') and opts.scalars is not None:
            msg = "Color input of PlotTube is not set to 'scalars'. However, scalars is provided.\n"
            msg += "The scalars data will be ignored unless color='scalars' is explicitly specified."
            logger.warning(msg)
        
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
        
        object.__setattr__(self.opts, "_is_restricted", True)
      
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_generic(self, attr_name, attr_input, default_val, logger=None):
        
        target_shape = (len(self._raw_coords),3) if attr_name=='color' else (len(self._raw_coords),)
        
        try:
            if attr_input is None:
                raise TypeError(f"Require input for {attr_name!r}. Got None instead.")
            elif callable(attr_input):
                resolved = np.asarray(attr_input(self._raw_coords), dtype=np.float32)
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
            if default_val:
                resolved = np.full(target_shape, default_val, dtype=np.float32)
                logger.recovery(f"Reset {attr_name!r} to default: {default_val} everywhere.")
            else:
                logger.recovery(f"Ignore this modification of {attr_name!r}")
              
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
    def _helper_resolver_spec(self, attr_name, is_keep_on_error=False, logger=None):
        
        # is_keep_on_error: whether keep the current settings when resolver failed. If false, use the default settings in DEFAULT_VAL_DIR
        
        if attr_name not in ['color', 'radius', 'scalars', 'opacity']:
            raise ValueError(f"Attribute resolved by `_helper_resolver_init()` must be in ['color', 'radius', 'scalars', 'opacity']. Got {attr_name} instead.")
            
        default_var = None if is_keep_on_error else DEFAULT_VAL_DIR[attr_name]
        
        self._helper_resolver_generic(attr_name, getattr(self.opts, attr_name), default_val)
        
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
    def _helper_make_figure(self, Figure: PlotFigure, is_reset_camera: bool = True, logger=None):
        """
        Creates or updates the rendering in a PyVista Plotter.
        """
        
        is_scalars = (isinstance(self.opts.color, str) and self.opts.color == 'scalars')
        
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
        
        self.opts._is_category_locked = True
        Figure._helper_register_entity(self, self.opts.category, self.opts.is_reset_camera)
        
        
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
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_switch_scalars_to_rgba(self, logger=None):
        logger.detail("Change the color from 'scalars' to current color settings")
        mapper = self._entities.mapper
        mapper.scalar_visibility = True
        mapper.color_mode = 'direct'
        mapper.SetColorModeToDirectScalars()
        mapper.lookup_table = None
        self._entities.mapper.dataset.set_active_scalars('rgba')
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
        
        
    @logging_and_warning_decorator()
    def act_commit(self, is_setattr=True, logger=None, **kwargs):
        
        is_needs_remesh = False
        current_shading = kwargs.get("shading_type", getattr(self.opts, "shading_type"))
        current_shading = as_str((current_shading, name='shading_type', replace=getattr(self.opts, "shading_type"), pool=('phong', 'pbr')))

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
                    
                    # if key in "is_reset_camera":
                    # if key in ["is_visible", "shading_type"]:
    
                    pbr_params = ["metallic", "roughness"]
                    phong_params = ["ambient", "diffuse", "specular", "specular_pow", "specular_color"]
                    
                    if key in pbr_params and current_shading != "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is '{current_shading}'. PBR effects may not show.")
                    elif key in phong_params and current_shading == "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is 'pbr'. Phong lighting parameters may be ignored.")
    
                    if entity_path:
                        parts = attr_path_actor.split('.')
                        obj = self._entities
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value)
                        
                    if is_setattr:
                        object.__setattr__(self.opts, key, value)
                
                # Dealing with LEVEL_RECALC (resolver for color, opacity and scalars)
                elif level == self.LEVEL_RECALC:
                    self._helper_resolver_spec(key)
    
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