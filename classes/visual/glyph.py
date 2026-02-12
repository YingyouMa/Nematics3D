from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Literal, Mapping, Sequence, Type, List
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
    as_ColorRGB_array,
    as_Vect,
    as_points
)
from ..host_base import OptsBase, HostBase
from .plot_figure import PlotFigure
from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.general import pop_exclusive

#!!! resolver source
#!!! colorbar name args
#!!! is_reset_camera commit

# --- Type aliases ---
ColorMode = ColorRGB | Callable | Sequence 
OpacityMode = float | Callable | Sequence
RadiusMode = float | Callable | Sequence
ScalarsMode = Callable | Sequence | None
ClipGeometryLike = list[float] | pv.PolyData | None


@dataclass(slots=True, repr=False)
class OptsGlyph(OptsBase):
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

    # --- Scalars (used if color == "scalars") ---
    scalars_cmap:               str | Unset                         = UNSET
    scalars_clim:               Vect(2) | None | Unset              = UNSET
    is_scalar_bar:              bool | Unset                        = UNSET
    scalar_bar_title:           str | Unset                         = UNSET

    # --- Geometry & Clipping ---
    sides:                      int | Unset                         = UNSET
    clip_geometry:              ClipGeometryLike | Unset            = UNSET

    __descriptions__: ClassVar[Mapping[str, str]] = {
        **(OptsBase.__descriptions__),
        
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
        
        # === Scalars Control (Needs color_rule='scalars') ===
        "scalars_cmap":         "Colormap name (e.g., 'viridis') used if color is set to scalar.",
        "scalars_clim":         "Color limits [min, max] for scalar mapping.",
        "is_scalar_bar":        "Whether to display the color legend (scalar bar).",
        "scalar_bar_title":     "Title for the scalar bar (e.g., 'Stress (MPa)').",
        
        # --- Geometry & Clipping ---
        "sides":                "Number of facets around the glyph (higher = smoother).",
        "clip_geometry": (
            "(INVALID FOR NOW!!!) Define clipping boundary. Can be: "
            "1) List of 6 floats [xmin, xmax...] for axis-aligned box, "
            "2) A Mesh/PolyData representing any closed shape (e.g. 8-point box)."
        ),
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **(OptsBase._validators),
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
        "scalars_cmap":         lambda v, d: as_str(v, name=d),
        "scalars_clim":         lambda v, d: (v if v is None else as_Vect(v, name=d, dim=2)),
        "is_scalar_bar":        lambda v, d: as_bool(v, name=d),
        "scalar_bar_title":     lambda v, d: as_str(v, name=d),
        "sides":                lambda v, d: as_Number(v, name=d, is_int=True, value_range=(3, 128), bounded=True),
        }


    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
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
        "scalars_cmap":         "viridis",
        "scalars_clim":         None,
        "is_scalar_bar":        True,
        "scalar_bar_title":     "scalar",
        "sides":                12,
        "clip_geometry":        None,
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
    
    def _helper_host_apply(self, key, value):
        if self.host:
            self.host.act_commit(**{key: value})
            return value
    
  
class PlotGlyph(HostBase):
    
    __descriptions__: ClassVar[Mapping[str, str]] = {
        **dict(HostBase.__descriptions__),
        
        "raw_category":                 "The category of the glyph, used in the classfication of PlotFigure",
        "raw_coords":                   "The N x 3 input coordinates of each glyph",
        
        "_calc_poly":                   "The generated PyVista PolyData",
        
        "_calc_color":                  "The resolved per-point RGB color array of the glyph.",
        "_calc_opacity":                "The resolved per-point opacity array of the glyph.",
        "_calc_radius":                 "The resolved per-point radius array used for glyph thickness.",
        "_calc_scalars":                "The resolved per-point scalar array used for scalar coloring.",
        
        "_entity":                      "The PyVista Actor corresponding to this object in the plotter.",
        "_entity_silhouette":           "The PyVista Actor as the silhouette of this object to highlight.",
        
        "_impl_name_pv":            "The unique identifier of this glyph stored in the PyVista plotter.",
        "_impl_resolver_source":    "Field used to drive visual variations (e.g. color, opacity)",
        
        "_impl_figure_ref":          ("A weak reference to the PlotFigure instance containing this glyph."
                                         "To access it, use .fig or ._impl_figure."),
        
        "_state_is_interactable":       "Whether to create a control window when the instance is double right-clicked."
        }
    
    __slots__ = tuple(
            k for k, v in __descriptions__.items() 
            if not v.startswith("Property:") and k not in HostBase.__slots__
        )
    
    _pending_resolution_attrs: List[str] = [
        "radius", "opacity", "color", "scalars"
        ]

    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        coords: np.ndarray,
        opts_type: Type[OptsBase],
        category: str,
        name: str,
        name_replace: str,
        opts: OptsGlyph | None = None ,
        figure: PlotFigure | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger = None,
        **kwargs
            ):
        
        coords = as_points(coords, name="The positions of PlotGlyph") 
        object.__setattr__(self, "raw_coords", coords)
        category = as_str(category, name="The category of the glyph")
        object.__setattr__(self, "raw_category", category)
        
        object.__setattr__(self, "_impl_resolver_source", "raw_coords")
        object.__setattr__(self, "_opts_backup", {})
        object.__setattr__(self, "_state_is_interactable", True)
        
        super().__init__(
            opts_type,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs
            )
        
        if self.opts.paint_by is UNSET:
            if self.opts.color is UNSET and self.opts.scalars is not UNSET:
                self.opts.paint_by = 'scalars'
            elif self.opts.color is not UNSET and self.opts.scalars is UNSET:
                self.opts.paint_by = 'color'
            elif self.opts.color is not UNSET and self.opts.scalars is not UNSET:
                logger.warning("Both 'color' and 'scalars' are provided, but 'paint_by' is not explicitly specified."
                               "The default paint_by strategy will be applied.")

        
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
                logger.recovery("Create a new PlotFigure object and store it in self.fig")
                figure = PlotFigure()
        elif figure is None:
            figure = PlotFigure()
        object.__setattr__(self, "_impl_figure_ref", weakref.ref(figure))
            

        logger.detail("Examining the options before plotting ...")
        self.opts.act_finalize(self._opts_defaults)
        str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        unique_id = self.name + str_now
        object.__setattr__(self, "_impl_name_pv", unique_id)
            
    def _helper_init_end(self):
        
        for attr in self._pending_resolution_attrs:
            self._helper_resolver_spec(attr)
        self._helper_make_figure()
        
        figure = self.fig
        figure.pl.render()
        figure.act_register(self)
        # if figure.pl_type == "P":
        #     figure.pl.show(interactive_update=True)
        
    
    @property
    def fig(self):
        ref = self._impl_figure_ref
        return ref() if ref is not None else None
    
    _impl_figure = fig
        
    
        
    def _helper_setattr_glyph_basic(self, key, value, allowed_extra=[]):
        allowed_extra = ['raw_category', 'category', "raw_coords", "coords"] + list(allowed_extra)
        self._helper_setattr_basic(key, value, allowed_extra=allowed_extra)
        
        
    # ----------------------------------------------------------------------------------------------------
    # Resolver function: to resolve point-wise properties (color, opacity, etc) for each inidividual glyph
    # ----------------------------------------------------------------------------------------------------
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_generic(self, attr_name, attr_input, default_val, is_recover=False, logger=None):
        
        target_shape = (len(self.raw_coords),3) if attr_name=='color' else (len(self.raw_coords),)
        source = getattr(self, self._impl_resolver_source)
        
        try:
            if attr_input is None:
                raise TypeError(f"Require input for {attr_name!r}. Got None instead.")
            elif callable(attr_input):
                resolved = np.asarray(attr_input(source), dtype=np.float32)
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
                resolved = as_ColorRGB_array(resolved, name='The pairwise color data of glyph')
                
            object.__setattr__(self, '_calc_'+attr_name, resolved)
            object.__setattr__(self.opts, attr_name, attr_input)
    
    
        except Exception:
            if is_recover:
                raise ValueError(f"The default value is not valid for {attr_name!r}!")
            else:
                logger.exception(f"Failed to resolve {attr_name!r}")
                if getattr(self, "_entity", None):
                    logger.recovery("Automatically ignore this modification.")
                else:
                    logger.recovery(f"Reset {attr_name!r} to default."
                                    f"To find it, check self._opts_defaults['{attr_name}'].")
                    self._helper_resolver_generic(attr_name, default_val, default_val, is_recover=True)


            
            
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_spec(self, attr_name, attr_value=None, logger=None):
        
        if attr_value is None:
            attr_value = getattr(self.opts, attr_name)
        
        return self._helper_resolver_generic(attr_name, attr_value, self._opts_defaults[attr_name])
    
    
    # ----------------------------------------------------------------------------------------------------
    # Create the polydata, mesh and actor.
    # ----------------------------------------------------------------------------------------------------
    
    def _helper_build_poly(self):
        poly = pv.PolyData(self.raw_coords)
        object.__setattr__(self, "_calc_poly", poly)
        self._helper_set_poly(poly)

    
    def _helper_set_poly(self, poly):
        if hasattr(self, "_calc_radius"):
            poly.point_data['radius'] = self._calc_radius 
        poly.point_data['opacity'] = self._calc_opacity
        poly.point_data['scalars'] = self._calc_scalars
        rgba_values = np.hstack([self._calc_color, self._calc_opacity.reshape(-1, 1)])
        poly.point_data['rgba'] = rgba_values 
        
    
    def _helper_build_mesh(self):
        raise NotImplementedError(...)
    
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_make_figure(self, logger=None):
        """
        Creates or updates the rendering in a PyVista Plotter.
        """
        
        is_scalars = self.opts.paint_by == 'scalars'
        unique_id = self._impl_name_pv
        
        input_dir = {
            "name":             unique_id,
            "pbr":              self.opts.shading_type == 'pbr',
            "rgb":              not is_scalars,
            "scalars":          'scalars' if is_scalars else 'rgba',
            "reset_camera":     self.opts.is_reset_camera,
            }
        if is_scalars:
            input_dir["opacity"] = "opacity"
            input_dir["cmap"] = self.opts.scalars_cmap
            input_dir["show_scalar_bar"] = self.opts.is_scalar_bar
            input_dir["clim"] = self.opts.scalars_clim
            input_dir["scalar_bar_args"] = {"title": self.opts.scalar_bar_title}
            
        logger.detail("Creating glyph polydata and mesh")
        self._helper_build_poly()
        mesh = self._helper_build_mesh()
            
        logger.detail("Removing the existing actor")
        plotter = self.fig.pl
        if unique_id in plotter.actors:
            plotter.remove_actor(unique_id)
        old_actor = getattr(self, "_entity", None)
        if old_actor is not None:
            pm = getattr(self.fig, "_entity_pick_manager", None)
            if pm is not None:
                pm.act_unregister(old_actor)
            
        logger.detail("Visualizing the glyph")
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
        prop.specular_power = self.opts.specular_power
        prop.specular_color = self.opts.specular_color
        
        if shading == 'pbr':
            prop.metallic = self.opts.metallic
            prop.roughness = self.opts.roughness
            
        actor.visibility = self.opts.is_visible
        actor.pickable = self.opts.is_pickable

        object.__setattr__(self, "_entity", actor)
        self._helper_register_pick(actor)
        
        self._helper_add_silhouette()
        
    
    def _helper_add_silhouette(self):
    
        plotter = self.fig.pl

        self._helper_clear_silhouette()
            
        mesh = self._entity.mapper.dataset
        surf = mesh.extract_surface().triangulate().clean()
            
        actor_silhouette = plotter.add_silhouette(
            surf,
            color=(0,0,0),
            line_width=6,
            opacity=0.8,
        )

        actor_silhouette.visibility = False
        actor_silhouette.pickable = False
        
        object.__setattr__(self, "_entity_silhouette", actor_silhouette)

    def _helper_clear_silhouette(self):
        plotter = self.fig.pl
        if getattr(self, "_entity_silhouette", None):
            plotter.remove_actor(self._entity_silhouette)
        object.__setattr__(self, "_entity_silhouette", None)

        
        
        

    # ----------------------------------------------------------------------------------------------------
    # The functios to update the given point-wise data values
    # ----------------------------------------------------------------------------------------------------
    
    def _helper_update_rgba(self):
        mapper = self._entity.mapper
        mapper.scalar_visibility = True
        mapper.color_mode = 'direct'
        mapper.lookup_table = None
        mapper.dataset.set_active_scalars('rgba')
        mapper.SetArrayName('rgba')
        if self.opts.scalar_bar_title in self.fig.pl.scalar_bars.keys():
            self.fig.pl.remove_scalar_bar(title=self.opts.scalar_bar_title)

        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_update_scalars(self, logger=None, **kwargs):
        
        mapper = self._entity.mapper
        mesh_data = mapper.dataset.point_data

        if "__custom_rgba" in mesh_data.keys():
            mesh_data.remove("__custom_rgba")
            
        if not isinstance(mapper.lookup_table, pv.LookupTable):
            mapper.lookup_table = pv.LookupTable()
        
        mapper.set_scalars(
            mesh_data['scalars'], 
            'scalars',
            cmap = self.opts.scalars_cmap,
            clim = self.opts.scalars_clim,
            custom_opac=True,
            opacity=mesh_data['opacity']
            )
        
        if self.opts.scalar_bar_title not in self.fig.pl.scalar_bars.keys():
            self.fig.pl.add_scalar_bar(
                title=self.opts.scalar_bar_title,
                mapper=mapper,
                render=False
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
        self.fig.pl.remove_actor(self._entity)
        pm = getattr(self.fig, "_entity_pick_manager", None)
        if pm:
            pm._impl_registry.pop(self._entity)
        self.fig._entity.remove(self)
        
        
    # ----------------------------------------------------------------------------------------------------
    # The functions to apply modications.
    # ----------------------------------------------------------------------------------------------------
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_pre_opts(self, logger=None, **kwargs):
        
        kwargs = super()._helper_commit_pre_opts(**kwargs)
        
        found, category = pop_exclusive(kwargs, "category", "raw_category")
        if found:
            try:
                category = as_str(category, name=self.__descriptions__["raw_category"])
                object.__setattr__(self, "raw_category", category)
            except:
                logger.exception("Check input.")
                logger.recovery("Automatically ignore this modification.")
        
        is_new_topology = False
        
        found, coords = pop_exclusive(kwargs, "coords", "raw_coords")
        if found:
            try:
                object.__setattr__(self, "raw_coords", as_points(coords))
                is_new_topology = True
            except:
                logger.exception("Invalid input of coords for PlotGlyph.")
                logger.recovery("Ignore this modification in the following")
                
        
        return is_new_topology, kwargs
    
    
    def act_commit(self, opts=None, is_silhouette=True, **kwargs):
        is_new_topology, kwargs = self._helper_commit_pre_opts(**kwargs)
        kwargs = self._helper_merge_opts_kwargs(opts=opts, **kwargs)
        self._helper_commit_apply_opts(is_new_topology, 
                                       is_silhouette=is_silhouette, 
                                       **kwargs)
    
    
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_apply_opts(self, 
                             is_new_topology, 
                             is_silhouette=True,
                             logger=None, 
                             **kwargs):
        
        if not is_new_topology and not kwargs:
            return
        
        logger.detail("Check if a recoloring is requested by input kwargs; if so, determine the paint method")
        paint_method = kwargs.pop('paint_by', None)
        if paint_method is None:
            has_color, has_scalars = 'color' in kwargs, 'scalars' in kwargs
            if has_color ^ has_scalars:   # exactly one is provided
                paint_method = 'scalars' if has_scalars else 'color'
        if paint_method is None:
            paint_method = self.opts.paint_by
        else:
            object.__setattr__(self.opts, 'paint_by', paint_method)
            
            
        current_shading = kwargs.get("shading_type", getattr(self.opts, "shading_type"))
        current_shading = as_str(current_shading, name='shading_type', replace=getattr(self.opts, "shading_type"), pool=('phong', 'pbr'))
        
        
        is_needs_remesh = is_new_topology
        for attr in self._pending_resolution_attrs:
            if attr not in kwargs.keys():
                if is_new_topology: 
                    self._helper_resolver_spec(attr)
            else:
                self._helper_resolver_spec(attr, attr_value=kwargs[attr])
                kwargs.pop(attr)
                is_needs_remesh = True
                
        if "sides" in kwargs:
            object.__setattr__(self.opts, 'sides', kwargs['sides'])
            is_needs_remesh = True
                
        if is_needs_remesh:
            self._helper_build_poly()
            mesh = self._helper_build_mesh()
            self._entity.mapper.SetInputData(mesh)
            self._entity.mapper.Update()
            if is_silhouette:
                self._helper_add_silhouette()
                

        pbr_params = ["metallic", "roughness"]
        phong_params = ["ambient", "diffuse", "specular", "specular_power", "specular_color"]

        for key, value in kwargs.items():
            #!!! is_reset_camera is_colorbar scalars_name
            try:

                if key in pbr_params and current_shading != "pbr":
                    logger.warning(f"Setting '{key}' but current shading_type is '{current_shading}'. PBR effects may not show.")
                elif key in phong_params and current_shading == "pbr":
                    logger.warning(f"Setting '{key}' but current shading_type is 'pbr'. Phong lighting parameters may be ignored.")
                    
                attr_path_actor = self.opts._actor_attr.get(key, None)
                if attr_path_actor:
                    parts = attr_path_actor.split('.')
                    obj = self._entity
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                    
                object.__setattr__(self.opts, key, value)
            except:
                logger.exception(f"Failed to reset value of {key!r}")
                logger.recovery("Ignore this modification")

        
        if paint_method == "color":
            self._helper_update_rgba()
        else:
            self._helper_update_scalars()
            
        self.fig.pl.render()
        
        self._helper_trigger_sync_batch(**kwargs)

         
    
    def act_highlight(self, 
                      color: ColorRGB | None = None,
                      opacity: float | None = None,
                      width: float | None = None):
        
        silhouette = getattr(self, '_entity_silhouette', None)
        
        if silhouette:
            
            self._entity_silhouette.visibility = True
            
            color = as_ColorRGB(color, name="The color of silhouette", replace=None) if color is not None else None
            opacity = as_Number(opacity, name="The opacity of silhouette", value_range=(0,1), replace=None) if opacity is not None else None
            width = as_Number(width, name="The line width of silhouette", value_range=(0,np.inf), replace=None) if width is not None else None
            
            sil_prop = self._entity_silhouette.prop
            if color is not None:
                sil_prop.color = color
            if opacity is not None:
                sil_prop.opacity = opacity
            if width is not None:
                sil_prop.line_width = width
            
    def act_dehighlight(self):
        silhouette = getattr(self, '_entity_silhouette', None)
        if silhouette:
            self._entity_silhouette.visibility = False
            
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})"
        return msg 
    
    
        
