from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Literal, Mapping, Sequence, Type
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
from Nematics3D.general import pop_exclusive, is_given_str
from ..opts import merge_opts_all, build_dict_override

#!!! resolver source

LEVEL_ACTOR = 0  # Only changes GPU/Rendering state. (Fastest)
LEVEL_RECALC = 1  # Needs to re-calculate data arrays (colors, etc.) but keeps geometry.
LEVEL_REMESH = 2  # Needs to re-run the glyph filter to rebuild the 3D mesh. (Heaviest)

# --- Type aliases ---
ColorMode = ColorRGB | Callable | Sequence | Literal["scalars"]
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
    specular_pow:               float | Unset                       = UNSET
    specular_color:             ColorRGB | Unset                    = UNSET

    # --- PBR Lighting ---
    metallic:                   float | Unset                       = UNSET
    roughness:                  float | Unset                       = UNSET

    # --- Shape & Color ---
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
        "specular_pow":         "Focus of gloss (1-100). Higher = shinier/smaller spot.",
        "specular_color":       "The color of the glossy highlight (RGB). Usually white [1,1,1].",
        
        # === Lighting - PBR ===
        "metallic":             "PBR metallic effect (0-1). Needs PBR enabled.",
        "roughness":            "PBR surface roughness (0-1). Needs PBR enabled.",
        
        # === Shape and Color Control ===
        "color": (
            "Determines point colors. Options: "
            "1) ColorRGB for entire glyph (e.g. (1,0,0)) "
            "2) Function (mapping function), "
            "3) color data set manually, "
            "4) 'scalars' (maps 1D data to colors using scalars_cmap/scalars_clim)."
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
        "tag":                  "glyph options",
        "is_visible":           True,
        "is_pickable":          True,
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
        "is_pickable":          "pickable",
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
        self._helper_setattr_basic(key, value)

    def act_finalize(self, defaults: Mapping[str, Any] | None = None) -> None:
        self._helper_finalize_basic(defaults)

    def act_asdict(self, is_include_UNSET: bool = False) -> dict[str, Any]:
        return self._helper_asdict_basic(is_include_UNSET=is_include_UNSET)
    
    def _helper_owner_apply(self, key, value):
        owner = self._internal_owner_ref()
        if owner is not None:
            owner.act_commit(**{key: value})
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
        "_internal_name_pv":            "The unique identifier of this glyph stored in the PyVista plotter.",
        "_internal_resolver_source":    "Field used to drive visual variations (e.g. color, opacity)",
        
        "_internal_owner_ref":          ("A weak reference to the PlotFigure instance containing this glyph."
                                         "To access it, use .owner or ._internal_owner."),
        }

    
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
        
        object.__setattr__(self, "_internal_resolver_source", "raw_coords")
        
        super().__init__(
            opts_type,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=name_replace,
            **kwargs
            )
        
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
            

        logger.detail("Examining the options before plotting ...")
        self.opts.act_finalize(self.opts_defaults)
        str_now = datetime.datetime.now().strftime("_%Y/%m/%d_%H:%M:%S.%f")[:-4]
        unique_id = self.name + str_now
        object.__setattr__(self, "_internal_name_pv", unique_id)
        
        
        if not (isinstance(self.opts.color, str) and self.opts.color == 'scalars') and self.opts.scalars not in (None, UNSET):
            msg = "Color input of PlotGlyph is not set to 'scalars'. However, scalars is provided.\n"
            msg += "The scalars data will be ignored unless color='scalars' is explicitly specified."
            logger.warning(msg)
            
    def _helper_init_end(self):
        figure = self.owner
        figure.pl.render()
        figure.pl.show()
        figure.act_register(self)
        
        
        
    
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_setattr_glyph_basic(self, key, value, allowed_extra=[], logger=None):
        allowed_extra = ['raw_category', 'category', "raw_coords", "coords"] + list(allowed_extra)
        self._helper_setattr_basic(key, value, allowed_extra=allowed_extra)
        
        
    # ----------------------------------------------------------------------------------------------------
    # Resolver function: to resolve point-wise properties (color, opacity, etc) for each inidividual glyph
    # ----------------------------------------------------------------------------------------------------
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_generic(self, attr_name, attr_input, default_val, logger=None):
        
        target_shape = (len(self.raw_coords),3) if attr_name=='color' else (len(self.raw_coords),)
        source = getattr(self, self._internal_resolver_source)
        
        is_set_success = False
        
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
            is_set_success = True
    
    
        except:
            logger.exception(f"Failed to resolve {attr_name!r}")
            if getattr(self, "_entity", None):
                logger.recovery("Automatically ignore this modification.")
                is_set_success = False
            else:
                if attr_name=="scalars" and default_val is None:
                    default_val = lambda x: np.linalg.norm(x, axis=-1)
                    resolved = default_val(self.raw_coords)
                    logger.recovery(f"Reset {attr_name!r} to default: the distance of each point to origin.")
                else:
                    resolved = np.full(target_shape, default_val, dtype=np.float32)
                    logger.recovery(f"Reset {attr_name!r} to default: {default_val} everywhere.")
                    object.__setattr__(self.opts, attr_name, default_val)
                object.__setattr__(self, '_calc_'+attr_name, resolved)
    
        return is_set_success
    
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_init(self, extra=[], logger=None):
        logger.detail("Resolving data for color, opacity, radius ...")
        
        self._helper_resolver_spec('opacity')
        self._helper_resolver_spec('radius')
        for attr in extra:
            self._helper_resolver_spec(attr)
        
        if isinstance(self.opts.color, str) and self.opts.color == 'scalars':
            self._helper_resolver_spec('scalars')
        else:
            self._helper_resolver_spec('color')
            
            
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_spec(self, attr_name, attr_value=None, logger=None):
        
        if attr_value is None:
            attr_value = getattr(self.opts, attr_name)
        
        return self._helper_resolver_generic(attr_name, attr_value, self.opts_defaults[attr_name])
    
    
    # ----------------------------------------------------------------------------------------------------
    # Create the mesh and actor.
    # ----------------------------------------------------------------------------------------------------
    
    
    def _helper_build_mesh(self):
        raise NotImplementedError(...)
    
    @logging_and_warning_decorator(start_finish_level=5)    
    def _helper_make_figure(self, logger=None):
        """
        Creates or updates the rendering in a PyVista Plotter.
        """
        
        is_scalars = (isinstance(self.opts.color, str) and self.opts.color == 'scalars')
        unique_id = self._internal_name_pv
        
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
            input_dir["scalar_bar_args"] = {"title": self.opts.scalar_bar_title}
            input_dir["clim"] = self.opts.scalars_clim
            
        logger.detail("Creating glyph mesh")
        mesh = self._helper_build_mesh()
            
        logger.detail("Removing the existing actor")
        plotter = self._internal_owner.pl
        if unique_id in plotter.actors:
            plotter.remove_actor(unique_id)
        old_actor = getattr(self, "_entity", None)
        if old_actor is not None:
            pm = getattr(self._internal_owner, "_entity_pick_manager", None)
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
        prop.specular_power = self.opts.specular_pow
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
    
        plotter = self.owner.pl

        silhouette_id = f"{self._internal_name_pv}__silhouette"
        if silhouette_id in plotter.actors:
            plotter.remove_actor(silhouette_id) 
            
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
        
        
        

    # ----------------------------------------------------------------------------------------------------
    # The functios to update the given point-wise data values
    # ----------------------------------------------------------------------------------------------------
        
        
    def _helper_replace_data_pv(self, attr: str, data: np.ndarray):
        mesh = self._entity.mapper.dataset
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
        mapper = self._entity.mapper
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
        
        mapper = self._entity.mapper
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
            
        
    # ----------------------------------------------------------------------------------------------------
    # The register and un-register of glyphs onto PlotFigure instance
    # ----------------------------------------------------------------------------------------------------


    def _helper_register_pick(self, actor):

        fig = self.owner
        if fig is None:
            return
        pm = getattr(fig, "_entity_pick_manager", None)
        if pm is None:
            return
        pm.act_register(actor=actor, owner=self)   
        
    def act_remove(self):
        self.owner.pl.remove_actor(self._entity)
        self.owner.pick_manager._internal_registry.pop(self._entity)
        
        
    # ----------------------------------------------------------------------------------------------------
    # The functions to apply modications.
    # ----------------------------------------------------------------------------------------------------
        
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_pre_opts(self, logger=None, **kwargs):
        
        kwargs = super()._helper_commit_pre_opts(**kwargs)
        
        found, category, kwargs = pop_exclusive(kwargs, "name", "raw_category")
        if found:
            try:
                category = as_str(category, name=self.__descriptions__["raw_category"])
            except:
                logger.exception("Check input.")
                logger.recovery("Automatically ignore this modification.")
        
        is_needs_remesh = False
        
        found, coords, kwargs = pop_exclusive(kwargs, "coords", "raw_coords")
        if found:
            try:
                object.__setattr__(self, "raw_coords", as_points(coords))
                is_needs_remesh = True
            except:
                logger.exception("Invalid input of coords for PlotGlyph.")
                logger.recovery("Ignore this modification in the following")
                
        
        return is_needs_remesh, kwargs
    
    
    def act_commit(self, opts=None, **kwargs):
        is_needs_remesh, kwargs = self._helper_commit_pre_opts(**kwargs)
        kwargs = self._helper_merge_opts_kwargs(opts=opts, **kwargs)
        self._helper_commit_apply(is_needs_remesh, **kwargs)
    
    
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_apply(self, 
                             is_needs_remesh, 
                             attr_resolve_extra=[], 
                             is_radius=True,
                             logger=None, **kwargs):
        
        attr_resolve = ['radius', 'color', 'opacity'] + list(attr_resolve_extra)

        if not is_radius:
            attr_resolve.remove('radius')
        
        if is_needs_remesh:
            for attr in attr_resolve:
                if attr not in kwargs.keys():
                    if attr == 'color' and is_given_str(self.opts.color, 'scalars'):
                        self._helper_resolver_spec('scalars')
                    else:
                        self._helper_resolver_spec(attr)
        
        current_shading = kwargs.get("shading_type", getattr(self.opts, "shading_type"))
        current_shading = as_str(current_shading, name='shading_type', replace=getattr(self.opts, "shading_type"), pool=('phong', 'pbr'))
        
        color_method = None
        if ('scalars' in kwargs.keys()
            or 'scalars_cmap' in kwargs.keys()
            or 'scalars_clim' in kwargs.keys()
            ) and kwargs['scalars'] is not None:
            if 'color' in kwargs.keys() and not is_given_str(self.opts.color, 'scalars'):
                logger.warning("You are attempting to modify both 'color' and 'scalars' simultaneously. "
                               "This is a potentially confusing operation."
                               "The values will be updated accordingly, but rendering will use 'scalars' for coloring.")
            color_method = 'scalars'
        elif 'color' in kwargs.keys():
            color_method = 'scalars' if is_given_str(kwargs['color'], 'scalars') else 'color'
        elif 'opacity' in kwargs.keys(): 
            color_method = 'scalars' if is_given_str(self.opts.color, 'scalars') else 'color'


        for key, value in kwargs.items():

            if key == 'scalars' and value is None:
                continue
             
            is_set_success = False
            
            try:
                if key not in self.opts.__descriptions__.keys():
                    raise ValueError(f"Unknown attribute: {key} in class: PlotGlyph.opts")
                
                level = self.opts._commit_level.get(key, LEVEL_ACTOR)
                attr_path_actor = self.opts._actor_attr.get(key, None)
    
                # Dealing with LEVEL ACTOR (simply resetting values)
                if level == LEVEL_ACTOR:
                    
                    # if key in "is_reset_camera":
    
                    pbr_params = ["metallic", "roughness"]
                    phong_params = ["ambient", "diffuse", "specular", "specular_pow", "specular_color"]
                    
                    if key in pbr_params and current_shading != "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is '{current_shading}'. PBR effects may not show.")
                    elif key in phong_params and current_shading == "pbr":
                        logger.warning(f"Setting '{key}' but current shading_type is 'pbr'. Phong lighting parameters may be ignored.")
    
                    if attr_path_actor and not is_needs_remesh:
                        parts = attr_path_actor.split('.')
                        obj = self._entity
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value)

                    is_set_success = True
                    
                
                # Dealing with LEVEL_RECALC (resolver for color, opacity and scalars)
                elif level == LEVEL_RECALC:
                    if key in attr_resolve:
                        if key == 'color' and is_given_str(kwargs['color'], 'scalars'):
                            if kwargs.get('scalars') is not None:
                                is_set_success = self._helper_resolver_spec(
                                    'scalars', 
                                    attr_value=kwargs.get('scalars')
                                    )
                            else:
                                is_set_success = self._helper_resolver_spec('scalars')
                        else:
                            is_set_success = self._helper_resolver_spec(key, attr_value=value)
                    else:
                        is_set_success = True
                    
    
                # Dealing with LEVEL_REMESH (Geometry)
                elif level == LEVEL_REMESH:
                    is_needs_remesh = True
                    if key == 'radius':
                        is_set_success = self._helper_resolver_spec('radius', attr_value=value)
                    else:
                        is_set_success = True
        
            except:
                logger.exception(f"Failed to reset value of {key!r}")
                logger.recovery("Ignore this modification")
                is_set_success = False
        
            if is_set_success:
                object.__setattr__(self.opts, key, value)
                
        if is_needs_remesh:
            self._helper_make_figure()
        else:
            if color_method == 'scalars':
                self._helper_update_scalars()
            elif color_method == 'color':
                self._helper_update_rgba()
                
        self.owner.pl.render()

         
    
    def act_highlight(self, 
                      color: ColorRGB | None = None,
                      opacity: float | None = None,
                      width: float | None = None):
        
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
        self._entity_silhouette.visibility = False
            
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.name!r})"
        return msg 
    
    
        
