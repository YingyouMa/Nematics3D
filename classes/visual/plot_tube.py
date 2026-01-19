from dataclasses import dataclass, field, fields
from typing import Callable, Sequence, Literal, Any, Mapping, ClassVar
import numpy as np
import pyvista as pv
import datetime
import weakref
from types import MappingProxyType

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import UNSET, Unset, as_bool, as_Number, as_points, as_ColorRGB_array, as_str
from .plot_figure import PlotFigure
from .glyph import OptsGlyph, LEVEL_ACTOR, LEVEL_RECALC, LEVEL_REMESH
from ..opts import merge_opts_all, build_defaults_with_override
from Nematics3D.general import pop_exclusive

#! scalars_limit
#! scalars_bar     change is_scalars_bar
#! clip_geometry
#! light dark pbr

#! only change cmap

#! info log extra attr
#1 del
#! orphan figure

#! @coords

#! test
#! color invalid


@dataclass(slots=True)
class OptsTube(OptsGlyph):

    # --- Geometry & Topology (Tube-specific) ---
    is_capping:             bool | Unset = UNSET
    smooth_iter:            int | Unset  = UNSET


    __descriptions__: ClassVar[Mapping[str, str]] = {
        **dict(OptsGlyph.__descriptions__),
        "is_capping":        "Whether to close the ends of the tube.",
        "smooth_iter":       "Path smoothing iterations to remove jagged edges.",
    }

    _validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **dict(OptsGlyph._validators),
        "is_capping":        lambda v, d: as_bool(v, name=d),
        "smooth_iter":       lambda v, d: as_Number(v, name=d, is_int=True, value_range=(0, 1000), bounded=True),
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType({
        **dict(OptsGlyph._DEFAULTS_FROZEN),
        "name":              "tube",
        "category":          "line",
        "is_capping":        True,
        "smooth_iter":       0,
    })

    _commit_level: ClassVar[Mapping[str, Any]] = {
        **dict(OptsGlyph._commit_level),
        "is_capping":        LEVEL_REMESH,
        "smooth_iter":       LEVEL_REMESH,
    }


        
        
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
    
        "_calc_poly": "The generated PyVista PolyData representing the polyline(s) ",
        "_calc_mesh": "The generated PyVista surface mesh representing the tube ",
    
        "_calc_color": "The resolved per-point RGB color array of the tube.",
        "_calc_opacity": "The resolved per-point opacity array of the tube.",
        "_calc_radius": "The resolved per-point radius array used for tube thickness.",
        "_calc_scalars": "The resolved per-point scalar array used for scalar coloring.",
    
        "_entity": "The PyVista Actor corresponding to this tube in the plotter.",
        
        
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
    
    __slots__ = tuple(__descriptions__.keys()) + ("__weakref__",)
    

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
        
        if opts is None:
            opts = OptsTube()
        
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
                    raise TypeError('`figure` for PlotTube must be PlotFigure object!')
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

            
        if line_index is not None:
            try:
                line_index = self._helper_check_index(line_index)
            except:
                logger.exception("Invalid `line_index` input")
                logger.recovery("Set line_index=None in the following (no stop points within the tube)")
                line_index = None
        object.__setattr__(self, "raw_line_index", line_index)
            
        self._helper_resolver_init()
        self._helper_make_figure()
        
        figure.pl.render()
        figure.pl.show(interactive_update=True)
        object.__setattr__(self.opts, '_state_is_category_locked', True)
        figure._helper_register_entity(self, self.opts.category, self.opts.is_reset_camera)
        
    @property
    def _internal_owner(self):
        return self._internal_owner_ref()
    
    @property
    def owner(self):
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
    
        allowed_core = ("raw_coords", "coords", "raw_line_index", "line_index")
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
        
        is_set_success = False
        
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
                resolved = as_ColorRGB_array(resolved, name='The pairwise color data of tube')
                
            object.__setattr__(self, '_calc_'+attr_name, resolved)
            is_set_success = True
    
    
        except:
            logger.exception(f"Failed to resolve {attr_name!r}")
            if self.opts._state_is_functioning:
                logger.recovery("Automatically ignore this modification.")
                is_set_success = False
            else:
                resolved = np.full(target_shape, default_val, dtype=np.float32)
                logger.recovery(f"Reset {attr_name!r} to default: {default_val} everywhere.")
                object.__setattr__(self.opts, attr_name, default_val)
                object.__setattr__(self, '_calc_'+attr_name, resolved)
    
        return is_set_success
              
        
            
            
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_resolver_init(self, extra={}, logger=None):
        logger.detail("Resolving data for color, opacity and radius")
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
        object.__setattr__(self, "_calc_mesh", mesh)
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
            
        logger.detail("Removing the existing actor")
        plotter = self._internal_owner.pl
        if unique_id in plotter.actors:
            plotter.remove_actor(unique_id)
        old_actor = getattr(self, "_entity", None)
        if old_actor is not None:
            pm = getattr(self._internal_owner, "_entity_pick_manager", None)
            if pm is not None:
                pm.act_unregister(old_actor)
            
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

        object.__setattr__(self, "_entity", actor)
        self._helper_register_pick(actor)
        
        
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
        
        
    @logging_and_warning_decorator()
    def act_commit(self, logger=None, **kwargs):
        
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
        if ('scalars' in kwargs.keys()
            or 'scalars_cmap' in kwargs.keys()
            or 'scalars_clim' in kwargs.keys()
            ):
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
            
            is_set_success = False
            
            try:
                if key not in self.opts.__descriptions__.keys():
                        raise ValueError(f"Unknown attribute: {key} in class: PlotTube.opts")
                
                level = self.opts._commit_level.get(key, LEVEL_ACTOR)
                attr_path_actor = self.opts._actor_attr.get(key, None)
    
                # Dealing with LEVEL ACTOR (simply resetting values)
                if level == LEVEL_ACTOR:
                    
                    if key == "category":
                        raise AttributeError("Modification of 'category' is not allowed, because it is used as the key in dir: PlotFigure._entity")
                    
                    if key == "name":
                        msg = "Changing 'name' of PlotTube object is not recommended because: \n"
                        msg += "1) There is no guarantee that name collisions will be avoided in PlotFigure._entity; and\n"
                        msg += "2) The corresponding actor name stored in the PyVista renderer cannot be updated accordingly."
                        logger.warning(msg)
                        is_set_success = True
                        continue
                    
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
                        obj = self._entity
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], value)
                        is_set_success = True
                        continue
                
                # Dealing with LEVEL_RECALC (resolver for color, opacity and scalars)
                elif level == LEVEL_RECALC:
                    if key in ['radius', 'color', 'opacity', 'scalars']:
                        if key == 'color' and value == 'scalars':
                            self._helper_resolver_spec('scalars')
                            is_set_success = True
                        else:
                            is_set_success = self._helper_resolver_spec(key, attr_value=value)
                    else:
                        is_set_success = True
                    continue
    
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
                
        self._internal_owner.pl.render()
        
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
            

    def _helper_register_pick(self, actor):

        fig = self._internal_owner
        if fig is None:
            return

        pm = getattr(fig, "_entity_pick_manager", None)
        if pm is None:
            return

        pm.act_register(actor=actor, owner=self)            

            
    def act_remove(self):
        self.owner.pl.remove_actor(self._entity)
            
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}({self.opts.name!r})"
        return msg 

            
                
        
