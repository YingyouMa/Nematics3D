from dataclasses import dataclass, field
import numpy as np
import pyvista as pv

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    as_Number,
    as_str,
    as_bool,
    Vect,
    as_Vect,
)
from ..opts import merge_opts_all


#!!! property act_view

opts_cam = {"azimuth", "elevation", "roll", "distance", "focal_point"}
opts_bg = {"bg_color", "bg_opacity"}


@dataclass(slots=True)
class OptsFigure:
    name: str = 'figure'
    azimuth: float = 0.0
    elevation: float = 0.0
    roll: float = 0.0
    distance: float = 10.0
    focal_point: Vect(3) = (0.0, 0.0, 0.0)
    size_init: Vect(2) = (1920, 1080)
    bg_color: ColorRGB = (1, 1, 1)
    bg_opacity: float = 0
    _state_is_name_locked: bool = False

    _on_change: callable = field(default=None, repr=False, compare=False)

    __descriptions__ = {
        "name": "The name of figure",
        "azimuth": "The azimuthal angle (degrees) of the camera around the focal point.",
        "elevation": "The elevation angle (degrees) of the camera relative to the focal plane.",
        "roll": "The rotation (degrees) of the camera about the direction of projection.",
        "distance": "The distance from the camera position to the focal point.",
        "focal_point": "The point the camera is looking at (x, y, z).",
        "size_init": "The window size of figure (ONLY valid during initialization)",
        "bg_color": "The background color of figure",
        "bg_opacity": "The background opacity of figure",
    }

    _validators = {
        "name": lambda self, v: as_str(
            v, name=self.__descriptions__["name"], replace='figure'),
        "azimuth": lambda self, v: as_Number(
            v, name=self.__descriptions__["azimuth"], value_range=(0, 360)
        ),
        "elevation": lambda self, v: as_Number(
            v,
            name=self.__descriptions__["elevation"],
            value_range=(-90, 90),
        ),
        "roll": lambda self, v: as_Number(
            v, name=self.__descriptions__["roll"], value_range=(-180, 180)
        ),
        "distance": lambda self, v: as_Number(
            v, name=self.__descriptions__["distance"], value_range=(0, np.inf)
        ),
        "focal_point": lambda self, v: as_Vect(
            v, name=self.__descriptions__["focal_point"], dim=3
        ),
        "size_init": lambda self, v: as_Vect(
            v, name=self.__descriptions__["size_init"], dim=2, replace=(1920, 1080)
        ),
        "bg_color": lambda self, v: as_ColorRGB(
            v, name=self.__descriptions__["bg_color"], replace=(0, 0, 0)
        ),
        "bg_opacity": lambda self, v: as_Number(
            v,
            name=self.__descriptions__["bg_opacity"],
            value_range=(0, 1),
            bounded=True,
            replace=0,
        ),
    }

    def __setattr__(self, key, value):
        
        if key == 'name' and getattr(self, "_state_is_name_locked", False):
            raise AttributeError(
                f"Name of PlotFigure {self.name!r} could not be modified"
                " because it is used as the key in figure manager")
            
        if key in self._validators:
            value = self._validators[key](self, value)
            old_value = getattr(self, key, None)
        else:
            old_value = None

        object.__setattr__(self, key, value)
        
        if (
            old_value is not None
            and (key in opts_cam or key in opts_bg)
            and not np.allclose(old_value, value, atol=1e-7)
        ):
            if self._on_change:
                self._on_change(key, value)


class PlotFigure:

    __descriptions__ = {
        "opts": "The OptsFigure object controlling the options beyond specific actors (glyphs)",
        "_entities_plotter": "The underlying PyVista Plotter instance that owns the VTK rendering pipeline. ",
        "_entities": "A registry (dict) for objects attached to this figure.",
    }

    __slots__ = tuple(__descriptions__.keys()) + ("__weakref__",)
    
    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        plotter: pv.Plotter | None = None,
        opts: OptsFigure | None = None,
        logger = None,
        **kwargs
    ):

        if plotter is None:
            if opts is None:
                opts = OptsFigure()
            opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
            plotter = pv.Plotter(window_size=opts.size_init)
            rgba = np.r_[ opts.bg_color, [opts.bg_opacity] ] * 255
            rgba = rgba.astype(int)
            plotter.background_color = rgba
            object.__setattr__(self, "_entities_plotter", plotter)
        else:
            object.__setattr__(self, "_entities_plotter", plotter)
            if opts is not None or not (len(kwargs) == 1 and 'name' in kwargs):
                msg = (
                    "Since 'plotter' is provided, this constructor will not create/configure a new Plotter. "
                    "Any plotter-related fields in 'opts' or 'kwargs' except `name` are ignored. "
                    "Configure the provided Plotter instance directly if needed."
                )
                logger.warning(msg)
            if opts is None:
                opts = OptsFigure()
            opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        
        object.__setattr__(self, "opts", opts)
        object.__setattr__(self, "_entities", {})
        
        self._helper_sync_from_plotter()
        self.opts._on_change = self._helper_sync_from_opts
        
        def _on_interaction_end(obj, event):
            self._helper_sync_from_plotter()
            
        self.pl.iren.add_observer('EndInteractionEvent', _on_interaction_end)

    @property
    def pl(self):
        return self._entities_plotter

            
    def _helper_sync_from_plotter(self):
        
        cb = self.opts._on_change
        self.opts._on_change = None
        
        camera = self.pl.camera
        
        self.opts.roll = camera.roll
        self.opts.focal_point = camera.focal_point
        
        temp = self._helper_convert_pos_to_spherical(camera.position, 
                                                     camera.focal_point)
        self.opts.azimuth = temp[0]
        self.opts.elevation = temp[1]
        self.opts.distance = temp[2]
        
        self.opts.bg_color = self.pl.background_color.float_rgb
        self.opts.bg_opacity = self.pl.background_color.opacity / 255
        
        self.opts._on_change = cb
        
    def _helper_sync_from_opts(self, key, value):
        if key in opts_cam: 
            camera = self.pl.camera
            if key == 'distance': #!!! comment
                pos = self._helper_convert_spherical_to_pos(
                    self.opts.azimuth, self.opts.elevation, self.opts.distance, self.opts.focal_point
                    )
                setattr(camera, 'position', pos)
            else:
                setattr(camera, key, value)
            self.pl.render()

        elif key in opts_bg:
            rgba = np.r_[ self.opts.bg_color, [self.opts.bg_opacity] ] * 255
            rgba = rgba.astype(int)
            self.pl.background_color = rgba
            
        
    @staticmethod
    def _helper_convert_pos_to_spherical(position, focal_point):

        pos = np.array(position)
        foc = np.array(focal_point)
        vec = pos - foc
        
        dist = np.linalg.norm(vec)
        
        if dist < 1e-9:
            return 0.0, 0.0, 0.0, focal_point

        elevation = np.degrees(np.arcsin(vec[2] / dist))

        az_rad = np.arctan2(vec[1], vec[0])
        azimuth = np.degrees(az_rad) % 360  # 转为 [0, 360)
        
        return azimuth, elevation, dist
    
    @staticmethod
    def _helper_convert_spherical_to_pos(azimuth, elevation, distance, focal_point):

        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
    
        z = distance * np.sin(el_rad)
        
        rcos_el = distance * np.cos(el_rad)
        
        x = rcos_el * np.cos(az_rad)
        y = rcos_el * np.sin(az_rad)
    
        pos = np.array(focal_point) + np.array([x, y, z])
        
        return pos
    
    
    def _helper_register_entity(
        self, entity_instance, entity_category, is_reset_camera
    ):
        if entity_category in self._entities.keys():
            self._entities[entity_category].append(entity_instance)
        else:
            self._entities[entity_category] = [entity_instance]
        if is_reset_camera:
            self._helper_sync_from_plotter()

    def act_get_entities_names(self):
        names = [
            entity.opts.name
            for entity_list in self._entities.values()
            for entity in entity_list
        ]
        return names

    def act_check_is_alive(self):
        try:
            plotter = self._entities_plotter
            if plotter._closed:
                return False

            iren = plotter.iren
            return iren is not None and iren.initialized
        except Exception:
            return False

    def __bool__(self):
        return self.act_check_is_alive()
