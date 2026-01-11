from dataclasses import dataclass, field, asdict
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
    UNSET,
    Unset,
)
from ..opts import merge_opts_all
from ..class_function import cover_value

#!!! property act_view
#!!! load save 

opts_cam = {"azimuth", "elevation", "roll", "distance", "focal_point"}
opts_bg = {"bg_color", "bg_opacity"}


@dataclass(slots=True)
class OptsFigure:
    name: str = "figure"
    azimuth: float | Unset = UNSET
    elevation: float | Unset = UNSET
    roll: float | Unset = UNSET
    distance: float | Unset = UNSET
    focal_point: Vect(3) | Unset = UNSET
    size_init: Vect(2) = (1920, 1080)
    bg_color: ColorRGB | Unset = UNSET
    bg_opacity: float | Unset = UNSET
    
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
            v, name=self.__descriptions__["name"], replace="figure"
        ),
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

    @logging_and_warning_decorator(start_finish_level=0)
    def __setattr__(self, key, value, logger=None):

        if key == "name" and getattr(self, "_state_is_name_locked", False):
            raise AttributeError(
                f"Name of PlotFigure {self.name!r} could not be modified"
                " because it is used as the key in figure manager"
            )

        if value is not UNSET and key in self._validators:
            old_value = getattr(self, key, None)
            try:
                value = self._validators[key](self, value)
            except:
                logger.exception("Wrong settings for figure camera.")
                logger.recovery("Automatically ignore this modification.")
                value = old_value
        else:
            old_value = None

        object.__setattr__(self, key, value)

        if (
            old_value is not None
            and old_value is not UNSET
            and (key in opts_cam or key in opts_bg)
            and not np.allclose(old_value, value, atol=1e-7)
        ):
            if self._on_change:
                self._on_change()
                
    def act_asdict(self):
        result = {}
        for key in self.__descriptions__.keys():
            result[key] = getattr(self, key)
        return result
            


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
        logger=None,
        **kwargs,
    ):

        if opts is None:
            opts = OptsFigure()
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(self, "opts", opts)

        if plotter is None:
            plotter = pv.Plotter(window_size=self.opts.size_init)
        else:
            if not isinstance(plotter, pv.Plotter):
                try:
                    raise TypeError(
                        "`plotter` for PlotFigure must be PyVista plotter object, or None."
                    )
                except:
                    logger.exception("Check input")
                    logger.recovery("Create a new figure instead.")
                    plotter = pv.Plotter(window_size=self.opts.size_init)

        object.__setattr__(self, "_entities_plotter", plotter)
        object.__setattr__(self, "_entities", {})

        self._helper_sync_from_plotter(is_cover_set=False)
        self.act_commit(is_init=True, opts=self.opts)
        self.opts._on_change = self._helper_sync_from_opts

        def _on_interaction_end(obj, event):
            self._helper_sync_from_plotter()

        self.pl.iren.add_observer("EndInteractionEvent", _on_interaction_end)

    @logging_and_warning_decorator(start_finish_level=5)
    def act_commit(self, 
                   is_init: bool = False, 
                   opts: OptsFigure | None = None,
                   logger=None, 
                   **kwargs):
        
        if opts is not None:
            if not isinstance(opts, OptsFigure):
                try:
                    raise TypeError(f"`opts` must be OptsFigure object. Got type={type(opts)} instead")
                except:
                    logger.exception("Check input.")
                    logger.recovery("Ignoring `opts` and using `kwargs` only.")
                    opts = {}
            else:
                opts = opts.act_asdict()
                overlap = opts.keys() & kwargs.keys()
                if overlap:
                    logger.warning(
                        f"Overlapping configuration detected: {list(overlap)}. "
                        f"The values in **kwargs will take precedence over `opts`.",
                        )
        else:
            opts = {}
            
        merged_opts = opts | kwargs
        if not merged_opts:
            logger.warning("No configuration provided for commit.")
            return

        for key, value in merged_opts.items():

            try:

                if key.startswith("_"):
                    if not is_init:
                        msg = (
                            "Attributes prefixed with `_` are not intended to be modified manually."
                            "If you are certain about the consequences, modify it via explicit direct assignment."
                        )
                        raise AttributeError(msg)
                    continue

                if key not in OptsFigure.__descriptions__:
                    raise AttributeError(
                        f"Unknown attribute: {key} in class: PlotFigure.opts"
                    )

                if key == "size_init" and not is_init:
                    raise AttributeError(
                        f"The figure {self.opts.name!r} is already initialized."
                        " The size could not be changed."
                    )

                if not is_init:
                    if value is not UNSET:
                        object.__setattr__(self.opts, key, value)

            except:
                logger.exception("Wrong setting.")
                logger.recovery("Automatically ignore this modification.")

        self._helper_sync_from_opts(is_cam=True, is_bg=True)

    @property
    def pl(self):
        return self._entities_plotter

    def _helper_sync_from_plotter(self, is_cover_set=True):

        cb = self.opts._on_change
        self.opts._on_change = None

        camera = self.pl.camera
        temp = self._helper_convert_pos_to_spherical(
            camera.position, camera.focal_point, camera.up
        )

        alter = {
            "focal_point":  camera.focal_point,
            "azimuth":      temp[0],
            "elevation":    temp[1],
            "roll":         temp[2],
            "distance":     temp[3],
            "bg_color":     self.pl.background_color.float_rgb,
            "bg_opacity":   self.pl.background_color.opacity / 255.0,
        }

        cover_value(self.opts, is_cover_set=is_cover_set, **alter)

        self.opts._on_change = cb

    def _helper_sync_from_opts(self, is_cam=True, is_bg=False):
        if is_cam:
            camera = self.pl.camera
            pos, focal, up = self._helper_convert_spherical_to_pos(
                self.opts.azimuth,
                self.opts.elevation,
                self.opts.roll,
                self.opts.distance,
                self.opts.focal_point,
            )
            camera.position = pos
            camera.focal_point = focal
            camera.up = up
            self.pl.render()

        if is_bg:
            rgba = np.r_[self.opts.bg_color, [self.opts.bg_opacity]] * 255
            rgba = rgba.astype(int)
            self.pl.background_color = rgba

    @staticmethod
    def _helper_convert_pos_to_spherical(position, focal_point, view_up):

        pos = np.array(position)
        foc = np.array(focal_point)
        up = np.array(view_up)
        vec = pos - foc

        dist = np.linalg.norm(vec)

        if dist < 1e-9:
            return 0.0, 0.0, 0.0, 0.0, foc

        elevation = np.degrees(np.arcsin(vec[2] / dist))

        az_rad = np.arctan2(vec[1], vec[0])
        azimuth = np.degrees(az_rad) % 360

        view_dir = -vec / dist
        right = np.cross(view_dir, [0, 1, 0])
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(view_dir, [0, 0, 1])
        right /= np.linalg.norm(right)
        up_ref = np.cross(right, view_dir)
        roll = np.degrees(np.arctan2(np.dot(up, right), np.dot(up, up_ref)))

        return azimuth, elevation, roll, dist, foc

    @staticmethod
    def _helper_convert_spherical_to_pos(
        azimuth, elevation, roll, distance, focal_point
    ):

        az = np.radians(azimuth)
        el = np.radians(elevation)
        r = np.radians(roll)
        focal = np.asarray(focal_point, dtype=float)

        x = distance * np.cos(el) * np.cos(az)
        y = distance * np.cos(el) * np.sin(az)
        z = distance * np.sin(el)
        pos = focal + np.array([x, y, z])

        view_dir = focal - pos
        view_dir = view_dir / np.linalg.norm(view_dir)
        up_candidate = np.array([0, 1, 0])
        up_proj = up_candidate - np.dot(up_candidate, view_dir) * view_dir
        if np.linalg.norm(up_proj) != 0:
            up = up_proj / np.linalg.norm(up_proj)
        else:
            up = np.array([0, 1, 0])

        if abs(r) > 1e-8:
            k = view_dir
            cos_r, sin_r = np.cos(r), np.sin(r)
            up = up * cos_r + np.cross(k, up) * sin_r + k * np.dot(k, up) * (1 - cos_r)

        return pos, focal, up

    def act_reset_camera(self):
        self.pl.reset_camera()
        self._helper_sync_from_plotter()

    def act_view_xy(self):
        self.pl.view_xy()
        self._helper_sync_from_plotter()

    def act_view_xz(self):
        self.pl.view_xz()
        self._helper_sync_from_plotter()

    def act_view_yz(self):
        self.pl.view_yz()
        self._helper_sync_from_plotter()

    def act_view_isometric(self):
        self.pl.view_isometric()
        self._helper_sync_from_plotter()

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
    
    def __getitem__(self, name: str):
        mapping = {
            entity.opts.name: entity
            for entity_list in self._entities.values()
            for entity in entity_list
            }
        
        return mapping.get(name)
