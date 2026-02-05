from dataclasses import dataclass, field
import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import BackgroundPlotter
from types import MappingProxyType
from typing import Mapping, Any
from PyQt5 import QtCore
import weakref

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    as_Number,
    Vect,
    as_Vect,
    UNSET,
    Unset,
)
from ..host_base import OptsBase, HostBase
from ..class_function import cover_value
from ..registry_base import RegistryBase
from .pick_manager import PickManager
from .qt.console import ScopedConsoleDock


#!!! property act_view
#!!! load save 
#!!! overlay blanket
#!!! name figure manager registry

# opts_cam = {"azimuth", "elevation", "roll", "distance", "focal_point"}
# opts_bg = {"bg_color", "bg_opacity"}

@dataclass(slots=True, repr=False)
class OptsFigure(OptsBase):
    azimuth:                float | Unset       = UNSET
    elevation:              float | Unset       = UNSET
    roll:                   float | Unset       = UNSET
    distance:               float | Unset       = UNSET
    focal_point:            Vect(3) | Unset     = UNSET
    size:                   Vect(2) | Unset     = UNSET
    bg_color:               ColorRGB | Unset    = UNSET
    bg_opacity:             float | Unset       = UNSET

    __descriptions__ = {
        **(OptsBase.__descriptions__),
        "azimuth":          "The azimuthal angle (degrees) of the camera around the focal point.",
        "elevation":        "The elevation angle (degrees) of the camera relative to the focal plane.",
        "roll":             "The rotation (degrees) of the camera about the direction of projection.",
        "distance":         "The distance from the camera position to the focal point.",
        "focal_point":      "The point the camera is looking at (x, y, z).",
        "size":             "The window size of figure",
        "bg_color":         "The background color of figure",
        "bg_opacity":       "The background opacity of figure",
    }

    _validators = {
        **(OptsBase._validators),
        "azimuth":          lambda v, d: as_Number(v, name=d, value_range=(0, 360)),
        "elevation":        lambda v, d: as_Number(v, name=d, value_range=(-90, 90)),
        "roll":             lambda v, d: as_Number(v, name=d, value_range=(-180, 180)),
        "distance":         lambda v, d: as_Number(v, name=d, value_range=(0, np.inf)),
        "focal_point":      lambda v, d: as_Vect(v, name=d, dim=3),
        "size":             lambda v, d: as_Vect(v, name=d, dim=2),
        "bg_color":         lambda v, d: as_ColorRGB(v, name=d),
        "bg_opacity":       lambda v, d: as_Number(v, name=d, value_range=(0, 1), bounded=True),
    }
    
    _DEFAULTS_FROZEN = MappingProxyType({
        **(OptsBase._DEFAULTS_FROZEN),
        "tag":              "figure options",
        "size":             (2542, 1305),
        "bg_color":         (1,1,1),
        "bg_opacity":       0
    })
            


class PlotFigure(HostBase, RegistryBase):
    
    _DEFAULT_NAME = "unamed figure"

    __descriptions__ = {
        **(HostBase.__descriptions__),
        
        "raw_name": "The name identifier of the figure",
        "_entity_plotter": "The underlying PyVista BackgroundPlotter instance that owns the VTK rendering pipeline. ",
        "_entity": "A registry for objects attached to this figure.",
        "_entity_overlay": (
            "A foreground VTK renderer (layer=1) sharing the main camera. "
            "Actors added to this renderer are drawn after the main scene and "
            "are not occluded by 3D geometry in the base layer."
        ),
        
        "_entity_pick_manager": "The PickManager instance attached to this figure. ",
        "_entity_console": "The ScopedConsoleDock instance attached to this figure. ",
        "_entity_scalar_bars": "The RegistryBase instance to manage scalar bars in this figure. "

    }
    

    __slots__ = tuple(__descriptions__.keys()) #+ ("__weakref__",)

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        plotter: pv.Plotter | None = None,
        opts: OptsFigure | None = None,
        name: str | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):
        
        logger.detail("Resovle the input plotter")
        if plotter is None:
            plotter = BackgroundPlotter()
        else:
            if not isinstance(plotter, BackgroundPlotter):
                try:
                    raise TypeError(
                        "`plotter` for PlotFigure must be PyVista BackgroundPlotter object, or None."
                    )
                except:
                    logger.exception("Check input")
                    logger.recovery("Create a new figure instead.")
                    plotter = BackgroundPlotter()
        
        object.__setattr__(self, "_entity_plotter", plotter)
        object.__setattr__(self, "_entity", [])
        
        super().__init__(
            OptsFigure,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=self._DEFAULT_NAME,
            **kwargs
            )
        self.opts.act_finalize(is_allow_UNSET=True)
        plotter.resize(*self.opts.size)

        self._helper_sync_from_plotter(is_allow_cover_target_set=False, is_only_camera=True)
        self._helper_commit_apply_opts()
        
        def _on_interaction_start(obj, event):
            pm = getattr(self, "_entity_pick_manager", None)
            if pm is not None:
                pm._helper_hide_marker_label_during_interaction()
            self.pl.render()
        
        def _on_interaction_end(obj, event):
            self._helper_sync_from_plotter()
            pm = getattr(self, "_entity_pick_manager", None)
            if pm is not None:
                pm._helper_show_marker_label_after_interaction()
            self.pl.render()
    
    
        self.pl.iren.add_observer("StartInteractionEvent", _on_interaction_start)
        self.pl.iren.add_observer("EndInteractionEvent", _on_interaction_end)
        
        # --- Create overlay renderer (layer=1) at initialization ---
        overlay = self._helper_init_overlay_renderer()
        object.__setattr__(self, "_entity_overlay", overlay)
        
        pm = PickManager(self)
        object.__setattr__(self, "_entity_pick_manager", pm)
        self.pl.enable_point_picking(
            callback = self.pick_manager._helper_callback,
            left_clicking=True,
            pickable_window=True,
            use_picker=True,
            show_point=False,
            picker="cell",
            tolerance=0.003,
            show_message=False,
        )
        
        main_window = self.pl.app_window
        console = ScopedConsoleDock(parent=main_window)
        main_window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, console)
        
        object.__setattr__(self, "_entity_console", console)
        
        scalar_bars = RegistryBase("scalar bars manager")
        scalar_bars._impl_registry_ref = weakref.ref(self)
        object.__setattr__(self, "_entity_scalar_bars", scalar_bars)
        
    
    @property
    def console(self):
        return self._entity_console
    
    def act_set_name(self, name):
        name = super().act_set_name(name)
        if name:
            self.pl.app_window.setWindowTitle(name)
        

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_apply_opts(self, **kwargs):
        
        with self.opts._helper_internal_update():
            cover_value(self.opts,
                        is_allow_cover_target_set=True,
                        is_allow_unset_source=False,
                        **kwargs
                        )
        
        self._helper_sync_from_opts()
        

    @property
    def pl(self):
        return self._entity_plotter
    
    @property
    def pick_manager(self):
        return self._entity_pick_manager
    

    def act_check_is_alive(self):
        try:
            if len(self.pl.renderer.actors) == 0:
                return True
            
            if self.pl._closed:
                return False
            
            return True if self.pl.render_window.GetGenericWindowId() else False

            # iren = plotter.iren
            # return iren is not None and bool(iren.initialized)
        except Exception:
            return False

    def __bool__(self):
        return self.act_check_is_alive()
    
    
    
    
    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_init_overlay_renderer(self, logger=None) -> vtk.vtkRenderer:
        """
        Initialize a foreground overlay renderer (layer=1) that shares the main camera.

        Notes
        -----
        - The overlay renderer is drawn after the main renderer, so its actors
          are not occluded by the 3D geometry from the base layer.
        - The overlay renderer is non-interactive and shares the main camera.
        """
        rw = self.pl.render_window

        # Ensure we have at least two layers.
        rw.SetNumberOfLayers(2)

        # Main renderer in layer 0.
        self.pl.renderer.SetLayer(0)

        overlay = vtk.vtkRenderer()
        overlay.SetLayer(1)
        overlay.SetInteractive(False)
        overlay.SetActiveCamera(self.pl.renderer.GetActiveCamera())

        rw.AddRenderer(overlay)

        return overlay
    
    
    
    
    # -------------------------------
    # Functions about camera settings
    # -------------------------------




    def _helper_sync_from_plotter(self, 
                                  is_allow_cover_target_set=True,
                                  is_only_camera=False):
        
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
        
        if not is_only_camera:
            size = self.pl.size()
            alter = {
                **alter,
                "bg_color":     self.pl.background_color.float_rgb,
                "bg_opacity":   self.pl.background_color.opacity / 255.0,
                "size":         (size.width(), size.height())
                }
        
        with self.opts._helper_internal_update():
            cover_value(self.opts, is_allow_cover_target_set=is_allow_cover_target_set, **alter)


    def _helper_sync_from_opts(self):
        
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

            rgba = np.r_[self.opts.bg_color, [self.opts.bg_opacity]] * 255
            rgba = rgba.astype(int)
            self.pl.background_color = rgba
            
            self.pl.resize(*self.opts.size)

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
        right = np.cross(view_dir, [0, 0, 1])
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(view_dir, [0, 1, 0])
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
        up_candidate = np.array([0, 0, 1])
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

            
    @logging_and_warning_decorator(start_finish_level=5)
    def act_register(self, term, is_contain_ok=False, logger=None):
    
        if term in self._entity:
            if not is_contain_ok:
                try:
                    raise ValueError(f"term {term!r} is already registered in Registry {self.name!r}")
                except ValueError:
                    logger.exception("Check input.")
                    logger.recovery("Ignore this process.")
            return
        
        if not hasattr(self, "name"):
            raise TypeError("term must have attribute `.name`.")
        name = self._helper_check_name(term.name)
        term.name = name
        self._entity.append(term)
        object.__setattr__(term, "_impl_figure_ref", weakref.ref(self))
        
        if term.opts.is_reset_camera:
            self._helper_sync_from_plotter()
            
    
    def __repr__(self):
        msg = HostBase.__repr__(self) + "\n"
        msg += RegistryBase._helper_repr_by_category(self)
        return msg



