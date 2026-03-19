from dataclasses import dataclass, field
import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import BackgroundPlotter
from types import MappingProxyType
from typing import Mapping, Any
from PyQt5 import QtCore
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
from ..opts import cover_value
from ..registry_base import RegistryBase
from .pick_manager import PickManager
from .qt.console import ScopedConsoleDock


#!!! property act_view
#!!! load save


@dataclass(slots=True, repr=False)
class OptsFigure(OptsBase):
    azimuth: float | Unset = UNSET
    elevation: float | Unset = UNSET
    roll: float | Unset = UNSET
    distance: float | Unset = UNSET
    focal_point: Vect(3) | Unset = UNSET
    size: Vect(2) | Unset = UNSET
    bg_color: ColorRGB | Unset = UNSET

    # fmt: off
    __attrs__ = {
        **(OptsBase.__attrs__),
        "azimuth":          "The azimuthal angle (degrees) of the camera around the focal point.",
        "elevation":        "The elevation angle (degrees) of the camera relative to the focal plane.",
        "roll":             "The rotation (degrees) of the camera about the direction of projection.",
        "distance":         "The distance from the camera position to the focal point.",
        "focal_point":      "The point the camera is looking at (x, y, z).",
        "size":             "The window size of figure",
        "bg_color":         "The background color of figure",
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
    }

    _DEFAULTS_FROZEN = MappingProxyType(
        {
            **(OptsBase._DEFAULTS_FROZEN),
            "tag":          "figure options",
            "size":         (1900, 1000),
            "bg_color":     (1, 1, 1),
        }
    )
    # fmt: on


# Subclassing rules:
# - PlotFigure combines HostBase and RegistryBase. Subclasses must preserve both
#   contracts: host-style commit/update behavior and registry-style management
#   of attached visual entities.
# - Keep `_entity_plotter` as the single source of truth for the backend plotter
#   object, and keep `_entity`, `_entity_scalar_bars`, and other attached
#   entities synchronized with that plotter state.
# - If a subclass changes figure initialization, preserve the distinction
#   between creating a new plotter and wrapping an existing one.
# - If a subclass overrides camera/application sync behavior, keep opts, camera,
#   and plotter state mutually consistent after every accepted update.
# - Keep `__repr__` as the detailed combined host/registry view and `__str__`
#   as the short host-style identity view unless there is a strong reason to
#   change both.


class PlotFigure(HostBase, RegistryBase):
    """
    Figure object for interactive or off-screen 3D visualization.

    For most users, PlotFigure is the main visualization container used by
    plotting and visualization methods across the package.

    Typical usage:

    - create a new figure with `PlotFigure()` or `PlotFigure(is_off_screen=True)`
    - pass the figure into other visualization helpers so multiple objects share
      the same scene
    - read the current figure settings through `figure.opts`
    - modify figure settings through `figure.opts.<name> = value` or
      `figure.act_commit(...)`
    - inspect available figure settings through `figure.opts`
    - access attached plotted objects through the registry behavior inherited by
      the figure
    - use `str(figure)` for a short identity view and `repr(figure)` for a more
      detailed summary of attached objects

    PlotFigure manages both the rendering backend and the registry of objects
    attached to that backend.
    """

    _DEFAULT_NAME = "unnamed figure"

    __attrs__ = {
        **(HostBase.__attrs__),
        # -----------------
        # Public identity
        # -----------------
        "raw_name": (
            "Human-readable identifier of the figure. "
            "Used as the window title for BackgroundPlotter."
        ),
        # -----------------
        # Core plot backend
        # -----------------
        "_entity_plotter": (
            "The underlying plotting backend. "
            "Either a pyvista.Plotter or a pyvistaqt.BackgroundPlotter instance. "
        ),
        # -----------------
        # Attached entities
        # -----------------
        "_entity": ("Internal registry of objects attached to this figure."),
        "_entity_pick_manager": (
            "The PickManager instance associated with this figure. "
            "Available only in interactive (on-screen) sessions."
        ),
        "_entity_console": (
            "The ScopedConsoleDock attached to the Qt main window. "
            "Available only in interactive (on-screen) sessions."
        ),
        "_entity_scalar_bars": (
            "RegistryBase instance managing scalar bars attached to this figure."
        ),
        "_entity_interacts": (
            "RegistryBase instance managing live interact panels attached to this figure."
        ),
        "_impl_interact_count": (
            "Monotonic counter used to assign interact panel ids for this figure."
        ),
        # -----------------
        # VTK overlay layer
        # -----------------
        "_entity_overlay": (
            "Foreground vtkRenderer (layer=1) that shares the main camera "
            "with the base renderer. "
            "Actors added to this renderer are drawn on top of the main scene "
            "and are not occluded by 3D geometry."
        ),
    }
    __properties__ = {
        **(HostBase.__properties__),
        "pl": "Read-only: Alias of `_entity_plotter`.",
        "pl_type": (
            "Read-only: Short identifier of the plotter type. "
            "'B' for BackgroundPlotter, 'P' for pyvista.Plotter."
        ),
        "pick_manager": (
            "Read-only: Alias of `_entity_pick_manager` "
            "(or None if not initialized)."
        ),
        "console": (
            "Read-only: Alias of `_entity_console` " "(or None if not initialized)."
        ),
        "interacts": (
            "Read-only: Alias of `_entity_interacts` (or None if not initialized)."
        ),
    }

    __slots__ = tuple(k for k in __attrs__.keys() if k not in HostBase.__slots__)

    # ==================== OVERRIDE ====================
    # PlotFigure overrides HostBase.__init__ because it must construct or wrap
    # a plotting backend before binding figure opts and attached figure entities.
    # ==================================================

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        plotter: pv.Plotter | BackgroundPlotter | None = None,
        opts: OptsFigure | None = None,
        is_off_screen: bool = False,
        name: str | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):

        is_new_plotter = False
        if plotter is None:
            is_new_plotter = True
        else:
            if not isinstance(plotter, (BackgroundPlotter, pv.Plotter)):
                try:
                    raise TypeError(
                        "`plotter` for PlotFigure must be either"
                        "pyvistaqt BackgroundPlotter object or PyVista Plotter object, or None."
                    )
                except TypeError:
                    logger.exception("Check input")
                    logger.recovery("Create a new figure instead.")
                is_new_plotter = True
            else:
                if is_off_screen and not plotter.off_screen:
                    logger.warning(
                        "The provided plotter is not in off-screen mode, "
                        "but is_off_screen=True was requested. "
                        "The existing plotter's display mode will take precedence."
                    )
                    is_off_screen = False

        if is_new_plotter:
            if is_off_screen:
                plotter = pv.Plotter(off_screen=True)
            else:
                plotter = BackgroundPlotter()

        object.__setattr__(self, "_entity_plotter", plotter)
        object.__setattr__(self, "_entity", [])
        object.__setattr__(self, "_impl_interact_count", 0)

        if name is None:
            name = self._DEFAULT_NAME

        super().__init__(
            OptsFigure,
            opts,
            opts_defaults_override,
            name=name,
            name_replace=self._DEFAULT_NAME,
            **kwargs,
        )
        self.opts.act_finalize(is_allow_UNSET=True)
        plotter.window_size = tuple(int(x) for x in self.opts.size)

        self._helper_sync_from_plotter(
            is_allow_cover_target_set=False, is_only_camera=True
        )
        self._helper_commit_apply_opts(is_reapply_opts=True)

        scalar_bars = RegistryBase("scalar bars manager")
        scalar_bars.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "_entity_scalar_bars", scalar_bars)

        interacts = RegistryBase("interact panel manager")
        interacts.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "_entity_interacts", interacts)

        # --- Create overlay renderer (layer=1) at initialization ---
        overlay = self._helper_init_overlay_renderer()
        object.__setattr__(self, "_entity_overlay", overlay)

        if not is_off_screen:

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

            pm = PickManager(self)
            object.__setattr__(self, "_entity_pick_manager", pm)
            self.pl.enable_point_picking(
                callback=self.pick_manager._helper_callback,
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

            original_close_event = main_window.closeEvent

            def _close_event_with_interacts(event):
                self._helper_close_interacts()
                original_close_event(event)

            main_window.closeEvent = _close_event_with_interacts

    @property
    def console(self):
        return getattr(self, "_entity_console", None)

    @property
    def interacts(self):
        return getattr(self, "_entity_interacts", None)

    def act_register_interact(self, panel):
        interacts = self.interacts
        if interacts is None:
            return None
        count = self._impl_interact_count + 1
        object.__setattr__(self, "_impl_interact_count", count)
        panel.name = f"panel{count}"
        interacts.act_register(
            panel,
            is_contain_ok=True,
            is_bind_registry_relation=False,
        )
        return panel.name

    def act_unregister_interact(self, panel):
        interacts = self.interacts
        if interacts is None:
            return
        interacts.act_unregister(panel, is_missing_ok=True)

    def _helper_close_interacts(self):
        interacts = self.interacts
        if interacts is None:
            return
        for panel in list(interacts):
            try:
                panel.close()
            except Exception:
                pass

    # ==================== OVERRIDE ====================
    # PlotFigure overrides HostBase/ClassBase name handling so BackgroundPlotter
    # window titles stay synchronized with the figure name.
    # ==================================================

    def act_set_name(self, name):
        if name is None:
            name = self._DEFAULT_NAME
        name = super().act_set_name(name)
        if name and self.pl_type == "B":
            self.pl.app_window.setWindowTitle(name)
        return name

    # ==================== OVERRIDE ====================
    # PlotFigure overrides HostBase._helper_commit_apply_opts_main to treat
    # figure opts as direct camera/window state updates on the live plotter.
    # ==================================================

    @logging_and_warning_decorator(start_finish_level=5)
    def _helper_commit_apply_opts_main(
        self, is_reapply_opts=False, logger=None, **kwargs
    ):

        if not is_reapply_opts and not kwargs:
            return

        with self.opts._helper_internal_update():
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )

        self._helper_sync_from_opts()

    @property
    def pl(self):
        return self._entity_plotter

    @property
    def pick_manager(self):
        return getattr(self, "_entity_pick_manager", None)

    @property
    def pl_type(self):
        if isinstance(self.pl, BackgroundPlotter):
            return "B"
        elif isinstance(self.pl, pv.Plotter):
            return "P"
        else:
            raise TypeError(
                f"Unsupported plotter type: {type(self.pl).__name__}. "
                "Expected pyvista.Plotter or pyvistaqt.BackgroundPlotter."
            )

    def act_check_is_alive(self):
        try:
            if self.pl._closed:
                return False
            else:
                if self.pl_type == "P":
                    return True

            return True if self.pl.render_window.GetGenericWindowId() else False

        except Exception:
            return False

    @property
    def is_alive(self):
        return self.act_check_is_alive()

    def __bool__(self):
        raise RuntimeError(
            "Boolean evaluation of this object is deprecated and no longer supported. "
            "This method is retained only to detect legacy usage during debugging. "
            "Please explicitly use the `is_alive` property instead."
        )

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

    def _helper_sync_from_plotter(
        self, is_allow_cover_target_set=True, is_only_camera=False
    ):

        camera = self.pl.camera
        temp = self._helper_convert_pos_to_spherical(
            camera.position, camera.focal_point, camera.up
        )

        alter = {
            "focal_point": camera.focal_point,
            "azimuth": temp[0],
            "elevation": temp[1],
            "roll": temp[2],
            "distance": temp[3],
        }

        if not is_only_camera:
            alter = {
                **alter,
                "bg_color": self.pl.background_color.float_rgb,
                "size": self.pl.window_size,
            }

        with self.opts._helper_internal_update():
            cover_value(
                self.opts, is_allow_cover_target_set=is_allow_cover_target_set, **alter
            )

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

        self.pl.set_background(self.opts.bg_color)
        self.pl.window_size = tuple(int(x) for x in self.opts.size)

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

    # ==================== OVERRIDE ====================
    # PlotFigure overrides RegistryBase.act_register so camera state can be
    # refreshed after registering objects that request a camera reset.
    # ==================================================

    def act_register(self, term, is_contain_ok=False):
        super().act_register(term, is_contain_ok=is_contain_ok)
        if term.opts.is_reset_camera:
            self._helper_sync_from_plotter()

    def act_savefig(
        self, filename, scale=1, is_transparent_background=False, window_size=None
    ):
        if window_size is None:
            window_size = self.opts.size
        self.pl.screenshot(
            filename,
            scale=scale,
            transparent_background=is_transparent_background,
            window_size=window_size,
        )

    # ==================== OVERRIDE ====================
    # PlotFigure overrides HostBase.__repr__ to combine the host summary with
    # the registry-style listing of objects attached to the figure.
    # ==================================================

    def __repr__(self):
        msg = HostBase.__repr__(self) + "\n"
        msg += RegistryBase._helper_repr_by_category(self)
        return msg

    # ==================== OVERRIDE ====================
    # PlotFigure overrides the default string form to keep the short host-style
    # identity view for compact logging and display.
    # ==================================================

    def __str__(self):
        return HostBase.__repr__(self)


FigureData = PlotFigure | BackgroundPlotter | pv.Plotter


@logging_and_warning_decorator()
def as_PlotFigure(figure, opts_figure=None, logger=None):

    if opts_figure is not None and not isinstance(opts_figure, OptsFigure):
        try:
            raise TypeError(
                "The variable `opts_figure` must be instance of OptsFigure."
                f"Got {type(opts_figure).__name__!r} instead."
            )
        except TypeError:
            logger.exception("Check input.")
            logger.recovery("Ignore this options in the following.")
        opts_figure = None

    try:
        if figure is None:
            figure = PlotFigure(opts=opts_figure)
        elif isinstance(figure, PlotFigure):
            figure.act_commit(opts_figure)
        elif isinstance(figure, (BackgroundPlotter, pv.Plotter)):
            figure = PlotFigure(plotter=figure, opts=opts_figure)
        else:
            raise ValueError(
                "`figure` input must be a valid PlotFigure object, "
                "or a valid pyvista plotter object (including BackgroundPlotter) "
                "or None (creating a new figure) "
                "Got type {type(figure)!r} instead."
            )
    except:
        logger.exception("Invalid figure input")
        logger.recovery("Create a new figure instead.")
        figure = PlotFigure(opts=opts_figure)

    return figure
