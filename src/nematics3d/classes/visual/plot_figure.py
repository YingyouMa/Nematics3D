"""Figure host and camera helpers for PyVista-based Nematics3D scenes."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import BackgroundPlotter
from qtpy import QtCore

from nematics3d.datatypes import (
    ColorRGB,
    UNSET,
    Unset,
    Vect,
    as_ColorRGB,
    as_bool,
    as_Number,
    as_Vect,
    as_str,
)
from nematics3d.logging_decorator import logging_and_warning_decorator

from ..host_base import HostBase, OptsBase
from ..opts import cover_value
from ..registry_base import RegistryBase
from .pick_manager import PickManager
from .qt.console import ScopedConsoleDock


@dataclass(slots=True, repr=False)
class OptsFigure(OptsBase):
    """
    Options object controlling a PlotFigure camera and window state.

    This is the user-facing options container paired with PlotFigure. Most figure
    configuration is read through `figure.opts` and updated through
    `figure.opts.<name> = value`, `figure.act_commit(...)`, or the inherited
    OptsBase helper methods.

    Important readable attributes:

    - `host`: the PlotFigure currently using this opts object, if any.
    - `azimuth`, `elevation`, `roll`, `distance`, `focal_point`: the camera pose
      stored in spherical form around the focal point.
    - `size`: the figure window size.
    - `bg_color`: the figure background color.

    Common user actions:

    - `act_finalize()`: validate defaults and lock the opts into functioning use.
    - `act_asdict()`: export the current opts values as a plain dictionary.
    - `act_save_json()`: save the current opts to JSON, with large arrays stored
      as sidecar `.npy` files when needed.
    - `act_load_json()`: load a JSON snapshot into this existing opts object.

    Representation:

    - `str(opts)` returns a short one-line identity.
    - `repr(opts)` returns the full current opts summary.
    """

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
        "elevation":        (
            "The elevation angle (degrees) of the camera relative to the focal "
            "plane."
        ),
        "roll":             (
            "The rotation (degrees) of the camera about the direction of "
            "projection."
        ),
        "distance":         "The distance from the camera position to the focal point.",
        "focal_point":      "The point the camera is looking at (x, y, z).",
        "size":             "The window size of figure",
        "bg_color":         "The background color of figure",
    }

    impl_validators = {
        **(OptsBase.impl_validators),
        "azimuth":          lambda v, d: as_Number(v, name=d, value_range=(0, 360)),
        "elevation":        lambda v, d: as_Number(v, name=d, value_range=(-90, 90)),
        "roll":             lambda v, d: as_Number(v, name=d, value_range=(-180, 180)),
        "distance":         lambda v, d: as_Number(v, name=d, value_range=(0, np.inf)),
        "focal_point":      lambda v, d: as_Vect(v, name=d, dim=3),
        "size":             lambda v, d: as_Vect(v, name=d, dim=2),
        "bg_color":         lambda v, d: as_ColorRGB(v, name=d),
    }

    impl_defaults_frozen = MappingProxyType(
        {
            **(OptsBase.impl_defaults_frozen),
            "tag":          "figure options",
            "size":         (1900, 1000),
            "bg_color":     (1, 1, 1),
        }
    )
    # fmt: on


# Subclassing rules:
# - PlotFigure is a HostBase scene container that owns several attached
#   RegistryBase helpers for glyphs, scalar bars, and interact panels.
# - Keep `entity_plotter` as the single source of truth for the backend plotter
#   object, and keep `entity_glyphs`, `entity_scalar_bars`, and other attached
#   entities synchronized with that plotter state.
# - If a subclass changes figure initialization, preserve the distinction
#   between creating a new plotter and wrapping an existing one.
# - If a subclass overrides camera/application sync behavior, keep opts, camera,
#   and plotter state mutually consistent after every accepted update.
# - Keep `__repr__` as the detailed host summary plus attached-registry view and
#   `__str__` as the short host-style identity view unless there is a strong
#   reason to change both.


class PlotFigure(HostBase):
    """
    Figure object for interactive or off-screen 3D visualization.

    PlotFigure is the main scene container used by the visual classes in this
    repository. It follows the HostBase opts/commit model and owns attached
    RegistryBase helpers that manage the plotted objects linked to the scene.

    Important readable attributes:

    - `opts`: the paired OptsFigure controlling camera and figure settings.
    - `pl`: the underlying PyVista plotter or BackgroundPlotter backend.
    - `pl_type`: short plotter identifier, `'B'` for BackgroundPlotter and `'P'`
      for plain PyVista Plotter.
    - `is_alive`: whether the wrapped plotter window/backend is still valid.

    Common inspection helpers:

    - `show_readable_attrs()`: show the main readable figure attributes.
    - `show_modifiable_attrs()`: show which figure or opts attributes can be
      changed through the public interface.
    - `show_attr_desc(name)`: describe a specific readable attribute.
    - `show_relations()`: show object relations inherited from HostBase/ClassBase.

    Common user actions:

    - `act_commit(...)`: update figure settings through the host/opts pipeline.
    - `act_set_name(name)`: rename the figure and sync the interactive window
      title when applicable.
    - `act_reset_camera()`: reset the camera and sync the resulting pose back to
      `opts`.
    - `act_view_xy()`, `act_view_xz()`, `act_view_yz()`, `act_view_isometric()`:
      apply a standard camera view and sync it back to `opts`.
    - `act_register(term)` / `act_unregister(term)`: manage plotted objects in
      the figure registry.
    - `act_clear_category(category, ...)`: remove all registered objects in one
      category.
    - `act_savefig(...)`: save a screenshot of the current figure.

    Representation:

    - `str(figure)` returns the short host-style identity.
    - `repr(figure)` returns the detailed host summary together with the attached
      registry contents.
    """

    _DEFAULT_NAME = "unnamed figure"

    __attr_defs__ = {
        **(HostBase.__attr_defs__),
        # -----------------
        # Public identity
        # -----------------
        "raw_name": {
            "doc": (
                "Human-readable identifier of the figure. "
                "Used as the window title for BackgroundPlotter."
            ),
        },
        # -----------------
        # Core plot backend
        # -----------------
        "entity_plotter": {
            "doc": (
                "The underlying plotting backend. "
                "Either a pyvista.Plotter or a pyvistaqt.BackgroundPlotter instance."
            ),
        },
        # -----------------
        # Attached entities
        # -----------------
        "entity_pick_manager": {
            "doc": (
                "The PickManager instance associated with this figure. "
                "Available only in interactive (on-screen) sessions."
            ),
        },
        "entity_console": {
            "doc": (
                "The ScopedConsoleDock attached to the Qt main window. "
                "Available only in interactive (on-screen) sessions."
            ),
        },
        "entity_scalar_bars": {
            "doc": "RegistryBase instance managing scalar bars attached to this figure.",
        },
        "entity_interacts": {
            "doc": (
                "RegistryBase instance managing live interact panels attached to this figure."
            ),
        },
        "entity_glyphs": {
            "doc": (
                "RegistryBase instance managing glyph visual objects attached to this figure."
            ),
        },
        "impl_interact_count": {
            "doc": "Monotonic counter used to assign interact panel ids for this figure.",
        },
        # -----------------
        # VTK overlay layer
        # -----------------
        "entity_overlay": {
            "doc": (
                "Foreground vtkRenderer (layer=1) that shares the main camera "
                "with the base renderer. Actors added to this renderer are drawn "
                "on top of the main scene and are not occluded by 3D geometry."
            ),
        },
        "pl": {
            "doc": "Read-only: Alias of `entity_plotter`.",
            "kind": "property",
        },
        "pl_type": {
            "doc": (
                "Read-only: Short identifier of the plotter type. "
                "'B' for BackgroundPlotter, 'P' for pyvista.Plotter."
            ),
            "kind": "property",
        },
        "pick_manager": {
            "doc": (
                "Read-only: Alias of `entity_pick_manager` "
                "(or None if not initialized)."
            ),
            "kind": "property",
        },
        "console": {
            "doc": "Read-only: Alias of `entity_console` (or None if not initialized).",
            "kind": "property",
        },
        "interacts": {
            "doc": (
                "Read-only: Alias of `entity_interacts` (or None if not initialized)."
            ),
            "kind": "property",
        },
        "scalar_bars": {
            "doc": "Read-only: Alias of `entity_scalar_bars`.",
            "kind": "property",
        },
        "glyphs": {
            "doc": "Read-only: Alias of `entity_glyphs`.",
            "kind": "property",
        },
        "overlay": {
            "doc": "Read-only: Alias of `entity_overlay`.",
            "kind": "property",
        },
        "is_alive": {
            "doc": "Read-only: Whether the wrapped plotter/window backend is still alive.",
            "kind": "property",
        },
    }

    __slots__ = (
        "entity_plotter",
        "entity_pick_manager",
        "entity_console",
        "entity_scalar_bars",
        "entity_interacts",
        "entity_glyphs",
        "impl_interact_count",
        "entity_overlay",
    )

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

        object.__setattr__(self, "entity_plotter", plotter)
        object.__setattr__(self, "impl_interact_count", 0)

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
        self.opts.act_finalize(is_allow_unset=True)
        plotter.window_size = tuple(int(x) for x in self.opts.size)

        self._helper_sync_from_plotter(
            is_allow_cover_target_set=False, is_only_camera=True
        )
        self._helper_commit_apply_opts(is_reapply_opts=True)

        glyphs = RegistryBase("figure glyph registry")
        glyphs.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "entity_glyphs", glyphs)

        scalar_bars = RegistryBase("scalar bars manager")
        scalar_bars.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "entity_scalar_bars", scalar_bars)

        interacts = RegistryBase("interact panel manager")
        interacts.act_bind_relation_base("owner", self, is_weak=True)
        object.__setattr__(self, "entity_interacts", interacts)

        # --- Create overlay renderer (layer=1) at initialization ---
        overlay = self._helper_init_overlay_renderer()
        object.__setattr__(self, "entity_overlay", overlay)

        if not is_off_screen:

            def _on_interaction_start(_obj, _event):
                pm = getattr(self, "entity_pick_manager", None)
                if pm is not None:
                    pm._helper_hide_marker_label_during_interaction()
                self.pl.render()

            def _on_interaction_end(_obj, _event):
                self._helper_sync_from_plotter()
                pm = getattr(self, "entity_pick_manager", None)
                if pm is not None:
                    pm._helper_show_marker_label_after_interaction()
                self.pl.render()

            self.pl.iren.add_observer("StartInteractionEvent", _on_interaction_start)
            self.pl.iren.add_observer("EndInteractionEvent", _on_interaction_end)

            pm = PickManager(self)
            object.__setattr__(self, "entity_pick_manager", pm)
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

            object.__setattr__(self, "entity_console", console)

            original_close_event = main_window.closeEvent

            def _close_event_with_interacts(event):
                self._helper_close_interacts()
                original_close_event(event)

            main_window.closeEvent = _close_event_with_interacts

    # -------------------------------
    # Interactive figure services
    # -------------------------------

    @property
    def console(self):
        """Return the attached scoped console dock when available."""
        return getattr(self, "entity_console", None)

    @property
    def interacts(self):
        """Return the registry managing interact panels for this figure."""
        return getattr(self, "entity_interacts", None)

    @property
    def scalar_bars(self):
        """Return the scalar-bar registry managed by this figure."""
        return self.entity_scalar_bars

    @property
    def glyphs(self):
        """Return the registry managing glyph objects attached to this figure."""
        return self.entity_glyphs

    @property
    def overlay(self):
        """Return the overlay renderer used for always-on-top VTK actors."""
        return self.entity_overlay

    def act_register_interact(self, panel):
        """Register one interact panel and assign it a figure-local panel name."""
        interacts = self.interacts
        if interacts is None:
            return None
        count = self.impl_interact_count + 1
        object.__setattr__(self, "impl_interact_count", count)
        panel.name = f"panel{count}"
        interacts.act_register(
            panel,
            is_contain_ok=True,
            is_bind_registry_relation=False,
        )
        return panel.name

    def act_unregister_interact(self, panel):
        """Unregister one interact panel from this figure."""
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
            except (AttributeError, RuntimeError, ReferenceError):
                pass

    def act_close(self, *, is_remove_glyphs: bool = True):
        """Close interact panels, optional glyphs, and the plotter backend."""
        is_remove_glyphs = as_bool(
            is_remove_glyphs,
            name="Whether to remove registered glyphs before closing the figure",
        )

        self._helper_close_interacts()

        if is_remove_glyphs:
            for glyph in list(self.glyphs):
                remove = getattr(glyph, "act_remove", None)
                try:
                    if callable(remove):
                        remove()
                    else:
                        self.act_unregister(glyph, is_missing_ok=True)
                except (AttributeError, RuntimeError, ReferenceError):
                    self.act_unregister(glyph, is_missing_ok=True)

        try:
            self.pl.close()
        except (AttributeError, RuntimeError, ReferenceError):
            pass

    # -------------------------------
    # Naming and commit hooks
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotFigure overrides HostBase/ClassBase name handling so BackgroundPlotter
    # window titles stay synchronized with the figure name.
    # ==================================================

    def act_set_name(self, name):
        """Set the figure name and sync the interactive window title when needed."""
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

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):

        if not is_reapply_opts and not kwargs:
            return

        with self.opts.act_internal_update():
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )

        self._helper_sync_from_opts()

    # -------------------------------
    # Plotter state and liveness
    # -------------------------------

    @property
    def pl(self):
        """Return the underlying PyVista plotter backend."""
        return self.entity_plotter

    @property
    def pick_manager(self):
        """Return the attached PickManager when interactive picking is enabled."""
        return getattr(self, "entity_pick_manager", None)

    @property
    def pl_type(self):
        """Return a short backend code: B for BackgroundPlotter, P for Plotter."""
        if isinstance(self.pl, BackgroundPlotter):
            return "B"
        if isinstance(self.pl, pv.Plotter):
            return "P"
        raise TypeError(
            f"Unsupported plotter type: {type(self.pl).__name__}. "
            "Expected pyvista.Plotter or pyvistaqt.BackgroundPlotter."
        )

    def act_check_is_alive(self):
        """Return whether the wrapped plotter backend is still usable."""
        try:
            if self.pl._closed:
                return False
            if self.pl_type == "P":
                return True

            return bool(self.pl.render_window.GetGenericWindowId())

        except (AttributeError, RuntimeError, ReferenceError):
            return False

    @property
    def is_alive(self):
        """Return whether the wrapped plotter/window backend is still alive."""
        return self.act_check_is_alive()

    def __bool__(self):
        raise RuntimeError(
            "Boolean evaluation of this object is deprecated and no longer supported. "
            "This method is retained only to detect legacy usage during debugging. "
            "Please explicitly use the `is_alive` property instead."
        )

    # -------------------------------
    # Plotter backend helpers
    # -------------------------------

    def _helper_init_overlay_renderer(self) -> vtk.vtkRenderer:
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
    # Camera synchronization
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

        with self.opts.act_internal_update():
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

    # -------------------------------
    # Camera view actions
    # -------------------------------

    def act_reset_camera(self):
        """Reset the plotter camera and sync the resulting pose back to opts."""
        self.pl.reset_camera()
        self._helper_sync_from_plotter()

    def act_view_xy(self):
        """Apply the XY view and sync the resulting camera pose back to opts."""
        self.pl.view_xy()
        self._helper_sync_from_plotter()

    def act_view_xz(self):
        """Apply the XZ view and sync the resulting camera pose back to opts."""
        self.pl.view_xz()
        self._helper_sync_from_plotter()

    def act_view_yz(self):
        """Apply the YZ view and sync the resulting camera pose back to opts."""
        self.pl.view_yz()
        self._helper_sync_from_plotter()

    def act_view_isometric(self):
        """Apply the isometric view and sync the resulting camera pose back to opts."""
        self.pl.view_isometric()
        self._helper_sync_from_plotter()

    # -------------------------------
    # Registry integration and export
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotFigure exposes a figure-level register action that forwards to the
    # owned glyph registry, then refreshes camera state when the new object
    # requests a camera reset.
    # ==================================================

    def act_register(self, term, is_contain_ok=False, is_bind_registry_relation=True):
        """Register a drawable term and refresh camera state if it resets camera."""
        self.glyphs.act_register(
            term,
            is_contain_ok=is_contain_ok,
            is_bind_registry_relation=is_bind_registry_relation,
        )
        if term.opts.is_reset_camera:
            self._helper_sync_from_plotter()

    def act_clear_category(self, category, is_missing_ok=True):
        """Remove all registered glyph terms whose category matches the given name."""
        category = as_str(category, name="The category to clear from figure")
        terms = [
            term for term in list(self) if getattr(term, "category", None) == category
        ]
        if not terms:
            if is_missing_ok:
                return []
            raise KeyError(f"Category {category!r} does not exist in this figure.")

        removed = []
        for term in terms:
            if callable(getattr(term, "act_remove", None)):
                term.act_remove()
            else:
                self.act_unregister(term, is_missing_ok=True)
            removed.append(term)
        return removed

    def act_unregister(self, term, is_missing_ok=False):
        """Unregister a drawable term from this figure glyph registry."""
        self.glyphs.act_unregister(term, is_missing_ok=is_missing_ok)

    def __call__(self):
        """Return the registered glyph objects as a tuple."""
        return self.glyphs()

    def __len__(self) -> int:
        """Return the number of glyph objects registered in this figure."""
        return len(self.glyphs)

    def __iter__(self):
        """Iterate over glyph objects registered in this figure."""
        return iter(self.glyphs)

    def __contains__(self, item):
        """Return whether one glyph object is registered in this figure."""
        return item in self.glyphs

    def __getitem__(self, key):
        """Lookup a registered glyph object by index or name."""
        return self.glyphs[key]

    def act_savefig(
        self, filename, scale=1, is_transparent_background=False, window_size=None
    ):
        """Save a screenshot of the current figure to an image file."""
        if window_size is None:
            window_size = self.opts.size
        window_size = tuple(int(x) for x in window_size)
        self.pl.screenshot(
            filename,
            scale=scale,
            transparent_background=is_transparent_background,
            window_size=window_size,
        )

    # -------------------------------
    # Representation
    # -------------------------------

    # ==================== OVERRIDE ====================
    # PlotFigure overrides HostBase.__repr__ to combine the host summary with
    # the registry-style listing of objects attached to the figure.
    # ==================================================

    def __repr__(self):
        """Return the detailed host summary together with the glyph registry view."""
        msg = HostBase.__repr__(self) + "\n"
        msg += repr(self.glyphs)
        return msg

    # ==================== OVERRIDE ====================
    # PlotFigure overrides the default string form to keep the short host-style
    # identity view for compact logging and display.
    # ==================================================

    def __str__(self):
        """Return the short host-style identity string for this figure."""
        return f"{type(self).__name__}({self.name!r})"


FigureData = PlotFigure | BackgroundPlotter | pv.Plotter


@logging_and_warning_decorator()
def as_plotfigure(figure, opts_figure=None, logger=None):
    """Normalize figure-like input into a PlotFigure instance."""

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
                f"Got type {type(figure)!r} instead."
            )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        logger.exception("Invalid figure input")
        logger.recovery("Create a new figure instead.")
        figure = PlotFigure(opts=opts_figure)

    return figure


# Backward-compatible alias; prefer `as_plotfigure`.
as_PlotFigure = as_plotfigure
