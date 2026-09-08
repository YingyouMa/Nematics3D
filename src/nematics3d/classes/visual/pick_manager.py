import weakref
import time
import vtk
import numpy as np
from qtpy import QtCore
from dataclasses import dataclass, field

from nematics3d.datatypes import (
    as_number,
    ColorRGB,
    as_ColorRGB,
)
from ...core.opts import merge_opts_all
from nematics3d.visual.qt.figure_options_dialog import FigureOptionsDialog
from nematics3d.visual.qt.pick_settings_dialog import PickSettingsDialog


@dataclass(slots=True)
class OptsPickManager:
    double_click_threshold: float = 0.3
    marker_proximity_threshold: float = 0.5
    marker_size: int = 14
    marker_color: ColorRGB = (1, 1, 0)
    marker_font_size: int = 14
    sil_color: ColorRGB = (0, 0, 0)
    sil_opacity: float = 0.8
    sil_width: float = 3
    slider_throttle_ms: int = 20
    _impl_owner_ref: weakref.ReferenceType | None = field(
        default=None, init=False, repr=False
    )

    __descriptions__ = {
        "double_click_threshold": (
            "The maximum time interval (in seconds)"
            " between two consecutive clicks to be registered as a double-click."
        ),
        "marker_proximity_threshold": (
            "The minimum distance (in meters) required between two markers"
            " to distinguish them as separate locations."
        ),
        "marker_size": "Screen-space size (in pixels) of the marker point.",
        "marker_color": "RGB color of the marker point",
        "marker_font_size": "Font size (in pixels) of the numeric label on top of the marker.",
        "sil_color": "RGB color of silhouette.",
        "sil_opacity": "Opacity of silhouette.",
        "sil_width": "Line width of silhouette.",
        "slider_throttle_ms": "Throttle interval (ms) for panel sliders in this figure window.",
    }

    _validators = {
        "double_click_threshold": lambda v, d: as_number(
            v, name=d, value_range=(0, np.inf), replace=0.3
        ),
        "marker_proximity_threshold": lambda v, d: as_number(
            v, name=d, value_range=(0, np.inf), replace=0.5
        ),
        "marker_size": lambda v, d: as_number(
            v,
            name=d,
            is_integer=True,
            value_range=(1, np.inf),
            replace=14,
        ),
        "marker_color": lambda v, d: as_ColorRGB(v, name=d, replace=(1, 1, 0)),
        "marker_font_size": lambda v, d: as_number(
            v,
            name=d,
            is_integer=True,
            value_range=(1, np.inf),
            replace=14,
        ),
        "sil_color": lambda v, d: as_ColorRGB(v, name=d, replace=(0, 0, 0)),
        "sil_opacity": lambda v, d: as_number(v, name=d, value_range=(0, 1)),
        "sil_width": lambda v, d: as_number(v, name=d, value_range=(0, np.inf)),
        "slider_throttle_ms": lambda v, d: as_number(
            v,
            name=d,
            is_integer=True,
            value_range=(1, 1000),
        ),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            desc = f"{key!r}: {self.__descriptions__.get(key)}"
            value = self._validators[key](value, desc)
        object.__setattr__(self, key, value)

        owner = getattr(self, "_impl_owner_ref", None)
        if owner:
            owner = owner()
            markers = list(owner._helper_iter_all_marker_packs())

            if key == "marker_size":
                for pack in markers:
                    pack.actor.GetProperty().SetPointSize(value)

            elif key == "marker_color":
                for pack in owner._entity_markers:
                    pack.actor.GetProperty().SetColor(*value)

            elif key == "marker_font_size":
                for pack in markers:
                    pack.text_actor.GetTextProperty().SetFontSize(value)

            elif key == "slider_throttle_ms":
                owner._helper_apply_panel_throttle(int(value))
                return

            else:
                for glyph in owner._impl_registry.values():
                    silhouette = getattr(glyph, "entity_silhouette", None)
                    if silhouette is not None and silhouette.visibility:
                        if key == "sil_color":
                            silhouette.prop.color = value
                        if key == "sil_opacity":
                            silhouette.prop.opacity = value
                        if key == "sil_width":
                            silhouette.prop.line_width = value

            owner.owner.pl.render()


@dataclass(slots=True)
class _ClickTracker:
    """Recognize same-actor double clicks for one mouse button."""

    last_time: float | None = None
    last_actor: object | None = None

    def consume(self, actor, now: float, threshold: float) -> bool:
        is_double = (
            self.last_time is not None
            and actor is self.last_actor
            and (now - self.last_time) <= threshold
        )
        if is_double:
            self.reset()
        else:
            self.last_time = now
            self.last_actor = actor
        return is_double

    def reset(self):
        self.last_time = None
        self.last_actor = None


@dataclass(slots=True)
class _Marker:
    """VTK resources and state for one visible marker."""

    overlay: object
    pts: object
    poly: object
    actor: object
    text_actor: object
    world_xyz: np.ndarray | None = None
    marker_id: int | None = None


class PickManager:
    """

    A minimal pick manager supporting:
      - Single click: print owner.name only
      - Double click (time-based):
          * If a marker is near the picked point -> delete the nearest marker (no new marker)
          * Else -> add a new numbered marker at the resolved position
            (PlotTube: picked point; PlotSphere/PlotRod: nearest point in owner.raw_coords)
      - Marker labels (2D text) are hidden during camera interaction and shown again after.
    """

    HELPER_MARKER_COLOR = (1.0, 0.3, 0.3)

    __descriptions__ = {
        "opts": "The OptsPickManager instance controlling behavior.",
        "_impl_owner_ref": (
            "A weak reference to the PlotFigure that owns this pick manager."
        ),
        "_impl_registry": "A registry dict: actor -> visual object",
        "_state_pick_count": "Monotonic counter for marker numbering (never decreases).",
        "_state_left_click": "Double-click tracker for left-button picking.",
        "_state_right_click": "Double-click tracker for right-button picking.",
        "_entity_markers": (
            "A list of marker packs; each pack holds VTK actors for one overlay point marker."
        ),
        "_entity_helper_markers": "A dict of panel/helper marker packs keyed by logical name.",
        "_entity_settings_action": "Menu action opening interaction settings for this window.",
        "_entity_figure_opts_action": "Menu action showing this PlotFigure opts snapshot.",
        "_entity_settings_dialog": "Live non-modal interaction settings dialog, if open.",
        "_entity_figure_opts_dialog": "Live non-modal figure-options dialog, if open.",
        "_impl_right_button_observer_id": (
            "VTK observer tag for the manager-owned right-button callback."
        ),
    }

    __slots__ = tuple(__descriptions__.keys()) + ("__weakref__",)

    def __init__(self, figure, opts: OptsPickManager | None = None, **kwargs):

        object.__setattr__(self, "_impl_owner_ref", weakref.ref(figure))
        object.__setattr__(self, "_impl_registry", {})
        object.__setattr__(self, "_state_pick_count", 0)
        object.__setattr__(self, "_state_left_click", _ClickTracker())
        object.__setattr__(self, "_state_right_click", _ClickTracker())
        object.__setattr__(self, "_entity_markers", [])
        object.__setattr__(self, "_entity_helper_markers", {})
        object.__setattr__(self, "_entity_settings_action", None)
        object.__setattr__(self, "_entity_figure_opts_action", None)
        object.__setattr__(self, "_entity_settings_dialog", None)
        object.__setattr__(self, "_entity_figure_opts_dialog", None)
        object.__setattr__(self, "_impl_right_button_observer_id", None)

        if opts is None:
            opts = OptsPickManager()
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_impl_owner_ref", weakref.ref(self))
        object.__setattr__(self, "opts", opts)

        fig = self.owner
        if fig is not None:
            iren = fig.pl.iren.interactor
            observer_id = iren.AddObserver(
                "RightButtonPressEvent", self._vtk_on_right_button_press
            )
            object.__setattr__(self, "_impl_right_button_observer_id", observer_id)
            self._helper_init_settings_menu()

    @property
    def owner(self):
        return self._impl_owner_ref()

    def _helper_apply_panel_throttle(self, value: int):
        fig = self.owner
        interacts = getattr(fig, "interacts", None) if fig is not None else None
        if interacts is None:
            return
        for panel in interacts:
            if hasattr(panel, "act_set_slider_throttle_ms"):
                panel.act_set_slider_throttle_ms(int(value))

    def _helper_get_or_create_settings_menu(self):
        fig = self.owner
        if fig is None:
            return None
        plotter = fig.pl
        menu_bar = getattr(plotter, "main_menu", None)
        if menu_bar is None and hasattr(plotter, "app_window"):
            menu_bar = plotter.app_window.menuBar()
        if menu_bar is None:
            return None
        for action in menu_bar.actions():
            text = action.text().replace("&", "").strip().lower()
            if text == "settings":
                return action.menu()
        return menu_bar.addMenu("Settings")

    def _helper_close_dialogs(self):
        for attr_name in (
            "_entity_settings_dialog",
            "_entity_figure_opts_dialog",
        ):
            dialog = getattr(self, attr_name, None)
            if dialog is None:
                continue
            try:
                dialog.close()
            except (AttributeError, RuntimeError, ReferenceError):
                pass

    def _helper_remove_right_button_observer(self):
        """Detach the VTK callback installed directly by this manager."""
        observer_id = self._impl_right_button_observer_id
        if observer_id is None:
            return
        fig = self.owner
        if fig is not None:
            try:
                fig.pl.iren.interactor.RemoveObserver(observer_id)
            except (AttributeError, RuntimeError, ReferenceError):
                pass
        object.__setattr__(self, "_impl_right_button_observer_id", None)

    def _helper_remove_settings_actions(self):
        """Remove Qt menu actions created by this manager."""
        settings_menu = self._helper_get_or_create_settings_menu()
        for attr_name in (
            "_entity_settings_action",
            "_entity_figure_opts_action",
        ):
            action = getattr(self, attr_name, None)
            if action is None:
                continue
            try:
                if settings_menu is not None:
                    settings_menu.removeAction(action)
                action.deleteLater()
            except (AttributeError, RuntimeError, ReferenceError):
                pass
            finally:
                object.__setattr__(self, attr_name, None)

    def _helper_open_settings_dialog(self):
        dialog_existing = self._entity_settings_dialog
        if dialog_existing is not None:
            dialog_existing.show()
            dialog_existing.raise_()
            dialog_existing.activateWindow()
            return

        fig = self.owner
        parent = (
            fig.pl.app_window
            if (fig is not None and hasattr(fig.pl, "app_window"))
            else None
        )
        dialog = PickSettingsDialog(self, parent=parent)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(
            lambda *_args: object.__setattr__(self, "_entity_settings_dialog", None)
        )
        object.__setattr__(self, "_entity_settings_dialog", dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _helper_open_figure_opts_dialog(self):
        dialog_existing = self._entity_figure_opts_dialog
        if dialog_existing is not None:
            dialog_existing.show()
            dialog_existing.raise_()
            dialog_existing.activateWindow()
            return

        fig = self.owner
        parent = (
            fig.pl.app_window
            if (fig is not None and hasattr(fig.pl, "app_window"))
            else None
        )
        dialog = FigureOptionsDialog(fig, parent=parent)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(
            lambda *_args: object.__setattr__(self, "_entity_figure_opts_dialog", None)
        )
        object.__setattr__(self, "_entity_figure_opts_dialog", dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _helper_init_settings_menu(self):
        fig = self.owner
        if (
            fig is None
            or self._entity_settings_action is not None
            or self._entity_figure_opts_action is not None
        ):
            return
        settings_menu = self._helper_get_or_create_settings_menu()
        if settings_menu is None:
            return
        action = settings_menu.addAction("Interaction Settings")
        action.triggered.connect(self._helper_open_settings_dialog)
        object.__setattr__(self, "_entity_settings_action", action)
        action = settings_menu.addAction("Show Figure Options")
        action.triggered.connect(self._helper_open_figure_opts_dialog)
        object.__setattr__(self, "_entity_figure_opts_action", action)

    # ---------------------------------------------------------------------
    # Registry: actor -> owner (PlotTube / PlotSphere / PlotRod / ...)
    # ---------------------------------------------------------------------
    def act_register(self, actor, owner):
        self._impl_registry[actor] = owner

    def act_unregister(self, actor, logger=None):
        if actor in self._impl_registry:
            del self._impl_registry[actor]

    def _helper_toggle_highlight(self, owner):
        """Apply the explicit right-click highlight/dehighlight contract."""
        silhouette = getattr(owner, "entity_silhouette", None)
        if silhouette is not None and silhouette.visibility:
            owner.act_dehighlight()
            return
        if not hasattr(owner, "act_highlight"):
            return
        if hasattr(owner, "state_is_silhouette"):
            object.__setattr__(owner, "state_is_silhouette", True)
        owner.act_highlight(
            color=self.opts.sil_color,
            opacity=self.opts.sil_opacity,
            width=self.opts.sil_width,
        )

    # ---------------------------------------------------------------------
    # Picking callback
    # ---------------------------------------------------------------------
    def _helper_callback(self, point, picker):

        actor = picker.GetActor() if picker is not None else None
        if actor is None or actor not in self._impl_registry:
            return

        owner = self._impl_registry[actor]

        is_double = self._state_left_click.consume(
            actor, time.monotonic(), self.opts.double_click_threshold
        )

        # Single click: do nothing.
        if not is_double:
            return

        # Double click: delete nearest marker if close; otherwise add a new marker.
        resolved, msg, _ = owner.act_resolve_pick(point)

        nearest_pack, nearest_d2 = self._helper_find_nearest_marker_pack(resolved)

        # Compare squared distances against the squared world-space threshold.
        thr = self.opts.marker_proximity_threshold
        if (
            nearest_pack is not None
            and nearest_d2 is not None
            and nearest_d2 <= (thr * thr)
        ):
            self._helper_remove_marker_pack(nearest_pack)
            pos = nearest_pack.world_xyz
            self.owner.console.println(
                f"remove point #{nearest_pack.marker_id}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
                f"on {str(owner)}"
            )
            self.owner.console.println(msg)

            return

        # No nearby marker -> add a new marker at resolved position.
        self._helper_add_marker(resolved)
        self.owner.console.println(
            f"picked point #{self._state_pick_count}: ({resolved[0]:.2f}, {resolved[1]:.2f}, {resolved[2]:.2f}) "
            f"on {owner.name!r}"
        )
        self.owner.console.println(msg)

    # ---------------------------------------------------------------------
    # Marker creation / removal
    # ---------------------------------------------------------------------
    def _helper_create_marker_pack(self):

        fig = self.owner
        if fig is None:
            return None

        # Expect PlotFigure to have overlay renderer prepared (layer=1)
        overlay = getattr(fig, "overlay", None)
        if overlay is None:
            return None

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(1)
        pts.SetPoint(0, 0.0, 0.0, 0.0)

        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)

        verts = vtk.vtkCellArray()
        verts.InsertNextCell(1)
        verts.InsertCellPoint(0)
        poly.SetVerts(verts)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToPoints()
        actor.GetProperty().SetRenderPointsAsSpheres(True)
        actor.GetProperty().SetPointSize(self.opts.marker_size)  # fixed for now
        actor.GetProperty().SetColor(*self.opts.marker_color)
        actor.GetProperty().LightingOff()
        actor.PickableOff()
        actor.SetVisibility(False)
        overlay.AddActor(actor)

        text = vtk.vtkTextActor()
        text.GetTextProperty().SetColor(0.0, 0.0, 0.0)  # black digits
        text.GetTextProperty().SetFontSize(
            self.opts.marker_font_size
        )  # tune with point size
        text.GetTextProperty().BoldOn()
        text.GetTextProperty().SetJustificationToCentered()
        text.GetTextProperty().SetVerticalJustificationToCentered()
        text.SetVisibility(False)
        overlay.AddActor2D(text)

        return _Marker(overlay, pts, poly, actor, text)

    def _helper_add_marker(self, xyz, marker_id=None):

        pack = self._helper_create_marker_pack()
        if pack is None:
            return

        fig = self.owner
        if fig is None:
            return

        xyz = np.asarray(xyz, dtype=float).reshape(
            3,
        )
        pack.world_xyz = xyz

        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        pack.pts.SetPoint(0, x, y, z)
        pack.pts.Modified()
        pack.poly.Modified()
        pack.actor.SetVisibility(True)

        if marker_id is None:
            object.__setattr__(self, "_state_pick_count", self._state_pick_count + 1)
            k = self._state_pick_count
            pack.marker_id = k
        else:
            pack.marker_id = marker_id
        k = pack.marker_id

        text = pack.text_actor
        text.SetInput(str(k))

        self._helper_update_one_marker_label_position(pack)
        text.SetVisibility(True)

        self._entity_markers.append(pack)

        fig.pl.render()

    def _helper_remove_marker_pack(self, pack):

        fig = self.owner
        if fig is None:
            return

        overlay = pack.overlay
        overlay.RemoveActor(pack.actor)
        overlay.RemoveActor2D(pack.text_actor)
        if pack in self._entity_markers:
            self._entity_markers.remove(pack)

        fig.pl.render()

    # ---------------------------------------------------------------------
    # Marker spatial query (world-space)
    # ---------------------------------------------------------------------
    def _helper_iter_all_marker_packs(self):
        for pack in self._entity_markers:
            yield pack
        for pack in self._entity_helper_markers.values():
            yield pack

    def act_set_helper_marker(self, key, xyz, marker_id=0):
        key = str(key)
        fig = self.owner
        if fig is None:
            return
        xyz = np.asarray(xyz, dtype=float).reshape(
            3,
        )
        pack = self._entity_helper_markers.get(key)
        if pack is None:
            pack = self._helper_create_marker_pack()
            if pack is None:
                return
            self._entity_helper_markers[key] = pack
        pack.world_xyz = xyz
        pack.marker_id = int(marker_id)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        pack.pts.SetPoint(0, x, y, z)
        pack.pts.Modified()
        pack.poly.Modified()
        pack.actor.GetProperty().SetColor(*self.HELPER_MARKER_COLOR)
        pack.actor.SetVisibility(True)
        text = pack.text_actor
        text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        text.SetInput(str(pack.marker_id))
        self._helper_update_one_marker_label_position(pack)
        text.SetVisibility(True)
        fig.pl.render()

    def act_remove_helper_marker(self, key):
        key = str(key)
        pack = self._entity_helper_markers.pop(key, None)
        if pack is None:
            return
        fig = self.owner
        if fig is None:
            return
        overlay = pack.overlay
        overlay.RemoveActor(pack.actor)
        overlay.RemoveActor2D(pack.text_actor)
        fig.pl.render()

    def act_clear_markers(self):
        """Remove all normal/helper markers and reset click-tracking state."""
        fig = self.owner
        for pack in list(self._entity_markers):
            pack.overlay.RemoveActor(pack.actor)
            pack.overlay.RemoveActor2D(pack.text_actor)
        for pack in list(self._entity_helper_markers.values()):
            pack.overlay.RemoveActor(pack.actor)
            pack.overlay.RemoveActor2D(pack.text_actor)
        self._entity_markers.clear()
        self._entity_helper_markers.clear()
        self._state_left_click.reset()
        self._state_right_click.reset()
        if fig is not None:
            fig.pl.render()

    def act_close(self):
        """Release dialogs and marker resources owned by this manager."""
        self._helper_close_dialogs()
        self._helper_remove_settings_actions()
        self._helper_remove_right_button_observer()
        self.act_clear_markers()

    def _helper_find_nearest_marker_pack(self, p):

        if not self._entity_markers:
            return None, None

        nearest_pack = None
        nearest_d2 = None

        for pack in self._entity_markers:
            xyz0 = pack.world_xyz
            if xyz0 is None:
                continue
            d = p - xyz0
            d2 = float(np.dot(d, d))
            if nearest_d2 is None or d2 < nearest_d2:
                nearest_d2 = d2
                nearest_pack = pack
        return nearest_pack, nearest_d2

    # ---------------------------------------------------------------------
    # Label update / interaction hooks
    # ---------------------------------------------------------------------
    def _helper_update_one_marker_label_position(self, pack):

        xyz = pack.world_xyz
        if xyz is None:
            return

        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        overlay = pack.overlay
        overlay.SetWorldPoint(x, y, z, 1.0)
        overlay.WorldToDisplay()
        dx, dy, _ = overlay.GetDisplayPoint()

        text = pack.text_actor

        text.SetDisplayPosition(int(dx), int(dy))

    def _helper_update_all_marker_labels_position(self):

        for pack in self._helper_iter_all_marker_packs():
            self._helper_update_one_marker_label_position(pack)

    def _helper_hide_marker_label_during_interaction(self):

        for pack in self._helper_iter_all_marker_packs():
            text = pack.text_actor
            if text.GetVisibility():
                text.SetVisibility(False)

    def _helper_show_marker_label_after_interaction(self):

        self._helper_update_all_marker_labels_position()

        for pack in self._helper_iter_all_marker_packs():
            pack.text_actor.SetVisibility(True)

    def _vtk_on_right_button_press(self, vtk_iren, _evt):

        fig = self.owner
        if fig is None:
            return

        # 1) pick (actor + world point)
        x, y = vtk_iren.GetEventPosition()

        # Right-click only needs actor-level picking.  Use the hardware prop
        # picker here rather than vtkCellPicker: instanced glyphs rendered by
        # vtkGlyph3DMapper do not expose a materialized polygonal output for a
        # geometric cell picker to interrogate, while vtkPropPicker is designed
        # to identify the rendered prop directly.
        picker = vtk.vtkPropPicker()
        picker.Pick(x, y, 0.0, fig.pl.renderer)

        actor = picker.GetActor() if picker is not None else None
        if actor is None or actor not in self._impl_registry:
            return
        owner = self._impl_registry[actor]

        # 2) right-double-click detect (time + same actor)
        is_double = self._state_right_click.consume(
            actor, time.monotonic(), self.opts.double_click_threshold
        )

        # Once clicked, switch the highlight status.  Silhouettes are created
        # lazily, so the first right-click must call act_highlight() even when
        # entity_silhouette does not exist yet.
        self._helper_toggle_highlight(owner)

        # Single click: print only.
        if not is_double:
            self.owner.console.println(str(owner))
            return

        # 3) on right-double-click

        if getattr(owner, "state_is_interactable", False):
            owner.act_interact()
