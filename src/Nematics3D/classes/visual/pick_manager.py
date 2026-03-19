import weakref
import time
import vtk
import numpy as np
from qtpy import QtCore, QtWidgets
from dataclasses import dataclass, field

from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.general import find_nearest_point, closest_point_on_polyline
from Nematics3D.datatypes import (
    as_Number,
    ColorRGB,
    as_ColorRGB,
)
from ..opts import merge_opts_all


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
        "double_click_threshold": lambda v, d: as_Number(v, name=d, replace=0.3),
        "marker_proximity_threshold": lambda v, d: as_Number(v, name=d, replace=0.5),
        "marker_size": lambda v, d: as_Number(v, name=d, replace=14),
        "marker_color": lambda v, d: as_ColorRGB(v, name=d, replace=(1, 1, 0)),
        "marker_font_size": lambda v, d: as_Number(v, name=d, replace=14),
        "sil_color": lambda v, d: as_ColorRGB(v, name=d, replace=(0, 0, 0)),
        "sil_opacity": lambda v, d: as_Number(v, name=d, value_range=(0, 1)),
        "sil_width": lambda v, d: as_Number(v, name=d, value_range=(0, np.inf)),
        "slider_throttle_ms": lambda v, d: int(
            as_Number(v, name=d, value_range=(1, 1000))
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
                    pack["actor"].GetProperty().SetPointSize(value)

            elif key == "marker_color":
                for pack in owner._entity_markers:
                    pack["actor"].GetProperty().SetColor(*value)

            elif key == "marker_font_size":
                for pack in markers:
                    pack["text_actor"].GetTextProperty().SetFontSize(value)

            elif key == "slider_throttle_ms":
                owner._helper_apply_panel_throttle(int(value))
                return

            else:
                for glyph in owner._impl_registry.values():
                    if (
                        hasattr(glyph, "_entity_silhouette")
                        and glyph._entity_silhouette.visibility
                    ):
                        if key == "sil_color":
                            glyph._entity_silhouette.prop.color = value
                        if key == "sil_opacity":
                            glyph._entity_silhouette.prop.opacity = value
                        if key == "sil_width":
                            glyph._entity_silhouette.prop.line_width = value

            owner.owner.pl.render()


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
        "_impl_active_glyphs": "Set of currently active glyphs (multi-selection).",
        "_state_pick_count": "Monotonic counter for marker numbering (never decreases).",
        "_state_last_click_time": "Last click timestamp (monotonic time) for double-click detection.",
        "_state_last_click_actor": "Last clicked actor for double-click detection.",
        "_state_last_rclick_time": "Last RIGHT click timestamp for right-double-click detection.",
        "_state_last_rclick_actor": "Last RIGHT clicked actor for right-double-click detection.",
        "_entity_markers": (
            "A list of marker packs; each pack holds VTK actors for one overlay point marker."
        ),
        "_entity_helper_markers": "A dict of panel/helper marker packs keyed by logical name.",
        "_entity_settings_action": "Menu action opening interaction settings for this window.",
    }

    __slots__ = tuple(__descriptions__.keys()) + ("__weakref__",)

    def __init__(self, figure, opts: OptsPickManager | None = None, **kwargs):

        object.__setattr__(self, "_impl_owner_ref", weakref.ref(figure))
        object.__setattr__(self, "_impl_registry", {})
        object.__setattr__(self, "_state_pick_count", 0)
        object.__setattr__(self, "_state_last_click_time", None)
        object.__setattr__(self, "_state_last_click_actor", None)
        object.__setattr__(self, "_state_last_rclick_time", None)
        object.__setattr__(self, "_state_last_rclick_actor", None)
        object.__setattr__(self, "_entity_markers", [])
        object.__setattr__(self, "_entity_helper_markers", {})
        object.__setattr__(self, "_impl_active_glyphs", [])
        object.__setattr__(self, "_entity_settings_action", None)

        if opts is None:
            opts = OptsPickManager()
        opts = merge_opts_all({"": opts}, kwargs, type(self).__name__)[""]
        object.__setattr__(opts, "_impl_owner_ref", weakref.ref(self))
        object.__setattr__(self, "opts", opts)

        fig = self.owner
        if fig is not None:
            iren = fig.pl.iren.interactor
            iren.AddObserver("RightButtonPressEvent", self._vtk_on_right_button_press)
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

    def _helper_open_settings_dialog(self):
        fig = self.owner
        parent = (
            fig.pl.app_window
            if (fig is not None and hasattr(fig.pl, "app_window"))
            else None
        )
        dialog = QtWidgets.QDialog(parent)
        dialog.setWindowTitle("Interaction Settings")
        layout = QtWidgets.QVBoxLayout(dialog)

        group_panel = QtWidgets.QGroupBox("Panel", dialog)
        form_panel = QtWidgets.QFormLayout(group_panel)
        layout.addWidget(group_panel)

        spin_throttle = QtWidgets.QSpinBox(dialog)
        spin_throttle.setRange(1, 1000)
        spin_throttle.setSingleStep(5)
        spin_throttle.setValue(int(self.opts.slider_throttle_ms))
        form_panel.addRow("Slider throttle (ms)", spin_throttle)

        spin_double_click = QtWidgets.QDoubleSpinBox(dialog)
        spin_double_click.setRange(0.01, 10.0)
        spin_double_click.setSingleStep(0.05)
        spin_double_click.setDecimals(3)
        spin_double_click.setValue(float(self.opts.double_click_threshold))
        form_panel.addRow("Double click threshold", spin_double_click)

        group_marker = QtWidgets.QGroupBox("Marker", dialog)
        form_marker = QtWidgets.QFormLayout(group_marker)
        layout.addWidget(group_marker)

        spin_marker_proximity = QtWidgets.QDoubleSpinBox(dialog)
        spin_marker_proximity.setRange(0.0, 1000.0)
        spin_marker_proximity.setSingleStep(0.05)
        spin_marker_proximity.setDecimals(3)
        spin_marker_proximity.setValue(float(self.opts.marker_proximity_threshold))
        form_marker.addRow("Proximity threshold", spin_marker_proximity)

        spin_marker_size = QtWidgets.QSpinBox(dialog)
        spin_marker_size.setRange(1, 200)
        spin_marker_size.setSingleStep(1)
        spin_marker_size.setValue(int(self.opts.marker_size))
        form_marker.addRow("Size", spin_marker_size)

        spin_marker_font_size = QtWidgets.QSpinBox(dialog)
        spin_marker_font_size.setRange(1, 200)
        spin_marker_font_size.setSingleStep(1)
        spin_marker_font_size.setValue(int(self.opts.marker_font_size))
        form_marker.addRow("Font size", spin_marker_font_size)

        marker_color_widget = QtWidgets.QWidget(dialog)
        marker_color_layout = QtWidgets.QHBoxLayout(marker_color_widget)
        marker_color_layout.setContentsMargins(0, 0, 0, 0)
        marker_color_layout.setSpacing(6)
        spin_marker_color_r = QtWidgets.QDoubleSpinBox(marker_color_widget)
        spin_marker_color_g = QtWidgets.QDoubleSpinBox(marker_color_widget)
        spin_marker_color_b = QtWidgets.QDoubleSpinBox(marker_color_widget)
        for spin, value in zip(
            (spin_marker_color_r, spin_marker_color_g, spin_marker_color_b),
            self.opts.marker_color,
        ):
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(3)
            spin.setMinimumWidth(80)
            spin.setValue(float(value))
            marker_color_layout.addWidget(spin)
        form_marker.addRow("Color (r g b)", marker_color_widget)

        group_silhouette = QtWidgets.QGroupBox("Silhouette", dialog)
        form_silhouette = QtWidgets.QFormLayout(group_silhouette)
        layout.addWidget(group_silhouette)

        spin_sil_opacity = QtWidgets.QDoubleSpinBox(dialog)
        spin_sil_opacity.setRange(0.0, 1.0)
        spin_sil_opacity.setSingleStep(0.05)
        spin_sil_opacity.setDecimals(3)
        spin_sil_opacity.setValue(float(self.opts.sil_opacity))
        form_silhouette.addRow("Opacity", spin_sil_opacity)

        spin_sil_width = QtWidgets.QDoubleSpinBox(dialog)
        spin_sil_width.setRange(0.0, 1000.0)
        spin_sil_width.setSingleStep(0.5)
        spin_sil_width.setDecimals(3)
        spin_sil_width.setValue(float(self.opts.sil_width))
        form_silhouette.addRow("Width", spin_sil_width)

        sil_color_widget = QtWidgets.QWidget(dialog)
        sil_color_layout = QtWidgets.QHBoxLayout(sil_color_widget)
        sil_color_layout.setContentsMargins(0, 0, 0, 0)
        sil_color_layout.setSpacing(6)
        spin_sil_color_r = QtWidgets.QDoubleSpinBox(sil_color_widget)
        spin_sil_color_g = QtWidgets.QDoubleSpinBox(sil_color_widget)
        spin_sil_color_b = QtWidgets.QDoubleSpinBox(sil_color_widget)
        for spin, value in zip(
            (spin_sil_color_r, spin_sil_color_g, spin_sil_color_b),
            self.opts.sil_color,
        ):
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(3)
            spin.setMinimumWidth(80)
            spin.setValue(float(value))
            sil_color_layout.addWidget(spin)
        form_silhouette.addRow("Color (r g b)", sil_color_widget)

        def _apply_settings():
            self.opts.slider_throttle_ms = int(spin_throttle.value())
            self.opts.double_click_threshold = float(spin_double_click.value())
            self.opts.marker_proximity_threshold = float(spin_marker_proximity.value())
            self.opts.marker_size = int(spin_marker_size.value())
            self.opts.marker_font_size = int(spin_marker_font_size.value())
            self.opts.marker_color = (
                float(spin_marker_color_r.value()),
                float(spin_marker_color_g.value()),
                float(spin_marker_color_b.value()),
            )
            self.opts.sil_opacity = float(spin_sil_opacity.value())
            self.opts.sil_width = float(spin_sil_width.value())
            self.opts.sil_color = (
                float(spin_sil_color_r.value()),
                float(spin_sil_color_g.value()),
                float(spin_sil_color_b.value()),
            )

        buttons = QtWidgets.QDialogButtonBox(parent=dialog)
        btn_ok = buttons.addButton(QtWidgets.QDialogButtonBox.Ok)
        btn_apply = buttons.addButton(QtWidgets.QDialogButtonBox.Apply)
        btn_cancel = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        btn_ok.clicked.connect(lambda: (_apply_settings(), dialog.accept()))
        btn_apply.clicked.connect(_apply_settings)
        btn_cancel.clicked.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.exec()

    def _helper_init_settings_menu(self):
        fig = self.owner
        if fig is None or self._entity_settings_action is not None:
            return
        settings_menu = self._helper_get_or_create_settings_menu()
        if settings_menu is None:
            return
        action = settings_menu.addAction("Interaction Settings")
        action.triggered.connect(self._helper_open_settings_dialog)
        object.__setattr__(self, "_entity_settings_action", action)

    # ---------------------------------------------------------------------
    # Registry: actor -> owner (PlotTube / PlotSphere / PlotRod / ...)
    # ---------------------------------------------------------------------
    def act_register(self, actor, owner):
        self._impl_registry[actor] = owner

    def act_unregister(self, actor, logger=None):
        if actor in self._impl_registry:
            del self._impl_registry[actor]

    # ---------------------------------------------------------------------
    # Picking callback
    # ---------------------------------------------------------------------
    def _helper_callback(self, point, picker):

        actor = picker.GetActor() if picker is not None else None
        if actor is None or actor not in self._impl_registry:
            return

        owner = self._impl_registry[actor]

        now = time.monotonic()
        last_t = self._state_last_click_time
        last_a = self._state_last_click_actor

        # Detect double-click: same actor within a short time window.
        is_double = (
            last_t is not None
            and (actor is last_a)
            and ((now - last_t) <= self.opts.double_click_threshold)
        )

        # Always update last-click state after printing.
        object.__setattr__(self, "_state_last_click_time", now)
        object.__setattr__(self, "_state_last_click_actor", actor)

        # Single click: do nothing.
        if not is_double:
            return

        # Double click: delete nearest marker if close; otherwise add a new marker.
        resolved, msg, _ = owner._helper_resolve_pick(point)

        nearest_pack, nearest_d2 = self._helper_find_nearest_marker_pack(resolved)

        # World-space threshold (tune as needed)
        thr = self.opts.marker_proximity_threshold
        if nearest_pack is not None and nearest_d2 is not None and nearest_d2 <= thr:
            self._helper_remove_marker_pack(nearest_pack)
            pos = nearest_pack["world_xyz"]
            self.owner.console.println(
                f"remove point #{nearest_pack['id']}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
                f"on {str(owner)}"
            )
            self.owner.console.println(msg)

            object.__setattr__(self, "_state_last_click_time", None)
            object.__setattr__(self, "_state_last_click_actor", None)
            return

        # No nearby marker -> add a new marker at resolved position.
        self._helper_add_marker(resolved)
        self.owner.console.println(
            f"picked point #{self._state_pick_count}: ({resolved[0]:.2f}, {resolved[1]:.2f}, {resolved[2]:.2f}) "
            f"on {owner.name!r}"
        )
        self.owner.console.println(msg)

        object.__setattr__(self, "_state_last_click_time", None)
        object.__setattr__(self, "_state_last_click_actor", None)

    # ---------------------------------------------------------------------
    # Marker creation / removal
    # ---------------------------------------------------------------------
    def _helper_create_marker_pack(self):

        fig = self.owner
        if fig is None:
            return None

        # Expect PlotFigure to have overlay renderer prepared (layer=1)
        overlay = getattr(fig, "_entity_overlay", None)
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

        pack = {
            "overlay": overlay,
            "pts": pts,
            "poly": poly,
            "actor": actor,
            "text_actor": text,
            "world_xyz": None,
            "id": None,
        }
        return pack

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
        pack["world_xyz"] = xyz

        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        pack["pts"].SetPoint(0, x, y, z)
        pack["pts"].Modified()
        pack["poly"].Modified()
        pack["actor"].SetVisibility(True)

        if marker_id is None:
            object.__setattr__(self, "_state_pick_count", self._state_pick_count + 1)
            k = self._state_pick_count
            pack["id"] = k
        else:
            pack["id"] = marker_id
        k = pack["id"]

        text = pack["text_actor"]
        text.SetInput(str(k))

        self._helper_update_one_marker_label_position(pack)
        text.SetVisibility(True)

        self._entity_markers.append(pack)

        fig.pl.render()

    def _helper_remove_marker_pack(self, pack):

        fig = self.owner
        if fig is None:
            return

        overlay = pack["overlay"]
        overlay.RemoveActor(pack["actor"])
        overlay.RemoveActor2D(pack["text_actor"])
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
        pack["world_xyz"] = xyz
        pack["id"] = int(marker_id)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        pack["pts"].SetPoint(0, x, y, z)
        pack["poly"].Modified()
        pack["actor"].GetProperty().SetColor(*self.HELPER_MARKER_COLOR)
        pack["actor"].SetVisibility(True)
        text = pack["text_actor"]
        text.GetTextProperty().SetColor(*self.HELPER_MARKER_COLOR)
        text.SetInput(str(pack["id"]))
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
        overlay = pack["overlay"]
        overlay.RemoveActor(pack["actor"])
        overlay.RemoveActor2D(pack["text_actor"])
        fig.pl.render()

    def _helper_find_nearest_marker_pack(self, p):

        if not self._entity_markers:
            return None, None

        nearest_pack = None
        nearest_d2 = None

        for pack in self._entity_markers:
            xyz0 = pack.get("world_xyz", None)
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

        xyz = pack.get("world_xyz", None)
        if xyz is None:
            return

        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

        overlay = pack["overlay"]
        overlay.SetWorldPoint(x, y, z, 1.0)
        overlay.WorldToDisplay()
        dx, dy, _ = overlay.GetDisplayPoint()

        text = pack.get("text_actor", None)
        if text is None:
            return

        text.SetDisplayPosition(int(dx), int(dy))

    def _helper_update_all_marker_labels_position(self):

        for pack in self._helper_iter_all_marker_packs():
            self._helper_update_one_marker_label_position(pack)

    def _helper_hide_marker_label_during_interaction(self):

        for pack in self._helper_iter_all_marker_packs():
            text = pack.get("text_actor", None)
            if text is not None and text.GetVisibility():
                text.SetVisibility(False)

    def _helper_show_marker_label_after_interaction(self):

        self._helper_update_all_marker_labels_position()

        for pack in self._helper_iter_all_marker_packs():
            text = pack.get("text_actor", None)
            if text is not None:
                text.SetVisibility(True)

    def _vtk_on_right_button_press(self, vtk_iren, _evt):

        fig = self.owner
        if fig is None:
            return

        # 1) pick (actor + world point)
        x, y = vtk_iren.GetEventPosition()

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.Pick(x, y, 0.0, fig.pl.renderer)

        actor = picker.GetActor() if picker is not None else None
        if actor is None or actor not in self._impl_registry:
            return
        owner = self._impl_registry[actor]

        # 2) right-double-click detect (time + same actor)
        now = time.monotonic()
        last_t = self._state_last_rclick_time
        last_a = self._state_last_rclick_actor

        is_double = (
            last_t is not None
            and (actor is last_a)
            and ((now - last_t) <= self.opts.double_click_threshold)
        )

        object.__setattr__(self, "_state_last_rclick_time", now)
        object.__setattr__(self, "_state_last_rclick_actor", actor)

        # Once clicked, switch the highlight status
        if hasattr(owner, "_entity_silhouette"):
            if owner._entity_silhouette.visibility == True:
                owner.act_dehighlight()
            else:
                owner.act_highlight(
                    color=self.opts.sil_color,
                    opacity=self.opts.sil_opacity,
                    width=self.opts.sil_width,
                )

        # Single click: print only.
        if not is_double:
            self.owner.console.println(str(owner))
            return

        # 3) on right-double-click

        if getattr(owner, "_state_is_interactable", False):
            owner.act_interact()

        # reset to avoid triple-trigger
        object.__setattr__(self, "_state_last_rclick_time", None)
        object.__setattr__(self, "_state_last_rclick_actor", None)
