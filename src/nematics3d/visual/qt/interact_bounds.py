"""Qt control panel for Bounds geometry."""

import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from nematics3d.visual.plot_sphere import OptsSphere, PlotSphere
from nematics3d.visual.plot_tube import OptsTube, PlotTube
from nematics3d.visual.qt.panel_base import (
    LogTickMapper,
    MovePointConsole,
    PanelBase,
    make_labeled_slider_row,
)
from nematics3d.geometry import frame_from_spherical_roll, roll_angle_from_frame


class InteractBounds(PanelBase):
    def __init__(self, host, figure):
        self._helper_origin_visual = None
        self._helper_axis1_visual = None
        self._helper_axis2_visual = None
        self._helper_origin_radius = None
        self._helper_axis1_radius = None
        self._helper_axis2_radius = None
        super().__init__(host, figure, title=f"Controls of {host.name!r}")

    def _helper_get_host_visual_in_current_figure(self):
        for entry in self.host.entity_visuals:
            if entry.figure is self.fig and entry.tube is not None:
                return entry.tube
        return None

    def _helper_init_visual_style_snapshot(self):
        tube = self._helper_get_host_visual_in_current_figure()
        base_radius = None
        if tube is not None:
            try:
                base_radius = float(tube.opts.radius)
            except Exception:
                base_radius = None
        if base_radius is None or base_radius <= 0:
            base_radius = max(self.host.lengths) / 100.0
        self._helper_axis1_radius = float(base_radius) * 2.6
        self._helper_axis2_radius = float(base_radius) * 1.7
        self._helper_origin_radius = float(base_radius) * 3.0

    def _helper_build_axes_coords(self):
        origin = np.asarray(self.host.opts.origin, dtype=float)
        axis1 = np.asarray(self.host.opts.axis1, dtype=float)
        axis2 = np.asarray(self.host.calc_axis2, dtype=float)
        length1, length2, _length3 = self.host.lengths
        return (
            origin.reshape(1, 3),
            np.vstack([origin, origin + axis1 * length1]),
            np.vstack([origin, origin + axis2 * length2]),
        )

    def _iter_helper_visuals(self):
        for visual in (
            self._helper_origin_visual,
            self._helper_axis1_visual,
            self._helper_axis2_visual,
        ):
            if visual is not None:
                yield visual

    def _helper_create_helper_visuals(self):
        if self._helper_origin_visual is not None:
            return
        self._helper_init_visual_style_snapshot()
        origin_coords, axis1_coords, axis2_coords = self._helper_build_axes_coords()
        self._helper_origin_visual = PlotSphere(
            coords=origin_coords,
            figure=self.fig,
            name=f"{self.host.name} helper origin",
            category="bounds helper",
            opts=OptsSphere(
                color=(0.0, 0.0, 0.0),
                radius=float(self._helper_origin_radius),
                is_pickable=False,
                is_reset_camera=False,
            ),
        )
        self._helper_axis1_visual = PlotTube(
            coords=axis1_coords,
            figure=self.fig,
            name=f"{self.host.name} helper axis1",
            category="bounds helper",
            opts=OptsTube(
                color=(0.9, 0.15, 0.15),
                radius=float(self._helper_axis1_radius),
                is_pickable=False,
                is_reset_camera=False,
            ),
        )
        self._helper_axis2_visual = PlotTube(
            coords=axis2_coords,
            figure=self.fig,
            name=f"{self.host.name} helper axis2",
            category="bounds helper",
            opts=OptsTube(
                color=(0.15, 0.45, 0.95),
                radius=float(self._helper_axis2_radius),
                is_pickable=False,
                is_reset_camera=False,
            ),
        )
        for visual in self._iter_helper_visuals():
            object.__setattr__(visual, "state_is_interactable", False)
            object.__setattr__(visual, "state_is_silhouette", False)
            if hasattr(visual, "_helper_clear_silhouette"):
                visual._helper_clear_silhouette()

    def _update_helper_visuals(self, is_visible=True):
        if self._helper_origin_visual is None:
            self._helper_create_helper_visuals()
        origin_coords, axis1_coords, axis2_coords = self._helper_build_axes_coords()
        for visual, coords, radius in (
            (self._helper_origin_visual, origin_coords, self._helper_origin_radius),
            (self._helper_axis1_visual, axis1_coords, self._helper_axis1_radius),
            (self._helper_axis2_visual, axis2_coords, self._helper_axis2_radius),
        ):
            visual.act_commit(
                coords=coords, radius=float(radius), is_visible=is_visible
            )

    def _on_toggle_show_helpers(self, _state: int):
        if self.chk_is_show_helpers.isChecked():
            self._update_helper_visuals(is_visible=True)
            return
        for visual in self._iter_helper_visuals():
            visual.opts.is_visible = False

    def _helper_axis1_info_text(self, axis1):
        return self._vect_text(axis1, "axis1") + " (red)"

    def _update_axis_info_labels(self):
        axis1 = np.asarray(self.host.opts.axis1, dtype=float)
        axis2 = np.asarray(self.host.calc_axis2, dtype=float)
        axis3 = np.asarray(self.host.calc_axis3, dtype=float)
        self.axis1_info.setText(self._helper_axis1_info_text(axis1))
        self.axis2_info.setText(self._vect_text(axis2, "axis2") + " (blue)")
        self.axis3_info.setText(self._vect_text(axis3, "axis3"))

    def build_ui(self):
        axis1 = np.asarray(self.host.opts.axis1, dtype=float)
        axis2 = np.asarray(self.host.calc_axis2, dtype=float)
        length1, length2, length3 = self.host.lengths
        self.state = {
            "origin": np.asarray(self.host.opts.origin, dtype=float).copy(),
            "origin_move_step": 1.0,
            "is_origin_center": self.host.opts.alignment == "center",
            "length1": length1,
            "length2": length2,
            "length3": length3,
            "axis1_azimuth": self.get_azimuth(axis1),
            "axis1_polar_angle": self.get_polar_angle(axis1),
            "axis2_roll": self._helper_get_axis2_roll(axis1, axis2),
        }
        self.chk_is_show_helpers = QtWidgets.QCheckBox(
            "Whether to visualize origin, axis1 and axis2", self
        )
        self.chk_is_show_helpers.setChecked(True)
        self.layout.addWidget(self.chk_is_show_helpers)
        self.chk_is_show_helpers.stateChanged.connect(self._on_toggle_show_helpers)
        self.point_console = MovePointConsole(
            parent=self,
            state=self.state,
            center_key="origin",
            step_key="origin_move_step",
            title="Move Origin",
            step_min=0.01,
            step_max=100.0,
            step_tick_max=1000,
            step_fmt="{:.2f}",
            center_fmt="{:.2f}",
            on_move=self._commit_origin,
        )
        self.layout.addWidget(self.point_console.group)
        self.sliders["origin_move_step"] = self.point_console.slider_step

        group_length = QtWidgets.QGroupBox("Lengths", self)
        gl_length = QtWidgets.QVBoxLayout(group_length)
        self.layout.addWidget(group_length)
        self.chk_is_origin_center = QtWidgets.QCheckBox(
            "Whether to set origin at center (if not, set it at minimum corner)",
            group_length,
        )
        self.chk_is_origin_center.setChecked(self.state["is_origin_center"])
        gl_length.addWidget(self.chk_is_origin_center)
        self.chk_is_origin_center.stateChanged.connect(self._on_toggle_is_origin_center)
        for key in ("length1", "length2", "length3"):
            value = float(self.state[key])
            mapper = LogTickMapper(
                value_min=max(1e-4, 0.2 * value),
                value_max=max(1e-3, 5.0 * value),
                base=10.0,
            )
            self.sliders[key] = make_labeled_slider_row(
                parent=group_length,
                layout=gl_length,
                name=key,
                state_key=key,
                value_min=mapper.value_min,
                value_max=mapper.value_max,
                value_init=value,
                tick_to_value=mapper.tick_to_value,
                value_to_tick=mapper.value_to_tick,
                input_out_of_range="expand_max",
            )

        group_orient = QtWidgets.QGroupBox("Orientation", self)
        gl_orient = QtWidgets.QVBoxLayout(group_orient)
        self.layout.addWidget(group_orient)
        self.axis1_info = QtWidgets.QLabel(self._helper_axis1_info_text(axis1), self)
        self.axis2_info = QtWidgets.QLabel(
            self._vect_text(axis2, "axis2") + " (blue)", self
        )
        self.axis3_info = QtWidgets.QLabel(
            self._vect_text(self.host.calc_axis3, "axis3"), self
        )
        gl_orient.addWidget(self.axis1_info)
        gl_orient.addWidget(self.axis2_info)
        gl_orient.addWidget(self.axis3_info)
        self._update_axis_info_labels()
        for key, label, value_max in (
            ("axis1_azimuth", "Azimuth of axis1", 360),
            ("axis1_polar_angle", "Polar angle of axis1", 180),
            ("axis2_roll", "Roll of axis2", 360),
        ):
            self.sliders[key] = make_labeled_slider_row(
                parent=group_orient,
                layout=gl_orient,
                name=label,
                state_key=key,
                value_min=0,
                value_max=value_max,
                value_init=self.state[key],
                tick_to_value=lambda tick: tick / 10,
                value_to_tick=lambda value: int(value * 10),
                value_fmt="{:.1f}",
            )
        self.on_changed(0, is_commit=False)
        self._update_helper_visuals(is_visible=self.chk_is_show_helpers.isChecked())

    def _commit_origin(self, center):
        self._is_gui_updating = True
        try:
            self.host.act_commit(origin=np.asarray(center, dtype=float))
        finally:
            self._is_gui_updating = False

    def _helper_get_axis2_roll(self, axis1, axis2):
        return float(np.degrees(roll_angle_from_frame(axis1, axis2)))

    def _helper_build_orientation(self):
        return frame_from_spherical_roll(
            np.deg2rad(self.state["axis1_azimuth"]),
            np.deg2rad(self.state["axis1_polar_angle"]),
            np.deg2rad(self.state["axis2_roll"]),
        )

    def commit(self):
        axis1_now, axis2_now, _axis3_now = self._helper_build_orientation()
        alignment = "center" if self.state["is_origin_center"] else "min_corner"
        self._is_gui_updating = True
        try:
            self.host.act_commit(
                alignment=alignment,
                length1=float(self.state["length1"]),
                length2=float(self.state["length2"]),
                length3=float(self.state["length3"]),
                axis1=axis1_now,
                axis2=axis2_now,
            )
        finally:
            self._is_gui_updating = False

    def _on_toggle_is_origin_center(self, _state: int):
        self.state["is_origin_center"] = self.chk_is_origin_center.isChecked()
        self.commit()

    def _sync_func(self, **kwargs):
        if "origin" in kwargs:
            self.state["origin"] = np.asarray(self.host.opts.origin, dtype=float).copy()
            self.point_console._update_center_label()
        if "alignment" in kwargs:
            checked = self.host.opts.alignment == "center"
            with QSignalBlocker(self.chk_is_origin_center):
                self.chk_is_origin_center.setChecked(checked)
            self.state["is_origin_center"] = checked
        for key, value in zip(("length1", "length2", "length3"), self.host.lengths):
            if key in kwargs:
                self._sync_from_host_slider(key, value)
        if "axis1" in kwargs or "axis2" in kwargs:
            axis1 = np.asarray(self.host.opts.axis1, dtype=float)
            axis2 = np.asarray(self.host.calc_axis2, dtype=float)
            self._sync_from_host_slider("axis1_azimuth", self.get_azimuth(axis1))
            self._sync_from_host_slider(
                "axis1_polar_angle", self.get_polar_angle(axis1)
            )
            self._sync_from_host_slider(
                "axis2_roll", self._helper_get_axis2_roll(axis1, axis2)
            )
            self._update_axis_info_labels()
        if self.chk_is_show_helpers.isChecked():
            self._update_helper_visuals(is_visible=True)

    def on_close(self):
        super().on_close()
        for visual in tuple(self._iter_helper_visuals()):
            visual.act_remove()
        self._helper_origin_visual = None
        self._helper_axis1_visual = None
        self._helper_axis2_visual = None


__all__ = ["InteractBounds"]
