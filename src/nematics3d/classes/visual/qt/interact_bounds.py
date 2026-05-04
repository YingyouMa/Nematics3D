import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from nematics3d.general import rotation_matrix_from_vectors
from nematics3d.geometry import calc_vec_from_azimuth_polar
from ..plot_sphere import PlotSphere, OptsSphere
from ..plot_tube import PlotTube, OptsTube

from .panel_base import (
    PanelBase,
    MovePointConsole,
    LogTickMapper,
    make_labeled_slider_row,
)


class InteractBounds(PanelBase):

    def __init__(self, host, figure):
        object.__setattr__(self, "_is_continuous_interacting", False)
        object.__setattr__(self, "_impl_silhouette_state_backup", {})
        object.__setattr__(self, "_helper_origin_visual", None)
        object.__setattr__(self, "_helper_axis1_visual", None)
        object.__setattr__(self, "_helper_axis2_visual", None)
        object.__setattr__(self, "_helper_origin_radius", None)
        object.__setattr__(self, "_helper_axis1_radius", None)
        object.__setattr__(self, "_helper_axis2_radius", None)
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
            lengths = [
                float(self.host.opts.length1),
                float(
                    self.host.opts.length2
                    if self.host.opts.length2 is not None
                    else self.host.opts.length1
                ),
                float(
                    self.host.opts.length3
                    if self.host.opts.length3 is not None
                    else self.host.opts.length1
                ),
            ]
            base_radius = max(lengths) / 100.0

        object.__setattr__(self, "_helper_axis1_radius", float(base_radius) * 2.6)
        object.__setattr__(self, "_helper_axis2_radius", float(base_radius) * 1.7)
        object.__setattr__(self, "_helper_origin_radius", float(base_radius) * 3.0)

    def _helper_build_axes_coords(self):
        origin = np.asarray(self.host.opts.origin, dtype=float)
        axis1 = np.asarray(self.host.opts.axis1, dtype=float)
        axis2 = np.asarray(self.host.calc_axis2, dtype=float)
        length1 = float(self.host.opts.length1)
        length2 = float(
            self.host.opts.length2
            if self.host.opts.length2 is not None
            else self.host.opts.length1
        )
        axis1_coords = np.vstack([origin, origin + axis1 * length1])
        axis2_coords = np.vstack([origin, origin + axis2 * length2])
        return origin.reshape(1, 3), axis1_coords, axis2_coords

    def _helper_create_helper_visuals(self):
        if self._helper_origin_visual is not None:
            return

        self._helper_init_visual_style_snapshot()
        origin_coords, axis1_coords, axis2_coords = self._helper_build_axes_coords()

        origin_visual = PlotSphere(
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
        axis1_visual = PlotTube(
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
        axis2_visual = PlotTube(
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

        for visual in (origin_visual, axis1_visual, axis2_visual):
            object.__setattr__(visual, "state_is_interactable", False)
            object.__setattr__(visual, "state_is_silhouette", False)
            if hasattr(visual, "_helper_clear_silhouette"):
                visual._helper_clear_silhouette()

        object.__setattr__(self, "_helper_origin_visual", origin_visual)
        object.__setattr__(self, "_helper_axis1_visual", axis1_visual)
        object.__setattr__(self, "_helper_axis2_visual", axis2_visual)

    def _update_helper_visuals(self, is_visible=True):
        if self._helper_origin_visual is None:
            self._helper_create_helper_visuals()

        origin_coords, axis1_coords, axis2_coords = self._helper_build_axes_coords()
        self._helper_origin_visual.act_commit(
            coords=origin_coords,
            radius=float(self._helper_origin_radius),
            is_visible=is_visible,
        )
        self._helper_axis1_visual.act_commit(
            coords=axis1_coords,
            radius=float(self._helper_axis1_radius),
            is_visible=is_visible,
        )
        self._helper_axis2_visual.act_commit(
            coords=axis2_coords,
            radius=float(self._helper_axis2_radius),
            is_visible=is_visible,
        )

    def _on_toggle_show_helpers(self, _state: int):
        checked = self.chk_is_show_helpers.isChecked()
        if checked:
            self._update_helper_visuals(is_visible=True)
        else:
            if self._helper_origin_visual is not None:
                self._helper_origin_visual.opts.is_visible = False
            if self._helper_axis1_visual is not None:
                self._helper_axis1_visual.opts.is_visible = False
            if self._helper_axis2_visual is not None:
                self._helper_axis2_visual.opts.is_visible = False

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
        length1 = float(self.host.opts.length1)
        length2 = float(
            self.host.opts.length2 if self.host.opts.length2 is not None else length1
        )
        length3 = float(
            self.host.opts.length3 if self.host.opts.length3 is not None else length1
        )

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
            "Whether to visualize origin, axis1 and axis2",
            self,
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
            on_press=lambda *_args: self._helper_begin_continuous_interaction(),
            on_release=lambda *_args: self._helper_end_continuous_interaction(),
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
        gl_orient.addWidget(self.axis1_info)

        self.axis2_info = QtWidgets.QLabel(
            self._vect_text(axis2, "axis2") + " (blue)", self
        )
        gl_orient.addWidget(self.axis2_info)

        self.axis3_info = QtWidgets.QLabel(
            self._vect_text(self.host.calc_axis3, "axis3"),
            self,
        )
        gl_orient.addWidget(self.axis3_info)
        self._update_axis_info_labels()

        self.sliders["axis1_azimuth"] = make_labeled_slider_row(
            parent=group_orient,
            layout=gl_orient,
            name="Azimuth of axis1",
            state_key="axis1_azimuth",
            value_min=0,
            value_max=360,
            value_init=self.state["axis1_azimuth"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )

        self.sliders["axis1_polar_angle"] = make_labeled_slider_row(
            parent=group_orient,
            layout=gl_orient,
            name="Polar angle of axis1",
            state_key="axis1_polar_angle",
            value_min=0,
            value_max=180,
            value_init=self.state["axis1_polar_angle"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )

        self.sliders["axis2_roll"] = make_labeled_slider_row(
            parent=group_orient,
            layout=gl_orient,
            name="Roll of axis2",
            state_key="axis2_roll",
            value_min=0,
            value_max=360,
            value_init=self.state["axis2_roll"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )

        self.on_changed(0, is_commit=False)
        self._update_helper_visuals(is_visible=self.chk_is_show_helpers.isChecked())

    def _iter_silhouette_targets(self):
        targets = []

        for entry in self.host.entity_visuals:
            figure = entry.figure
            tube = entry.tube
            if figure is self.fig and tube is not None:
                targets.append(tube)

        for glyph in self.host.glyph_subscribers:
            if getattr(glyph, "fig", None) is self.fig:
                targets.append(glyph)

        seen = set()
        for visual in targets:
            ident = id(visual)
            if ident in seen:
                continue
            seen.add(ident)
            yield visual

    def _helper_begin_continuous_interaction(self, *_args):
        if self._is_continuous_interacting:
            return
        object.__setattr__(self, "_is_continuous_interacting", True)
        backups = {}
        for visual in self._iter_silhouette_targets():
            if not hasattr(visual, "state_is_silhouette"):
                continue
            backups[id(visual)] = bool(getattr(visual, "state_is_silhouette", True))
            object.__setattr__(visual, "state_is_silhouette", False)
            if hasattr(visual, "_helper_clear_silhouette"):
                visual._helper_clear_silhouette()
        object.__setattr__(self, "_impl_silhouette_state_backup", backups)

    def _helper_end_continuous_interaction(self, *_args):
        if not self._is_continuous_interacting:
            return
        object.__setattr__(self, "_is_continuous_interacting", False)
        backups = getattr(self, "_impl_silhouette_state_backup", {})
        for visual in self._iter_silhouette_targets():
            if not hasattr(visual, "state_is_silhouette"):
                continue
            is_enabled = backups.get(id(visual), True)
            object.__setattr__(visual, "state_is_silhouette", bool(is_enabled))
            if (
                is_enabled
                and getattr(visual, "entity_actor", None) is not None
                and hasattr(visual, "_helper_add_silhouette")
            ):
                visual._helper_add_silhouette()
        object.__setattr__(self, "_impl_silhouette_state_backup", {})

    def _commit_origin(self, center):
        self._is_gui_updating = True
        try:
            self.host.act_commit(origin=np.asarray(center, dtype=float))
        finally:
            self._is_gui_updating = False

    def _helper_calc_axis2_reference(self, axis1):
        rotation = rotation_matrix_from_vectors((1, 0, 0), axis1)
        axis2_reference = rotation @ np.array([0.0, 1.0, 0.0])
        axis2_reference /= np.linalg.norm(axis2_reference)
        return axis2_reference

    def _helper_rotate_about_axis(self, vec, axis, angle_rad):
        vec = np.asarray(vec, dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return (
            vec * cos_a
            + np.cross(axis, vec) * sin_a
            + axis * (axis @ vec) * (1.0 - cos_a)
        )

    def _helper_get_axis2_roll(self, axis1, axis2):
        axis1 = np.asarray(axis1, dtype=float)
        axis2 = np.asarray(axis2, dtype=float)
        axis2_reference = self._helper_calc_axis2_reference(axis1)

        cross = np.cross(axis2_reference, axis2)
        sin_angle = float(axis1 @ cross)
        cos_angle = float(axis2_reference @ axis2)
        angle = np.rad2deg(np.arctan2(sin_angle, cos_angle))
        return angle % 360.0

    def _helper_build_orientation(self):
        axis1_azimuth = np.deg2rad(self.state["axis1_azimuth"])
        axis1_polar = np.deg2rad(self.state["axis1_polar_angle"])
        axis1_now = np.asarray(
            calc_vec_from_azimuth_polar(axis1_azimuth, axis1_polar),
            dtype=float,
        )

        axis2_reference = self._helper_calc_axis2_reference(axis1_now)
        axis2_roll = np.deg2rad(self.state["axis2_roll"])
        axis2_now = self._helper_rotate_about_axis(
            axis2_reference,
            axis1_now,
            axis2_roll,
        )
        axis2_now /= np.linalg.norm(axis2_now)
        axis3_now = np.cross(axis1_now, axis2_now)
        axis3_now /= np.linalg.norm(axis3_now)
        return axis1_now, axis2_now, axis3_now

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

    # ==================== OVERRIDE ====================
    # InteractBounds overrides PanelBase._sync_func because the bounds
    # panel must keep coupled origin/alignment/orientation widgets in sync
    # with Bounds updates that may come from outside the panel.
    # ==================================================
    def _sync_func(self, **kwargs):
        is_external = self._helper_sync_update_live_backup(kwargs)

        if is_external:
            if "origin" in kwargs:
                self.state["origin"] = np.asarray(
                    self.host.opts.origin, dtype=float
                ).copy()
                self.point_console._update_center_label()
            if "alignment" in kwargs:
                checked = self.host.opts.alignment == "center"
                with QSignalBlocker(self.chk_is_origin_center):
                    self.chk_is_origin_center.setChecked(checked)
                self.state["is_origin_center"] = checked
            if "length1" in kwargs:
                self._sync_from_host_slider("length1", self.host.opts.length1)
            if "length2" in kwargs:
                value = (
                    self.host.opts.length2
                    if self.host.opts.length2 is not None
                    else self.host.opts.length1
                )
                self._sync_from_host_slider("length2", value)
            if "length3" in kwargs:
                value = (
                    self.host.opts.length3
                    if self.host.opts.length3 is not None
                    else self.host.opts.length1
                )
                self._sync_from_host_slider("length3", value)
            if "axis1" in kwargs or "axis2" in kwargs:
                axis1 = np.asarray(self.host.opts.axis1, dtype=float)
                axis2 = np.asarray(self.host.calc_axis2, dtype=float)
                axis3 = np.asarray(self.host.calc_axis3, dtype=float)
                self._sync_from_host_slider("axis1_azimuth", self.get_azimuth(axis1))
                self._sync_from_host_slider(
                    "axis1_polar_angle",
                    self.get_polar_angle(axis1),
                )
                self._sync_from_host_slider(
                    "axis2_roll",
                    self._helper_get_axis2_roll(axis1, axis2),
                )
        if "axis1" in kwargs or "axis2" in kwargs:
            self._update_axis_info_labels()

        if self.chk_is_show_helpers.isChecked():
            self._update_helper_visuals(is_visible=True)

    def on_close(self):
        self._helper_end_continuous_interaction()
        super().on_close()
        if self._helper_origin_visual is not None:
            self._helper_origin_visual.act_remove()
        if self._helper_axis1_visual is not None:
            self._helper_axis1_visual.act_remove()
        if self._helper_axis2_visual is not None:
            self._helper_axis2_visual.act_remove()
