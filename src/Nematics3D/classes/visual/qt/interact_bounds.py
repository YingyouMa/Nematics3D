import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from Nematics3D.general import rotation_matrix_from_vectors
from Nematics3D.geometry import calc_vec_from_azimuth_polar

from .panel_base import (
    PanelBase,
    MovePointConsole,
    LogTickMapper,
    make_labeled_slider_row,
)


class InteractBounds(PanelBase):
    def __init__(self, host, figure):
        super().__init__(host, figure, title=f"Controls of {host.name!r}")

    def build_ui(self):
        axis1 = np.asarray(self.host.opts.axis1, dtype=float)
        axis2 = np.asarray(self.host._calc_axis2, dtype=float)
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
            )

        group_orient = QtWidgets.QGroupBox("Orientation", self)
        gl_orient = QtWidgets.QVBoxLayout(group_orient)
        self.layout.addWidget(group_orient)

        self.axis1_info = QtWidgets.QLabel(self._vect_text(axis1, "axis1"), self)
        gl_orient.addWidget(self.axis1_info)

        self.axis2_info = QtWidgets.QLabel(self._vect_text(axis2, "axis2"), self)
        gl_orient.addWidget(self.axis2_info)

        self.axis3_info = QtWidgets.QLabel(
            self._vect_text(self.host._calc_axis3, "axis3"),
            self,
        )
        gl_orient.addWidget(self.axis3_info)

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

        for key, item in self.sliders.items():
            if key == "origin_move_step":
                item.slider.valueChanged.connect(
                    lambda _v=0: self.on_changed(is_commit=False)
                )
            else:
                item.slider.valueChanged.connect(self.on_changed)

        self.on_changed(0, is_commit=False)

    def _commit_origin(self, center):
        self.host.act_commit(origin=np.asarray(center, dtype=float))

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

        self.host.act_commit(
            alignment=alignment,
            length1=float(self.state["length1"]),
            length2=float(self.state["length2"]),
            length3=float(self.state["length3"]),
            axis1=axis1_now,
            axis2=axis2_now,
        )

    def _on_toggle_is_origin_center(self, _state: int):
        self.state["is_origin_center"] = self.chk_is_origin_center.isChecked()
        self.commit()

    # ==================== OVERRIDE ====================
    # InteractBounds overrides PanelBase._sync_func because the bounds
    # panel must keep coupled origin/alignment/orientation widgets in sync
    # with Bounds updates that may come from outside the panel.
    # ==================================================
    def _sync_func(self, **kwargs):
        if not getattr(self, "_is_gui_updating", False):
            self.host._opts_backup[self.str_now_live].update(kwargs)

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
                axis2 = np.asarray(self.host._calc_axis2, dtype=float)
                axis3 = np.asarray(self.host._calc_axis3, dtype=float)
                self._sync_from_host_slider("axis1_azimuth", self.get_azimuth(axis1))
                self._sync_from_host_slider(
                    "axis1_polar_angle",
                    self.get_polar_angle(axis1),
                )
                self._sync_from_host_slider(
                    "axis2_roll",
                    self._helper_get_axis2_roll(axis1, axis2),
                )
                self.axis1_info.setText(self._vect_text(axis1, "axis1"))
                self.axis2_info.setText(self._vect_text(axis2, "axis2"))
                self.axis3_info.setText(self._vect_text(axis3, "axis3"))
