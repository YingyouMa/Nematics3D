"""Interactive controls for Cartesian plane-sampling grids."""

from __future__ import annotations

import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from nematics3d.geometry import (
    azimuth_from_vector,
    plane_azimuth_from_direction,
    polar_angle_from_vector,
    rotation_matrix_from_vectors,
    vector_from_spherical_angles,
)
from nematics3d.grid import is_grid_transform_identity
from nematics3d.visual.plot_rod import PlotRod
from nematics3d.visual.plot_sphere import PlotSphere
from nematics3d.visual.qt.panel_base import (
    LogTickMapper,
    MovePointConsole,
    PanelBase,
    make_labeled_slider_row,
)


class InteractPlane(PanelBase):
    """Control a Cartesian ``PlaneGrid`` and refresh its owning sampled field."""

    @staticmethod
    def _helper_format_three_vector(values) -> str:
        values = np.asarray(values, dtype=float).reshape(3)
        return f"({values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f})"

    @staticmethod
    def _helper_dataset_grid_spacing_lines(field) -> list[str]:
        interpolator = getattr(field, "interpolator", None)
        grid_field = getattr(interpolator, "owner", None)
        dataset = getattr(grid_field, "owner", None)
        if dataset is None:
            return ["Grid axes in physical space:", "Unavailable"]

        spacing = getattr(dataset, "calc_grid_spacing", None)
        transform = getattr(dataset, "raw_grid_transform", None)
        if spacing is None or transform is None:
            return ["Grid axes in physical space:", "Unavailable"]

        spacing = np.asarray(spacing, dtype=float).reshape(3)
        directions = (
            np.eye(3, dtype=float)
            if is_grid_transform_identity(transform)
            else np.asarray(transform, dtype=float)
        )
        lines = ["Grid axes in physical space:"]
        for idx, name in enumerate(("i", "j", "k")):
            basis = directions[:, idx]
            norm = np.linalg.norm(basis)
            direction = np.zeros(3) if np.isclose(norm, 0.0) else basis / norm
            lines.append(
                f"{name}: dir=({direction[0]:.2f}, {direction[1]:.2f}, "
                f"{direction[2]:.2f}), length={spacing[idx]:.2f}"
            )
        center = getattr(dataset, "calc_center", None)
        if center is not None:
            lines += [
                "",
                f"Center: {InteractPlane._helper_format_three_vector(center)}",
            ]
        corners = getattr(dataset, "calc_corners", None)
        if corners is not None:
            lines += ["", "Corners:"]
            lines.extend(
                f"{idx}: {InteractPlane._helper_format_three_vector(corner)}"
                for idx, corner in enumerate(np.asarray(corners, dtype=float))
            )
        return lines

    def __init__(self, field, figure):
        self.field = field
        object.__setattr__(field, "state_is_interactable", False)
        object.__setattr__(field.grid, "impl_is_warn_orthogonal", False)
        origin = np.asarray(field.grid.opts.origin, dtype=float).reshape(1, 3)
        normal = np.asarray(field.grid.opts.normal, dtype=float).reshape(1, 3)
        spacing = float(field.grid.opts.spacing)
        size = float(field.grid.opts.size)
        self.visual_normal = PlotRod(
            coords=origin,
            orient=normal,
            radius=spacing / 4,
            color=(1, 0, 0),
            length=size,
            figure=figure,
            name=f"The normal of {field.grid.name!r}",
            category="Interaction",
            is_reset_camera=False,
            is_visible=False,
        )
        self.visual_origin = PlotSphere(
            coords=origin,
            color=(1, 0, 0),
            radius=spacing,
            figure=figure,
            name=f"The origin of {field.grid.name!r}",
            category="Interaction",
            is_reset_camera=False,
            is_visible=False,
        )
        object.__setattr__(self.visual_normal, "state_is_interactable", False)
        object.__setattr__(self.visual_origin, "state_is_interactable", False)
        self._is_continuous_interacting = False
        super().__init__(field.grid, figure, title=f"Controls of {field.grid.name!r}")

    def _show_dataset_grid_spacing_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Grid Axes")
        layout = QtWidgets.QVBoxLayout(dialog)
        for line in self._helper_dataset_grid_spacing_lines(self.field):
            layout.addWidget(QtWidgets.QLabel(line, dialog))
        close = QtWidgets.QPushButton("Close", dialog)
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.show()
        self._dataset_grid_info_dialog = dialog

    def _iter_silhouette_targets(self):
        targets = [
            getattr(self.field, name, None)
            for name in (
                "visual",
                "visual_nb",
                "visual_nd",
                "visual_defect",
                "visual_S",
            )
        ] + [self.visual_normal, self.visual_origin]
        seen = set()
        for visual in targets:
            if visual is None or id(visual) in seen:
                continue
            seen.add(id(visual))
            yield visual

    def _helper_begin_continuous_interaction(self, *_args):
        if self._is_continuous_interacting:
            return
        self._is_continuous_interacting = True
        for visual in self._iter_silhouette_targets():
            if hasattr(visual, "state_is_silhouette"):
                object.__setattr__(visual, "state_is_silhouette", False)
                if hasattr(visual, "_helper_clear_silhouette"):
                    visual._helper_clear_silhouette()

    def _helper_end_continuous_interaction(self, *_args):
        if not self._is_continuous_interacting:
            return
        self._is_continuous_interacting = False
        for visual in self._iter_silhouette_targets():
            if hasattr(visual, "state_is_silhouette"):
                object.__setattr__(visual, "state_is_silhouette", True)
                if getattr(visual, "entity_actor", None) is not None and hasattr(
                    visual, "_helper_add_silhouette"
                ):
                    visual._helper_add_silhouette()

    def _make_vector_input_panel(self, parent, layout, title, values, callback):
        panel = QtWidgets.QWidget(parent)
        row = QtWidgets.QHBoxLayout(panel)
        row.addWidget(QtWidgets.QLabel(f"{title}:", panel))
        boxes = []
        for value in np.asarray(values, dtype=float):
            box = QtWidgets.QDoubleSpinBox(panel)
            box.setDecimals(3)
            box.setKeyboardTracking(False)
            box.setRange(-1e12, 1e12)
            box.setValue(float(value))
            row.addWidget(box)
            boxes.append(box)
        apply = QtWidgets.QPushButton("Apply", panel)
        apply.clicked.connect(lambda: callback(boxes))
        row.addWidget(apply)
        layout.addWidget(panel)
        return boxes

    def _set_vector_inputs(self, boxes, values):
        for box, value in zip(boxes, np.asarray(values, dtype=float), strict=True):
            with QSignalBlocker(box):
                box.setValue(float(value))

    def build_ui(self):
        row = QtWidgets.QWidget(self)
        row_layout = QtWidgets.QHBoxLayout(row)
        self.chk_is_show_axes = QtWidgets.QCheckBox("Show plane origin and normal", row)
        self.btn_dataset_grid_info = QtWidgets.QPushButton("Show grid axes", row)
        self.btn_dataset_grid_info.clicked.connect(
            self._show_dataset_grid_spacing_dialog
        )
        self.chk_is_show_axes.stateChanged.connect(self._on_toggle_show_axes)
        row_layout.addWidget(self.chk_is_show_axes)
        row_layout.addWidget(self.btn_dataset_grid_info)
        self.layout.addWidget(row)

        spacing_extra = self.host.opts.spacing_extra or self.host.opts.spacing
        size_extra = self.host.opts.size_extra or self.host.opts.size
        self.state = {
            "origin": np.asarray(self.host.opts.origin, dtype=float).copy(),
            "origin_move_step": 1.0,
            "is_origin_center": self.host.opts.alignment == "center",
            "spacing": float(self.host.opts.spacing),
            "spacing_extra": float(spacing_extra),
            "size": float(self.host.opts.size),
            "size_extra": float(size_extra),
            "is_use_control_spacing_extra": self.host.opts.spacing_extra is not None,
            "is_use_control_size_extra": self.host.opts.size_extra is not None,
            "normal_azimuth": self.get_azimuth(self.host.opts.normal),
            "normal_polar_angle": self.get_polar_angle(self.host.opts.normal),
            "axis1_azimuth": self.get_axis1_azimuth(
                self.host.opts.axis1, self.host.opts.normal
            ),
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
            center_fmt="{:.3f}",
            is_show_location=False,
            on_move=self._commit_origin,
            on_press=self._helper_begin_continuous_interaction,
            on_hold=None,
            on_release=self._helper_end_continuous_interaction,
            long_press_ms=450,
            repeat_ms=80,
        )
        self.layout.addWidget(self.point_console.group)
        self.sliders["origin_move_step"] = self.point_console.slider_step
        self.origin_inputs = self._make_vector_input_panel(
            self.point_console.group,
            self.point_console.gl,
            "Origin",
            self.host.opts.origin,
            self._commit_manual_origin,
        )

        scalars = QtWidgets.QGroupBox("Physical lengths and alignment", self)
        scalar_layout = QtWidgets.QVBoxLayout(scalars)
        self.layout.addWidget(scalars)
        self.chk_is_origin_center = QtWidgets.QCheckBox(
            "Place origin at plane center", scalars
        )
        self.chk_is_origin_center.setChecked(self.state["is_origin_center"])
        self.chk_is_origin_center.stateChanged.connect(self._on_toggle_is_origin_center)
        scalar_layout.addWidget(self.chk_is_origin_center)
        for key, value in (
            ("spacing", self.state["spacing"]),
            ("spacing_extra", self.state["spacing_extra"]),
            ("size", self.state["size"]),
            ("size_extra", self.state["size_extra"]),
        ):
            mapper = LogTickMapper(0.2 * value, 5.0 * value, base=10.0)
            self.sliders[key] = make_labeled_slider_row(
                parent=scalars,
                layout=scalar_layout,
                name=key,
                value_min=mapper.value_min,
                value_max=mapper.value_max,
                value_init=value,
                tick_to_value=mapper.tick_to_value,
                value_to_tick=mapper.value_to_tick,
                input_out_of_range="expand_max",
            )
        self.chk_use_spacing_extra = QtWidgets.QCheckBox("Use spacing_extra", scalars)
        self.chk_use_spacing_extra.setChecked(
            self.state["is_use_control_spacing_extra"]
        )
        self.chk_use_spacing_extra.stateChanged.connect(
            self._on_toggle_use_spacing_extra
        )
        scalar_layout.addWidget(self.chk_use_spacing_extra)
        self.sliders["spacing_extra"].set_enabled(
            self.state["is_use_control_spacing_extra"]
        )
        self.chk_use_size_extra = QtWidgets.QCheckBox("Use size_extra", scalars)
        self.chk_use_size_extra.setChecked(self.state["is_use_control_size_extra"])
        self.chk_use_size_extra.stateChanged.connect(self._on_toggle_use_size_extra)
        scalar_layout.addWidget(self.chk_use_size_extra)
        self.sliders["size_extra"].set_enabled(self.state["is_use_control_size_extra"])

        orient = QtWidgets.QGroupBox("Physical plane basis", self)
        orient_layout = QtWidgets.QVBoxLayout(orient)
        self.layout.addWidget(orient)
        self.normal_inputs = self._make_vector_input_panel(
            orient,
            orient_layout,
            "Normal",
            self.host.opts.normal,
            self._commit_manual_normal,
        )
        self.sliders["normal_azimuth"] = make_labeled_slider_row(
            parent=orient,
            layout=orient_layout,
            name="Azimuth of normal",
            state_key="normal_azimuth",
            value_min=0,
            value_max=360,
            value_init=self.state["normal_azimuth"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )
        self.sliders["normal_polar_angle"] = make_labeled_slider_row(
            parent=orient,
            layout=orient_layout,
            name="Polar angle of normal",
            state_key="normal_polar_angle",
            value_min=0,
            value_max=180,
            value_init=self.state["normal_polar_angle"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )
        self.axis1_info = QtWidgets.QLabel(
            self._vect_text(self.host.opts.axis1, "axis1"), orient
        )
        orient_layout.addWidget(self.axis1_info)
        self.sliders["axis1_azimuth"] = make_labeled_slider_row(
            parent=orient,
            layout=orient_layout,
            name="Azimuth of axis1",
            state_key="axis1_azimuth",
            value_min=0,
            value_max=360,
            value_init=self.state["axis1_azimuth"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )
        self.on_changed(0, is_commit=False)

    def _commit_origin(self, center):
        self.host.act_commit(origin=np.asarray(center, dtype=float))

    def _commit_manual_origin(self, boxes):
        origin = np.array([box.value() for box in boxes], dtype=float)
        self.state["origin"] = origin.copy()
        self.point_console._update_center_label()
        self._commit_origin(origin)

    def _commit_manual_normal(self, boxes):
        normal = np.array([box.value() for box in boxes], dtype=float)
        norm = np.linalg.norm(normal)
        if np.isclose(norm, 0.0):
            QtWidgets.QMessageBox.warning(
                self, "Invalid normal", "The normal vector must be non-zero."
            )
            return
        normal /= norm
        self._set_vector_inputs(boxes, normal)
        self.state["normal_azimuth"] = self.get_azimuth(normal)
        self.state["normal_polar_angle"] = self.get_polar_angle(normal)
        self.commit()

    def _on_toggle_show_axes(self):
        if self.chk_is_show_axes.isChecked():
            self._update_axes_visuals(True)
        else:
            self.visual_normal.opts.is_visible = False
            self.visual_origin.opts.is_visible = False

    def _update_axes_visuals(self, is_visible=True):
        origin = np.asarray(self.host.opts.origin, dtype=float).reshape(1, 3)
        normal = np.asarray(self.host.opts.normal, dtype=float).reshape(1, 3)
        self.visual_origin.act_commit(
            coords=origin, radius=float(self.host.opts.spacing), is_visible=is_visible
        )
        self.visual_normal.act_commit(
            coords=origin,
            orient=normal,
            radius=float(self.host.opts.spacing) / 4,
            length=float(self.host.opts.size),
            is_visible=is_visible,
        )

    def commit(self):
        normal = vector_from_spherical_angles(
            np.deg2rad(self.state["normal_azimuth"]),
            np.deg2rad(self.state["normal_polar_angle"]),
        )
        rotation = rotation_matrix_from_vectors((0, 0, 1), normal)
        x = rotation @ np.array([1.0, 0.0, 0.0])
        y = rotation @ np.array([0.0, 1.0, 0.0])
        phi = np.deg2rad(self.state["axis1_azimuth"])
        axis1 = np.cos(phi) * x + np.sin(phi) * y
        self.host.act_commit(
            alignment="center" if self.state["is_origin_center"] else "bottom-left",
            spacing=float(self.state["spacing"]),
            spacing_extra=(
                float(self.state["spacing_extra"])
                if self.state["is_use_control_spacing_extra"]
                else None
            ),
            size=float(self.state["size"]),
            size_extra=(
                float(self.state["size_extra"])
                if self.state["is_use_control_size_extra"]
                else None
            ),
            normal=np.asarray(normal, dtype=float),
            axis1=axis1,
        )

    def _on_toggle_is_origin_center(self, _state):
        self.state["is_origin_center"] = self.chk_is_origin_center.isChecked()
        self.commit()

    def _on_toggle_use_spacing_extra(self, _state):
        value = self.chk_use_spacing_extra.isChecked()
        self.state["is_use_control_spacing_extra"] = value
        self.sliders["spacing_extra"].set_enabled(value)
        self.commit()

    def _on_toggle_use_size_extra(self, _state):
        value = self.chk_use_size_extra.isChecked()
        self.state["is_use_control_size_extra"] = value
        self.sliders["size_extra"].set_enabled(value)
        self.commit()

    @staticmethod
    def get_azimuth(vec):
        return np.rad2deg(azimuth_from_vector(vec))

    @staticmethod
    def get_polar_angle(vec):
        return np.rad2deg(polar_angle_from_vector(vec))

    @staticmethod
    def get_axis1_azimuth(axis1, normal):
        return np.rad2deg(plane_azimuth_from_direction(axis1, normal))

    def _sync_func(self, **kwargs):
        if "origin" in kwargs:
            self.state["origin"] = np.asarray(self.host.opts.origin, dtype=float).copy()
            self.point_console._update_center_label()
            self._set_vector_inputs(self.origin_inputs, self.host.opts.origin)
        if "alignment" in kwargs:
            checked = self.host.opts.alignment == "center"
            with QSignalBlocker(self.chk_is_origin_center):
                self.chk_is_origin_center.setChecked(checked)
            self.state["is_origin_center"] = checked
        for key in ("spacing", "size"):
            if key in kwargs:
                self._sync_from_host_slider(key, getattr(self.host.opts, key))
        for key, checkbox, state_key in (
            (
                "spacing_extra",
                self.chk_use_spacing_extra,
                "is_use_control_spacing_extra",
            ),
            ("size_extra", self.chk_use_size_extra, "is_use_control_size_extra"),
        ):
            if key in kwargs:
                value = getattr(self.host.opts, key)
                controlled = value is not None
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(controlled)
                self.state[state_key] = controlled
                self.sliders[key].set_enabled(controlled)
                if controlled:
                    self._sync_from_host_slider(key, value)
        if "normal" in kwargs:
            self._sync_from_host_slider(
                "normal_azimuth", self.get_azimuth(self.host.opts.normal)
            )
            self._sync_from_host_slider(
                "normal_polar_angle", self.get_polar_angle(self.host.opts.normal)
            )
            self._set_vector_inputs(self.normal_inputs, self.host.opts.normal)
        if "axis1" in kwargs or "normal" in kwargs:
            self._sync_from_host_slider(
                "axis1_azimuth",
                self.get_axis1_azimuth(self.host.opts.axis1, self.host.opts.normal),
            )
            self.axis1_info.setText(self._vect_text(self.host.opts.axis1, "axis1"))
        if self.chk_is_show_axes.isChecked():
            self._update_axes_visuals(True)

    def on_close(self):
        self._helper_end_continuous_interaction()
        super().on_close()
        object.__setattr__(self.host, "impl_is_warn_orthogonal", True)
        object.__setattr__(self.field, "state_is_interactable", True)
        self.visual_normal.act_remove()
        self.visual_origin.act_remove()


__all__ = ["InteractPlane"]
