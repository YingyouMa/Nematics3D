import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from .panel_base import (
    PanelBase,
    make_labeled_slider_row,
    LogTickMapper,
    MovePointConsole,
)
from nematics3d.general import rotation_matrix_from_vectors
from nematics3d.geometry import (
    calc_vec_from_azimuth_polar,
    get_azimuth as geometry_get_azimuth,
    get_polar_angle as geometry_get_polar_angle,
    get_axis1_azimuth as geometry_get_axis1_azimuth,
)
from ..plot_rod import PlotRod
from ..plot_sphere import PlotSphere


class InteractPlane(PanelBase):
    # ==================== OVERRIDE ====================
    # InteractPlane overrides PanelBase.__init__ because this panel
    # controls a PlaneGrid host while also managing extra helper visuals
    # and the interactable state of the parent field object.
    # ==================================================
    def __init__(self, field, figure):
        self.field = field
        object.__setattr__(self.field, "state_is_interactable", False)

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
        object.__setattr__(self, "_is_continuous_interacting", False)
        super().__init__(field.grid, figure, title=f"Controls of {field.grid.name!r}")

    def _iter_silhouette_targets(self):
        targets = []
        for name in (
            "visual_nb",
            "visual_nd",
            "visual_defect",
            "visual_S",
        ):
            visual = getattr(self.field, name, None)
            if visual is not None:
                targets.append(visual)
        targets.extend([self.visual_normal, self.visual_origin])

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
        for visual in self._iter_silhouette_targets():
            if not hasattr(visual, "state_is_silhouette"):
                continue
            object.__setattr__(visual, "state_is_silhouette", False)
            if hasattr(visual, "_helper_clear_silhouette"):
                visual._helper_clear_silhouette()

    def _helper_end_continuous_interaction(self, *_args):
        if not self._is_continuous_interacting:
            return
        object.__setattr__(self, "_is_continuous_interacting", False)
        for visual in self._iter_silhouette_targets():
            if not hasattr(visual, "state_is_silhouette"):
                continue
            object.__setattr__(visual, "state_is_silhouette", True)
            if getattr(visual, "entity_actor", None) is not None and hasattr(
                visual, "_helper_add_silhouette"
            ):
                visual._helper_add_silhouette()

    def build_ui(self):
        self.chk_is_show_axes = QtWidgets.QCheckBox(
            "Whether to visualize normal and origin",
            self,
        )
        self.chk_is_show_axes.setChecked(False)
        self.layout.addWidget(self.chk_is_show_axes)
        self.chk_is_show_axes.stateChanged.connect(self._on_toggle_show_axes)

        spacing_extra_init = (
            self.host.opts.spacing_extra
            if self.host.opts.spacing_extra is not None
            else self.host.opts.spacing
        )
        size_extra_init = (
            self.host.opts.size_extra
            if self.host.opts.size_extra is not None
            else self.host.opts.size
        )
        axis1_azimuth_init = self.get_axis1_azimuth(
            self.host.opts.axis1, self.host.opts.normal
        )

        self.state = {
            "origin": np.asarray(self.host.opts.origin, dtype=float).copy(),
            "origin_move_step": 1.0,
            "is_origin_center": self.host.opts.alignment == "center",
            "spacing": float(self.host.opts.spacing),
            "spacing_extra": float(spacing_extra_init),
            "size": float(self.host.opts.size),
            "size_extra": float(size_extra_init),
            "is_use_control_spacing_extra": self.host.opts.spacing_extra is not None,
            "is_use_control_size_extra": self.host.opts.size_extra is not None,
            "normal_azimuth": self.get_azimuth(self.host.opts.normal),
            "normal_polar_angle": self.get_polar_angle(self.host.opts.normal),
            "axis1_azimuth": axis1_azimuth_init,
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
            on_press=self._helper_begin_continuous_interaction,
            on_hold=None,
            on_release=self._helper_end_continuous_interaction,
            long_press_ms=450,
            repeat_ms=80,
        )
        self.layout.addWidget(self.point_console.group)
        self.sliders["origin_move_step"] = self.point_console.slider_step

        group_scalar = QtWidgets.QGroupBox("Scalar parameter", self)
        gl_scalar = QtWidgets.QVBoxLayout(group_scalar)
        self.layout.addWidget(group_scalar)

        self.chk_is_origin_center = QtWidgets.QCheckBox(
            "Whether to set origin at center (if not, set it at bottom-left)",
            group_scalar,
        )
        self.chk_is_origin_center.setChecked(self.state["is_origin_center"])
        gl_scalar.addWidget(self.chk_is_origin_center)
        self.chk_is_origin_center.stateChanged.connect(self._on_toggle_is_origin_center)

        log_spacing = LogTickMapper(
            value_min=0.2 * self.state["spacing"],
            value_max=5 * self.state["spacing"],
            base=10.0,
        )
        self.sliders["spacing"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="spacing",
            value_min=log_spacing.value_min,
            value_max=log_spacing.value_max,
            value_init=self.state["spacing"],
            tick_to_value=log_spacing.tick_to_value,
            value_to_tick=log_spacing.value_to_tick,
        )

        log_spacing_extra = LogTickMapper(
            value_min=0.2 * self.state["spacing_extra"],
            value_max=5 * self.state["spacing_extra"],
            base=10.0,
        )
        self.sliders["spacing_extra"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="spacing_extra",
            value_min=log_spacing_extra.value_min,
            value_max=log_spacing_extra.value_max,
            value_init=self.state["spacing_extra"],
            tick_to_value=log_spacing_extra.tick_to_value,
            value_to_tick=log_spacing_extra.value_to_tick,
        )

        self.chk_use_spacing_extra = QtWidgets.QCheckBox(
            "Use controlled spacing_extra", group_scalar
        )
        self.chk_use_spacing_extra.setChecked(
            self.state["is_use_control_spacing_extra"]
        )
        gl_scalar.addWidget(self.chk_use_spacing_extra)
        self.chk_use_spacing_extra.stateChanged.connect(
            self._on_toggle_use_spacing_extra
        )
        self.sliders["spacing_extra"].set_enabled(
            self.state["is_use_control_spacing_extra"]
        )

        log_size = LogTickMapper(
            value_min=0.2 * self.state["size"],
            value_max=5 * self.state["size"],
            base=10.0,
        )
        self.sliders["size"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="size",
            value_min=log_size.value_min,
            value_max=log_size.value_max,
            value_init=self.state["size"],
            tick_to_value=log_size.tick_to_value,
            value_to_tick=log_size.value_to_tick,
        )

        log_size_extra = LogTickMapper(
            value_min=0.2 * self.state["size_extra"],
            value_max=5 * self.state["size_extra"],
            base=10.0,
        )
        self.sliders["size_extra"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="size_extra",
            value_min=log_size_extra.value_min,
            value_max=log_size_extra.value_max,
            value_init=self.state["size_extra"],
            tick_to_value=log_size_extra.tick_to_value,
            value_to_tick=log_size_extra.value_to_tick,
        )

        self.chk_use_size_extra = QtWidgets.QCheckBox(
            "Use controlled size_extra", group_scalar
        )
        self.chk_use_size_extra.setChecked(self.state["is_use_control_size_extra"])
        gl_scalar.addWidget(self.chk_use_size_extra)
        self.chk_use_size_extra.stateChanged.connect(self._on_toggle_use_size_extra)
        self.sliders["size_extra"].set_enabled(self.state["is_use_control_size_extra"])

        group_orient = QtWidgets.QGroupBox("Vector parameter", self)
        gl_orient = QtWidgets.QVBoxLayout(group_orient)
        self.layout.addWidget(group_orient)

        self.normal_info = QtWidgets.QLabel(
            self._vect_text(self.host.opts.normal, "normal"), self
        )
        gl_orient.addWidget(self.normal_info)

        self.sliders["normal_azimuth"] = make_labeled_slider_row(
            parent=group_orient,
            layout=gl_orient,
            name="Azimuth of normal",
            state_key="normal_azimuth",
            value_min=0,
            value_max=360,
            value_init=self.get_azimuth(self.host.opts.normal),
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )

        self.sliders["normal_polar_angle"] = make_labeled_slider_row(
            parent=group_orient,
            layout=gl_orient,
            name="Polar angle of normal",
            state_key="normal_polar_angle",
            value_min=0,
            value_max=180,
            value_init=self.get_polar_angle(self.host.opts.normal),
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )

        self.axis1_info = QtWidgets.QLabel(
            self._vect_text(self.host.opts.axis1, "axis1"), self
        )
        gl_orient.addWidget(self.axis1_info)

        self.sliders["axis1_azimuth"] = make_labeled_slider_row(
            parent=group_orient,
            layout=gl_orient,
            name="Azimuth of axis1",
            state_key="axis1_azimuth",
            value_min=0,
            value_max=360,
            value_init=self.get_axis1_azimuth(
                self.host.opts.axis1, self.host.opts.normal
            ),
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )

        self.on_changed(0, is_commit=False)

    def _commit_origin(self, center):
        self._is_gui_updating = True
        try:
            self.host.act_commit(origin=np.asarray(center, dtype=float))
        finally:
            self._is_gui_updating = False

    def _on_toggle_show_axes(self):
        checked = self.chk_is_show_axes.isChecked()
        if checked:
            self._update_axes_visuals(is_visible=True)
        else:
            self.visual_normal.opts.is_visible = False
            self.visual_origin.opts.is_visible = False

    def _update_axes_visuals(self, is_visible=True):
        origin = np.asarray(self.host.opts.origin, dtype=float).reshape(1, 3)
        normal = np.asarray(self.host.opts.normal, dtype=float).reshape(1, 3)
        self.visual_origin.act_commit(
            coords=origin,
            radius=float(self.host.opts.spacing),
            is_visible=is_visible,
        )
        self.visual_normal.act_commit(
            coords=origin,
            orient=normal,
            radius=float(self.host.opts.spacing) / 4,
            length=float(self.host.opts.size),
            is_visible=is_visible,
        )

    def commit(self):
        alignment = "center" if self.state["is_origin_center"] else "bottom-left"
        spacing_extra_now = (
            float(self.state["spacing_extra"])
            if bool(self.state.get("is_use_control_spacing_extra", False))
            else None
        )
        size_extra_now = (
            float(self.state["size_extra"])
            if bool(self.state.get("is_use_control_size_extra", False))
            else None
        )

        normal_azimuth = np.deg2rad(self.state["normal_azimuth"])
        normal_polar_angle = np.deg2rad(self.state["normal_polar_angle"])
        normal_now = np.asarray(
            calc_vec_from_azimuth_polar(normal_azimuth, normal_polar_angle), dtype=float
        )

        axis1_azimuth = np.deg2rad(self.state["axis1_azimuth"])
        rotation = rotation_matrix_from_vectors((0, 0, 1), normal_now)
        axisx = rotation @ np.array([1.0, 0.0, 0.0])
        axisy = rotation @ np.array([0.0, 1.0, 0.0])
        axis1_now = np.cos(axis1_azimuth) * axisx + np.sin(axis1_azimuth) * axisy

        self._is_gui_updating = True
        try:
            self.host.act_commit(
                alignment=alignment,
                spacing=float(self.state["spacing"]),
                spacing_extra=spacing_extra_now,
                size=float(self.state["size"]),
                size_extra=size_extra_now,
                normal=normal_now,
                axis1=axis1_now,
            )
        finally:
            self._is_gui_updating = False

    def _on_toggle_is_origin_center(self, _state: int):
        self.state["is_origin_center"] = self.chk_is_origin_center.isChecked()
        self.commit()

    def _on_toggle_use_spacing_extra(self, _state: int):
        result = self.chk_use_spacing_extra.isChecked()
        self.state["is_use_control_spacing_extra"] = result
        self.sliders["spacing_extra"].set_enabled(result)
        self.commit()

    def _on_toggle_use_size_extra(self, _state: int):
        result = self.chk_use_size_extra.isChecked()
        self.state["is_use_control_size_extra"] = result
        self.sliders["size_extra"].set_enabled(result)
        self.commit()

    @staticmethod
    def get_azimuth(vec):
        return geometry_get_azimuth(vec)

    @staticmethod
    def get_polar_angle(vec):
        return geometry_get_polar_angle(vec)

    @staticmethod
    def get_axis1_azimuth(axis1, normal):
        return geometry_get_axis1_azimuth(axis1, normal)

    # InteractPlane overrides PanelBase._sync_func because the plane
    # panel must keep multiple coupled widgets and helper visuals in sync
    # with PlaneGrid option changes from the host side.
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
            if "spacing" in kwargs:
                self._sync_from_host_slider("spacing", self.host.opts.spacing)
            if "size" in kwargs:
                self._sync_from_host_slider("size", self.host.opts.size)
            if "spacing_extra" in kwargs:
                result = self.host.opts.spacing_extra is not None
                with QSignalBlocker(self.chk_use_spacing_extra):
                    self.chk_use_spacing_extra.setChecked(result)
                self.state["is_use_control_spacing_extra"] = result
                self.sliders["spacing_extra"].set_enabled(result)
                if result:
                    self._sync_from_host_slider(
                        "spacing_extra", self.host.opts.spacing_extra
                    )
            if "size_extra" in kwargs:
                result = self.host.opts.size_extra is not None
                with QSignalBlocker(self.chk_use_size_extra):
                    self.chk_use_size_extra.setChecked(result)
                self.state["is_use_control_size_extra"] = result
                self.sliders["size_extra"].set_enabled(result)
                if result:
                    self._sync_from_host_slider("size_extra", self.host.opts.size_extra)
            if "normal" in kwargs:
                self._sync_from_host_slider(
                    "normal_azimuth", self.get_azimuth(self.host.opts.normal)
                )
                self._sync_from_host_slider(
                    "normal_polar_angle", self.get_polar_angle(self.host.opts.normal)
                )
                self.normal_info.setText(
                    self._vect_text(self.host.opts.normal, "normal")
                )
            if "axis1" in kwargs or "normal" in kwargs:
                self._sync_from_host_slider(
                    "axis1_azimuth",
                    self.get_axis1_azimuth(self.host.opts.axis1, self.host.opts.normal),
                )

        if "normal" in kwargs:
            self.normal_info.setText(self._vect_text(self.host.opts.normal, "normal"))
        if "axis1" in kwargs or "normal" in kwargs:
            self.axis1_info.setText(self._vect_text(self.host.opts.axis1, "axis1"))

        if self.chk_is_show_axes.isChecked():
            self._update_axes_visuals(is_visible=True)

    # ==================== OVERRIDE ====================
    # InteractPlane overrides PanelBase.on_close because it must restore
    # the field interactable state and remove helper visuals created only
    # for this panel session.
    # ==================================================
    def on_close(self):
        self._helper_end_continuous_interaction()
        super().on_close()
        object.__setattr__(self.field, "state_is_interactable", True)
        self.visual_normal.act_remove()
        self.visual_origin.act_remove()
