import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from .panel_base import PanelBase, make_labeled_slider_row, LogTickMapper, MovePointConsole
from Nematics3D.general import rotation_matrix_from_vectors
from ..plot_rod import PlotRod
from ..plot_sphere import PlotSphere


class InteractPlane(PanelBase):
    def __init__(self, field, figure):
        self.field = field
        object.__setattr__(self.field, "_state_is_interactable", False)
        
        self.visual_normal = PlotRod(
            coords=field.plane.opts.origin,
            orient=field.plane.opts.normal,
            radius=field.plane.opts.spacing/4,
            color=(1,0,0),
            length=field.plane.opts.size,
            figure=figure,
            name=f"The normal of {field.plane.name!r}",
            category="Interaction",
            is_reset_camera=False,
            is_visible=False
        )
        
        self.visual_origin = PlotSphere(
            coords=field.plane.opts.origin,
            color=(1,0,0),
            radius=field.plane.opts.spacing,
            figure=figure,
            name=f"The origin of {field.plane.name!r}",
            category="Interaction",
            is_reset_camera=False,
            is_visible=False
        )

        super().__init__(field.plane, title=f"Controls of {field.plane.name!r}")

    def build_ui(self):
        
        
        def _on_toggle_show_axes():
            checked = self.chk_is_show_axes.isChecked()
            self.visual_normal.opts.is_visible = checked
            self.visual_origin.opts.is_visible = checked
            
        self.chk_is_show_axes = QtWidgets.QCheckBox(
            "Whether to visualize normal and origin",
            self,
        )
        self.chk_is_show_axes.setChecked(False)
        self.layout.addWidget(self.chk_is_show_axes)
        self.chk_is_show_axes.stateChanged.connect(_on_toggle_show_axes)
        

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

        # fmt: off
        self.state = {
            "origin":                               self.host.opts.origin,
            "origin_move_step":                     1.0,
            "is_origin_center":                     self.host.opts.alignment == "center",
            "spacing":                              self.host.opts.spacing,
            "spacing_extra":                        spacing_extra_init,
            "size":                                 self.host.opts.size,
            "size_extra":                           size_extra_init,
            "is_use_control_spacing_extra":         self.host.opts.spacing_extra is not None,
            "is_use_control_size_extra":            self.host.opts.size_extra is not None,
            "normal_azimuth":                       self.get_azimuth(self.host.opts.normal),
            "normal_polar_angle":                   self.get_polar_angle(self.host.opts.normal),
            "axis1_azimuth":                        axis1_azimuth_init,
        }
        # fmt: on
        
        # ----------------------------
        # Origin group
        # ----------------------------
        
        def _origin_commit(center):
            self.host.act_commit(origin=center)
            self.visual_normal.act_commit(coords=[center], is_silhouette=False)
            self.visual_origin.act_commit(coords=[center], is_silhouette=False)
        
        def _origin_on_press(_x, _y):
            for entity in (
                "_entity_visual_nb",
                "_entity_visual_nd",
                "_entity_visual_defect",
            ):
                entity = getattr(self.field, entity, None)
                if entity:
                    entity._helper_clear_silhouette()
            self.visual_normal._helper_clear_silhouette()
            self.visual_origin._helper_clear_silhouette()
                    
        def _origin_on_release(_x, _y):
            for entity in (
                "_entity_visual_nb",
                "_entity_visual_nd",
                "_entity_visual_defect",
            ):
                entity = getattr(self.field, entity, None)
                if entity:
                    entity._helper_add_silhouette()
            self.visual_normal._helper_add_silhouette()
            self.visual_origin._helper_add_silhouette()
                    
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
            on_move=_origin_commit,
            on_press=_origin_on_press,
            on_hold=None,
            on_release=_origin_on_release,
            long_press_ms=450,
            repeat_ms=80,
        )
        self.layout.addWidget(self.point_console.group)
        self.sliders["origin_move_step"] = self.point_console.slider_step
        

        # ----------------------------
        # Scalar group
        # ----------------------------
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

        # ----------------------------
        # Orient group
        # ----------------------------
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

        for key, item in self.sliders.items():
            if key != "origin_move_step":
                item.slider.valueChanged.connect(self.on_changed)
                item.slider.sliderPressed.connect(self.visual_normal._helper_clear_silhouette)
                item.slider.sliderReleased.connect(self.visual_normal._helper_add_silhouette)
                for entity in (
                    "_entity_visual_nb",
                    "_entity_visual_nd",
                    "_entity_visual_defect",
                ):
                    entity = getattr(self.field, entity, None)
                    if entity:
                        item.slider.sliderPressed.connect(entity._helper_clear_silhouette)
                        item.slider.sliderReleased.connect(entity._helper_add_silhouette)
            else:
                item.slider.valueChanged.connect(lambda: self.on_changed(is_commit=False))

        self.on_changed(0, is_commit=False)

        self.host.opts._impl_sync_func["origin"][self.str_now] = self._sync_origin
        self.host.opts._impl_sync_func["alignment"][self.str_now] = (
            lambda: self.chk_is_center_origin.setChecked(
                self.host.opts.alignment == "center"
            )
        )
        self.host.opts._impl_sync_func["normal"][self.str_now] = self._sync_normal
        self.host.opts._impl_sync_func["axis1"][self.str_now] = self._sync_axis1
        self.host.opts._impl_sync_func["spacing"][self.str_now] = (
            lambda: self._sync_from_host("spacing", self.host.opts.spacing)
        )
        self.host.opts._impl_sync_func["size"][self.str_now] = (
            lambda: self._sync_from_host("size", self.host.opts.size)
        )
        self.host.opts._impl_sync_func["spacing_extra"][
            self.str_now
        ] = self._sync_spacing_extra
        self.host.opts._impl_sync_func["size_extra"][
            self.str_now
        ] = self._sync_size_extra


    def commit(self):
        # ----alignment ----
        alignment = "center" if self.state["is_origin_center"] else "bottom-left"

        # ---- spacing_extra ----
        if bool(self.state.get("is_use_control_spacing_extra", False)):
            spacing_extra_now = float(self.state["spacing_extra"])
        else:
            spacing_extra_now = None

        # ---- size_extra ----
        if bool(self.state.get("is_use_control_size_extra", False)):
            size_extra_now = self.state["size_extra"]
        else:
            size_extra_now = None

        # ---- normal ----
        normal_azimuth = np.deg2rad(self.state["normal_azimuth"])
        normal_poalr_angle = np.deg2rad(self.state["normal_polar_angle"])
        x = np.sin(normal_poalr_angle) * np.cos(normal_azimuth)
        y = np.sin(normal_poalr_angle) * np.sin(normal_azimuth)
        z = np.cos(normal_poalr_angle)
        normal_now = (x, y, z)
        self.normal_info.setText(self._vect_text(normal_now, "normal"))
        self.visual_normal.act_commit(orient=normal_now, is_silhouette=False)

        # ---- axis1 ----
        axis1_azimuth = np.deg2rad(self.state["axis1_azimuth"])
        _rotation_matrix = rotation_matrix_from_vectors((0, 0, 1), normal_now)
        axisx = _rotation_matrix @ np.array([1, 0, 0])
        axisy = _rotation_matrix @ np.array([0, 1, 0])
        axis1_now = np.cos(axis1_azimuth) * axisx
        axis1_now += np.sin(axis1_azimuth) * axisy
        self.axis1_info.setText(self._vect_text(axis1_now, "axis1"))

        self.host.act_commit(
            alignment=alignment,
            spacing=self.state["spacing"],
            spacing_extra=spacing_extra_now,
            size_extra=size_extra_now,
            size=self.state["size"],
            normal=normal_now,
            axis1=axis1_now,
        )

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
        az_rad = np.arctan2(vec[1], vec[0])
        azimuth = np.round(np.degrees(az_rad)) % 360
        return azimuth

    @staticmethod
    def get_polar_angle(vec):
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
        polar = np.arccos(vec[2])
        polar = np.degrees(polar)
        return polar

    @staticmethod
    def get_axis1_azimuth(axis1, normal):
        axis1 /= np.linalg.norm(axis1, axis=-1, keepdims=True)
        _rotation_matrix = rotation_matrix_from_vectors((0, 0, 1), normal)
        axisx = _rotation_matrix @ np.array([1, 0, 0])
        axisy = _rotation_matrix @ np.array([0, 1, 0])
        az_rad = np.arctan2(axis1 @ axisy, axis1 @ axisx)
        azimuth = np.round(np.degrees(az_rad)) % 360
        return azimuth

    def _sync_spacing_extra(self):
        result = self.host.opts.spacing_extra is not None
        with QSignalBlocker(self.chk_use_spacing_extra):
            self.chk_use_spacing_extra.setChecked(result)
        self.state["is_use_control_spacing_extra"] = result
        self.sliders["spacing_extra"].set_enabled(result)
        if result:
            self._sync_from_host("spacing_extra", self.host.opts.spacing_extra)

    def _sync_size_extra(self):
        result = self.host.opts.size_extra is not None
        with QSignalBlocker(self.chk_use_size_extra):
            self.chk_use_size_extra.setChecked(result)
        self.state["is_use_control_size_extra"] = result
        self.sliders["size_extra"].set_enabled(result)
        if result:
            self._sync_from_host("size_extra", self.host.opts.size_extra)

    def _sync_normal(self):
        self._sync_from_host("normal_azimuth", self.get_azimuth(self.host.opts.normal))
        self._sync_from_host(
            "normal_polar_angle", self.get_polar_angle(self.host.opts.normal)
        )
        self.normal_info.setText(
            self._vect_text(self.host.opts.normal, "normal")
        )
        self.visual_normal.orient = self.host.opts.normal
        
    def _sync_axis1(self):
        self._sync_from_host(
            "axis1_azimuth",
            self.get_axis1_azimuth(self.host.opts.axis1, self.host.opts.normal)
        )
        self.axis1_info.setText(
            self._vect_text(self.host.opts.axis1, "axis1"), self
        )
        
    def _sync_origin(self):
        self.state["origin"] = self.host.opts.origin
        self.point_console._update_center_label()
        self.visual_normal.coords=[self.host.opts.origin]
        self.visual_origin.coords=[self.host.opts.origin]
        

    @staticmethod
    def _vect_text(vect, name):
        text = f"{name}: ({vect[0]:.2f}, {vect[1]:.2f}, {vect[2]:.2f})"
        return text
    
    def on_close(self):
        super().on_close()
        object.__setattr__(self.field, "_state_is_interactable", True)
        self.visual_normal.act_remove()
        self.visual_origin.act_remove()
