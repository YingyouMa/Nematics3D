import numpy as np
from qtpy import QtWidgets

from .panel_base import PanelBase, make_labeled_slider_row
from ..plot_rod import PlotRod
from Nematics3D.general import rotation_matrix_from_vectors


class InteractPlane(PanelBase):
    def __init__(self, field):
        self.field = field
        object.__setattr__(self.field, "_state_is_interactable", False)

        super().__init__(field.plane, title="Plane grid Controls")

    def build_ui(self):
        # ----------------------------
        # initial state
        # ----------------------------
        self.state = {
            "is_origin_center":                 self.host.opts.alignment == "center",
            "spacing_rescale":                  1.0,
            "spacing_extra_rescale":            1.0,
            "size_rescale":                     1.0,
            "size_extra_rescale":               1.0,
            "is_use_control_spacing_extra":     self.host.opts.spacing_extra is not None,
            "is_use_control_size_extra":        self.host.opts.size_extra is not None,
            "normal_azimuth":                   self.get_azimuth(self.host.opts.normal),
            "normal_polar_angle":               self.get_polar_angle(self.host.opts.normal),
            "axis1_azimuth":                    self.get_axis1_azimuth(self.host.opts.axis1, self.host.opts.normal)
        }

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

        self.sliders["spacing_rescale"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="spacing_rescale",
            tick_min=25,
            tick_max=400,
            tick_init=100,
            tick_to_value=lambda t: t / 100.0,
            value_fmt="{:.4g}",
        )

        self.sliders["spacing_extra_rescale"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="spacing_extra_rescale",
            tick_min=25,
            tick_max=400,
            tick_init=100,
            tick_to_value=lambda t: t / 100.0,
            value_fmt="{:.4g}",
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
        self._apply_spacing_extra_enabled()

        self.sliders["size_rescale"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="size_rescale",
            tick_min=25,
            tick_max=400,
            tick_init=100,
            tick_to_value=lambda t: t / 100.0,
            value_fmt="{:.4g}",
        )

        self.sliders["size_extra_rescale"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="size_extra_rescale",
            tick_min=25,
            tick_max=400,
            tick_init=100,
            tick_to_value=lambda t: t / 100.0,
            value_fmt="{:.4g}",
        )

        self.chk_use_size_extra = QtWidgets.QCheckBox(
            "Use controlled size_extra", group_scalar
        )
        self.chk_use_size_extra.setChecked(self.state["is_use_control_size_extra"])
        gl_scalar.addWidget(self.chk_use_size_extra)
        self.chk_use_size_extra.stateChanged.connect(self._on_toggle_use_size_extra)
        self._apply_size_extra_enabled()
        
        
        # ----------------------------
        # Vector group
        # ----------------------------
        group_vector = QtWidgets.QGroupBox("Vector parameter", self)
        gl_vector = QtWidgets.QVBoxLayout(group_vector)
        self.layout.addWidget(group_vector)
        
        self.sliders["normal_azimuth"] = make_labeled_slider_row(
            parent=group_vector,
            layout=gl_vector,
            name="Azimuth of normal",
            tick_min=0,
            tick_max=360,
            tick_init=int(self.get_azimuth(self.host.opts.normal)),
            tick_to_value=lambda t: t,
            value_fmt="{:.0f}",
        )
        
        self.sliders["normal_polar_angle"] = make_labeled_slider_row(
            parent=group_vector,
            layout=gl_vector,
            name="Polar angle of normal",
            tick_min=0,
            tick_max=180,
            tick_init=int(self.get_polar_angle(self.host.opts.normal)),
            tick_to_value=lambda t: t,
            value_fmt="{:.0f}",
        )
        
        self.sliders["axis1_azimuth"] = make_labeled_slider_row(
            parent=group_vector,
            layout=gl_vector,
            name="Azimuth of axis1",
            tick_min=0,
            tick_max=360,
            tick_init=int(self.get_axis1_azimuth(self.host.opts.axis1, self.host.opts.normal)),
            tick_to_value=lambda t: t,
            value_fmt="{:.0f}",
        )
        
        
        for item in self.sliders.values():
            item.slider.valueChanged.connect(self.on_changed)
            for entity in (
                "_entity_visual_nb",
                "_entity_visual_nd",
                "_entity_visual_defect",
            ):
                entity = getattr(self.field, entity, None)
                if entity:
                    item.slider.sliderPressed.connect(entity._helper_clear_silhouette)
                    item.slider.sliderReleased.connect(entity._helper_add_silhouette)

        self.on_changed(0, is_commit=False)

        self.host.opts._impl_sync_func["alignment"][self.str_now] = (
            lambda: self.chk_is_center_origin.setChecked(
                self.host.opts.alignment == "center"
            )
        )
        self.host.opts._impl_sync_func["normal"][self.str_now] = self._sync_normal
        self.host.opts._impl_sync_func["axis1"][self.str_now] = (
            lambda: self._sync_sides_from_host(
                "axis1_azimuth", self.get_axis1_azimuth(
                    self.host.opts.axis1, 
                    self.host.opts.normal)
                )
            )

    def on_changed(self, _v: int = 0, is_commit: bool = True):
        # ---- spacing_rescale ----
        sr = float(self.sliders["spacing_rescale"].get_value())
        self.sliders["spacing_rescale"].label.setText(f"{sr:.4g}")
        self.state["spacing_rescale"] = sr

        # ---- spacing_extra_rescale ----
        self.state["is_use_control_spacing_extra"] = bool(
            self.chk_use_spacing_extra.isChecked()
        )
        sr = float(self.sliders["spacing_extra_rescale"].get_value())
        self.sliders["spacing_extra_rescale"].label.setText(f"{sr:.4g}")
        self.state["spacing_extra_rescale"] = sr

        # ---- size_extra_rescale ----
        self.state["is_use_control_size_extra"] = bool(
            self.chk_use_size_extra.isChecked()
        )
        sr = float(self.sliders["size_extra_rescale"].get_value())
        self.sliders["size_extra_rescale"].label.setText(f"{sr:.4g}")
        self.state["size_extra_rescale"] = sr

        # ---- size_rescale ----
        sr = float(self.sliders["size_rescale"].get_value())
        self.sliders["size_rescale"].label.setText(f"{sr:.4g}")
        self.state["size_rescale"] = sr
        
        # ---- normal_azimuth ----
        normal_azimuth = float(self.sliders["normal_azimuth"].get_value())
        self.sliders["normal_azimuth"].label.setText(f"{normal_azimuth:g}")
        self.state["normal_azimuth"] = normal_azimuth
        
        # ---- normal_polar_angle ----
        normal_polar_angle = float(self.sliders["normal_polar_angle"].get_value())
        self.sliders["normal_polar_angle"].label.setText(f"{normal_polar_angle:g}")
        self.state["normal_polar_angle"] = normal_polar_angle
        
        # ---- axis1_azimuth ----
        axis1_azimuth = float(self.sliders["axis1_azimuth"].get_value())
        self.sliders["axis1_azimuth"].label.setText(f"{axis1_azimuth:g}")
        self.state["axis1_azimuth"] = axis1_azimuth

        if is_commit:
            self.commit()

    def commit(self):
        # ----alignment ----
        alignment = "center" if self.state["is_origin_center"] else "bottom-left"

        # ---- spacing ----
        current_spacing = self.host._impl_opts_backup[self.str_now]["spacing"]
        scale = float(self.state["spacing_rescale"])
        spacing_now = scale * float(current_spacing)

        # ---- spacing_extra ----
        if bool(self.state.get("is_use_control_spacing_extra", False)):
            scale = float(self.state["spacing_extra_rescale"])
            current_spacing_extra = self.host._impl_opts_backup[self.str_now][
                "spacing_extra"
            ]
            if current_spacing_extra is None:
                current_spacing_extra = self.host._impl_opts_backup[self.str_now][
                    "spacing"
                ]
            spacing_extra_now = scale * float(current_spacing_extra)
        else:
            spacing_extra_now = None

        # ---- size_extra ----
        if bool(self.state.get("is_use_control_size_extra", False)):
            scale = float(self.state["size_extra_rescale"])
            current_size_extra = self.host._impl_opts_backup[self.str_now]["size_extra"]
            if current_size_extra is None:
                current_size_extra = self.host._impl_opts_backup[self.str_now]["size"]
            size_extra_now = scale * float(current_size_extra)
        else:
            size_extra_now = None

        # ---- size ----
        current_size = self.host._impl_opts_backup[self.str_now]["size"]
        scale = float(self.state["size_rescale"])
        size_now = scale * float(current_size)
        
        # ---- normal ----
        normal_azimuth = np.deg2rad(self.state["normal_azimuth"])
        normal_poalr_angle = np.deg2rad(self.state["normal_polar_angle"])
        x = np.sin(normal_poalr_angle) * np.cos(normal_azimuth)
        y = np.sin(normal_poalr_angle) * np.sin(normal_azimuth)
        z = np.cos(normal_poalr_angle)
        normal_now = (x,y,z)
        
        # ---- axis1 ----
        axis1_azimuth = np.deg2rad(self.state["axis1_azimuth"])
        _rotation_matrix = rotation_matrix_from_vectors((0,0,1), normal_now)
        axisx = _rotation_matrix @ np.array([1,0,0])
        axisy = _rotation_matrix @ np.array([0,1,0])
        axis1_now = np.cos(axis1_azimuth) * axisx
        axis1_now += np.sin(axis1_azimuth) * axisy
        

        self.host.act_commit(
            alignment=alignment,
            spacing=spacing_now,
            spacing_extra=spacing_extra_now,
            size_extra=size_extra_now,
            size=size_now,
            normal=normal_now,
            axis1=axis1_now
        )

    def _on_toggle_is_origin_center(self, _state: int):
        self.state["is_origin_center"] = self.chk_is_origin_center.isChecked()
        self.commit()

    def _on_toggle_use_spacing_extra(self, _state: int):
        self.state["is_use_control_spacing_extra"] = (
            self.chk_use_spacing_extra.isChecked()
        )
        self._apply_spacing_extra_enabled()
        self.commit()

    def _apply_spacing_extra_enabled(self):
        spacing_extra_enabled = bool(self.chk_use_spacing_extra.isChecked())
        item = self.sliders["spacing_extra_rescale"]
        item.slider.setEnabled(spacing_extra_enabled)
        item.label.setEnabled(spacing_extra_enabled)

    def _on_toggle_use_size_extra(self, _state: int):
        self.state["is_use_control_size_extra"] = self.chk_use_size_extra.isChecked()
        self._apply_size_extra_enabled()
        self.commit()

    def _apply_size_extra_enabled(self):
        size_extra_enabled = bool(self.chk_use_size_extra.isChecked())
        item = self.sliders["size_extra_rescale"]
        item.slider.setEnabled(size_extra_enabled)
        item.label.setEnabled(size_extra_enabled)

    def on_close(self):
        object.__setattr__(self.field, "_state_is_interactable", True)
        
        
    
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
        _rotation_matrix = rotation_matrix_from_vectors((0,0,1), normal)
        axisx = _rotation_matrix @ np.array([1,0,0])
        axisy = _rotation_matrix @ np.array([0,1,0]) 
        az_rad = np.arctan2(axis1@axisy, axis1@axisx)
        azimuth = np.round(np.degrees(az_rad)) % 360
        return azimuth

    def _sync_normal(self):
        self._sync_sides_from_host("normal_azimuth", self.get_azimuth(self.host.opts.normal))
        self._sync_sides_from_host("normal_polar_angle", self.get_polar_angle(self.host.opts.normal))
