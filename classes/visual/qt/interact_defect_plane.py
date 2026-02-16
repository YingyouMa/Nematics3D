import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from .panel_base import PanelBase, make_labeled_slider_row, LogTickMapper, MovePointConsole
from Nematics3D.general import rotation_matrix_from_vectors
from ..plot_rod import PlotRod
from Nematics3D.datatypes import UNSET


class InteractDefectPlane(PanelBase):
    def __init__(self, field, figure):
        self.field = field
        self.defect_plane = field.plane.owner
        object.__setattr__(self.field, "_state_is_interactable", False)
        
        self.visual_normal = PlotRod(
            coords=field.plane.opts.origin,
            orient=field.plane.opts.normal,
            radius=field.plane.opts.dr/4,
            length=field.plane.opts.R_max*5,
            color=(1,0,0),
            figure=figure,
            name=f"The normal of {field.plane.name!r}",
            category="Interaction",
            is_reset_camera=False,
            is_visible=False
        )

        super().__init__(field.plane, title=f"Controls of {field.plane.name!r}")
        
        
        
    def build_ui(self):
        
        
        def _on_toggle_show_axes():
            checked = self.chk_is_show_axes.isChecked()
            self.visual_normal.opts.is_visible = checked
            
        self.chk_is_show_axes = QtWidgets.QCheckBox(
            "Whether to visualize normal",
            self,
        )
        self.chk_is_show_axes.setChecked(False)
        self.layout.addWidget(self.chk_is_show_axes)
        self.chk_is_show_axes.stateChanged.connect(_on_toggle_show_axes)
        
        arc_dist_init = (
            self.host.opts.arc_dist
            if self.host.opts.arc_dist is not None
            else self.host.opts.arc_dist
        )
        
        # fmt: off
        self.state = {
            "x_param":                              self.defect_plane.x_param,
            "dr":                                   self.host.opts.dr,
            "arc_dist":                             arc_dist_init,
            "R_max":                                self.host.opts.R_max,
            "is_use_control_arc_dist":              self.host.opts.arc_dist is not None,
            "is_use_control_normal":                self.defect_plane.state_normal is not UNSET,
            "normal_azimuth":                       self.get_azimuth(self.host.opts.normal),
            "normal_polar_angle":                   self.get_polar_angle(self.host.opts.normal),
        }
        # fmt: on
        
        # ----------------------------
        # Vector group
        # ----------------------------
        group_vector = QtWidgets.QGroupBox("Placement", self)
        gl_vector = QtWidgets.QVBoxLayout(group_vector)
        self.layout.addWidget(group_vector)
        
        self.origin_info = QtWidgets.QLabel(
            self._vect_text(self.host.opts.origin, "origin"), self
        )
        gl_vector.addWidget(self.origin_info)
        
        self.sliders["x_param"] = make_labeled_slider_row(
            parent=group_vector,
            layout=gl_vector,
            name="x_param",
            value_min=0,
            value_max=100,
            value_init=self.state["x_param"],
            tick_to_value=lambda t: t/100,
            value_to_tick=lambda x: x*100,
        )
        
        
        self.normal_info = QtWidgets.QLabel(
            self._vect_text(self.host.opts.normal, "normal"), self
        )
        gl_vector.addWidget(self.normal_info)
        
        self.sliders["normal_azimuth"] = make_labeled_slider_row(
            parent=group_vector,
            layout=gl_vector,
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
            parent=group_vector,
            layout=gl_vector,
            name="Polar angle of normal",
            state_key="normal_polar_angle",
            value_min=0,
            value_max=180,
            value_init=self.get_polar_angle(self.host.opts.normal),
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.1f}",
        )
        
        self.chk_use_normal = QtWidgets.QCheckBox(
            "Use controlled normal", group_vector
        )
        self.chk_use_normal.setChecked(self.state["is_use_control_normal"])
        gl_vector.addWidget(self.chk_use_normal)
        self.chk_use_normal.stateChanged.connect(self._on_toggle_use_normal)
        self.sliders["normal_azimuth"].set_enabled(self.state["is_use_control_normal"])
        self.sliders["normal_polar_angle"].set_enabled(self.state["is_use_control_normal"])
        

        # ----------------------------
        # Scalar group
        # ----------------------------
        group_scalar = QtWidgets.QGroupBox("Scalar", self)
        gl_scalar = QtWidgets.QVBoxLayout(group_scalar)
        self.layout.addWidget(group_scalar)
        
        log_size = LogTickMapper(
            value_min=0.2*self.state["dr"],
            value_max=5*self.state["dr"],
            base=10.0,
        )
        
        self.sliders["dr"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="dr",
            value_min=log_size.value_min,
            value_max=log_size.value_max,
            value_init=self.state["dr"],
            tick_to_value=log_size.tick_to_value,
            value_to_tick=log_size.value_to_tick,
        )
        
        
        
        
        
        
        for key, item in self.sliders.items():
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
                    
        self.on_changed(0, is_commit=False)
        
        # self.defect_plane._impl_sync_func["x_param"][self.str_now] = self._sync_origin
        # self.defect_plane._impl_sync_func["state_normal"][self.str_now] = self._sync_normal
        
    def commit(self):
        # ---- normal ----
        if self.state["is_use_control_normal"]:
            normal_azimuth = np.deg2rad(self.state["normal_azimuth"])
            normal_poalr_angle = np.deg2rad(self.state["normal_polar_angle"])
            x = np.sin(normal_poalr_angle) * np.cos(normal_azimuth)
            y = np.sin(normal_poalr_angle) * np.sin(normal_azimuth)
            z = np.cos(normal_poalr_angle)
            normal_now = (x, y, z)
        else:
            normal_now = UNSET
        
        
        
        self.defect_plane.act_commit(
            x_param=self.state['x_param'],
            dr=self.state["dr"],
            state_normal=normal_now
        )
        
        
        
    def _on_toggle_use_normal(self, _state: int):
        result = self.chk_use_normal.isChecked()
        self.state["is_use_control_normal"] = result
        self.sliders["normal_azimuth"].set_enabled(result)
        self.sliders["normal_polar_angle"].set_enabled(result)
        self.commit()
        
      
    def _sync_origin(self):
        self._sync_from_host("x_param", self.defect_plane.x_param)
        self.origin_info.setText(
            self._vect_text(self.host.opts.origin, "origin")
        )
        # self.visual_normal.act_commit(
        #     coords=self.host.opts.origin, 
        #     orient=self.host.opts.normal,
        #     is_silhouette=False
        # )
        
    def _sync_normal(self):
        if self.defect_plane.state_normal is UNSET:
            self.chk_use_normal.setChecked(False)
        else: #!!! other const normals
            self.chk_use_normal.setChecked(True)
            self._sync_from_host("normal_azimuth", self.get_azimuth(self.host.opts.normal))
            self._sync_from_host(
                "normal_polar_angle", self.get_polar_angle(self.host.opts.normal)
            )
        self.normal_info.setText(
            self._vect_text(self.host.opts.normal, "normal")
        )
        # self.visual_normal.act_commit(
        #     orient=self.host.opts.normal,
        #     is_silhouette=False
        # )       

        
    def on_close(self):
        super().on_close()
        object.__setattr__(self.field, "_state_is_interactable", True)
        self.visual_normal.act_remove()
        sync = getattr(self.defect_plane, "_impl_sync_func", None)
        for k, sub in sync.items():
            sub.pop(self.str_now, None)
        
        
        
        
        
        
        
        
        
        
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
    def _vect_text(vect, name):
        text = f"{name}: ({vect[0]:.2f}, {vect[1]:.2f}, {vect[2]:.2f})"
        return text