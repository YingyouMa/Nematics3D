from qtpy import QtWidgets
import numpy as np

from .panel_base import PanelBase, make_labeled_slider_row, make_RGB_slider, LogTickMapper
from Nematics3D.datatypes import boundary_periodic_size_to_flag
from ..plot_sphere import PlotSphere

class InteractDisclinationLine(PanelBase):
    
    def __init__(self, obj):
        self.obj = obj
        self.owner = obj.owner
        object.__setattr__(self.owner, "_state_is_silhouette", False)
        super().__init__(self.obj._entity, title="Smoothed disclination line control")
        self.owner.act_save_opts(name=self.str_now)
        
        self.spheres = PlotSphere(
            self._helper_create_sphere_coords(self.obj.state_is_wrap),
            figure=self.host.fig,
            name="raw defect points",
            color=(0,0,0),
            is_reset_camera=False
            )
        
    def _helper_create_sphere_coords(self, is_wrap):
        if is_wrap:
            boundary_flag = boundary_periodic_size_to_flag(
                self.owner.owner._raw_box_size_periodic_index
            )
            coords = np.where(
                boundary_flag,
                self.owner.owner._calc_defect_coords % self.owner.owner._raw_box_size_periodic_index,
                self.owner.owner._calc_defect_coords,
            )
        else:
            coords = self.owner.owner._calc_defect_coords
        return coords
        
        
    def build_ui(self):
        # ----------------------------
        # initial state
        # ----------------------------
        self.state = {
            "window_length":            None,
            "is_smooth":                None,
            "radius_rescale":           1.0,
            "sides":                    None,
            "is_wrap":                  None,
            "is_use_control_color":     False,
            "is_use_control_opacity":   False,
            "color":                    None,
            "opacity":                  None,
            }
        
        # ----------------------------
        # Smooth group
        # ----------------------------
        group_smooth = QtWidgets.QGroupBox("Smooth", self)
        gl_smooth = QtWidgets.QVBoxLayout(group_smooth)
        self.layout.addWidget(group_smooth)
        
        self.sliders["window_length"] = make_labeled_slider_row(
            parent=group_smooth,
            layout=gl_smooth,
            name="window_length",
            tick_min=5,
            tick_max=np.min([100, self.owner.owner._calc_defect_num-1]),
            tick_init=int(self.owner.opts.window_length),   
            tick_to_value=lambda t: int(t),
            value_fmt="{:.0f}",
        )
        
        self.chk_is_smooth = QtWidgets.QCheckBox("Use smoothed coordinates", group_smooth)
        self.chk_is_smooth.setChecked(bool(self.obj.state_is_smooth))
        gl_smooth.addWidget(self.chk_is_smooth)
        self.chk_is_smooth.stateChanged.connect(self._on_toggle_is_smooth)
        self.sliders["window_length"].set_enabled(self.chk_is_smooth.isChecked())

        # ----------------------------
        # Geometry group
        # ----------------------------
        group_geometry = QtWidgets.QGroupBox("Geometry", self)
        gl_geometry = QtWidgets.QVBoxLayout(group_geometry)
        self.layout.addWidget(group_geometry)
        
        self.chk_is_wrap = QtWidgets.QCheckBox("Use wrapped coordinates", group_geometry)
        self.chk_is_wrap.setChecked(bool(self.obj.state_is_wrap))
        gl_geometry.addWidget(self.chk_is_wrap)
        self.chk_is_wrap.stateChanged.connect(self._on_toggle_is_wrap)

        log_mapper = LogTickMapper(
            value_min=0.2,
            value_max=5,
            base=10.0,
        )
        
        self.sliders["radius_rescale"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="radius_rescale",
            state_key="radius_rescale",
            tick_min=log_mapper.tick_min,
            tick_max=log_mapper.tick_max,
            tick_init=log_mapper.value_to_tick(1.0),
            tick_to_value=log_mapper.tick_to_value,
        )

        self.sliders["sides"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="sides",
            state_key="sides",
            tick_min=4,
            tick_max=40,
            tick_init=int(self.host.opts.sides),
            tick_to_value=lambda t: float(int(t)),
            value_fmt="{:.0f}",
        )
        
        # ----------------------------
        # RGB group
        # ----------------------------
        
        group_RGB = QtWidgets.QGroupBox("Color (RGB 0..1)", self)
        gl_RGB = QtWidgets.QVBoxLayout(group_RGB)
        self.layout.addWidget(group_RGB)
        
        make_RGB_slider(
            parent=group_RGB,
            layout=gl_RGB,
            sliders=self.sliders,
            prefix="color",
            init_rgb=self.host._calc_color[0],
        )
        
        self.chk_use_color = QtWidgets.QCheckBox("Use controlled color", group_RGB)
        self.chk_use_color.setChecked(self.state["is_use_control_color"])
        gl_RGB.addWidget(self.chk_use_color)
        self.chk_use_color.stateChanged.connect(self._on_toggle_use_color)
        for k in ("color_r", "color_g", "color_b"):
            self.sliders[k].set_enabled(self.chk_use_color.isChecked())
        
        # ----------------------------
        # Opacity group
        # ----------------------------
        
        group_opacity = QtWidgets.QGroupBox("Opacity (0..1)", self)
        gl_opacity = QtWidgets.QVBoxLayout(group_opacity)
        self.layout.addWidget(group_opacity)
        
        self.sliders["opacity"] = make_labeled_slider_row(
            parent=group_opacity,
            layout=gl_opacity,
            name="opacity",
            state_key="opacity",
            tick_min=0,
            tick_max=100,
            tick_init=int(self.host._calc_opacity[0] * 100),
            tick_to_value=lambda t: float(t / 100.0),
            value_fmt="{:.2f}",
        )
        
        self.chk_use_opacity = QtWidgets.QCheckBox("Use controlled opacity", group_opacity)
        self.chk_use_opacity.setChecked(self.state["is_use_control_opacity"])
        gl_opacity.addWidget(self.chk_use_opacity)
        self.chk_use_opacity.stateChanged.connect(self._on_toggle_use_opacity)
        self.sliders["opacity"].set_enabled(self.chk_use_opacity.isChecked())


        for key, item in self.sliders.items():
            if key == "window_length":
                item.slider.valueChanged.connect(lambda: self.on_changed(is_only_smooth=True))
            else:
                item.slider.valueChanged.connect(self.on_changed)
            item.slider.sliderPressed.connect(self.host._helper_clear_silhouette)
            item.slider.sliderReleased.connect(self.host._helper_add_silhouette)

        self.on_changed(0, is_commit=False)
        
        self.host.opts._impl_sync_func["sides"][self.str_now] = lambda: self._sync_sides_from_host("sides", self.host.opts.sides)
        self.owner.opts._impl_sync_func["window_length"][self.str_now] = lambda: self._sync_sides_from_host("window_length", self.owner.opts.window_length)
        self.obj._impl_sync_func["state_is_smooth"][self.str_now] = lambda: self._sync_sides_from_host("state_is_smooth", self.obj.state_is_smooth)
        self.obj._impl_sync_func["state_is_wrap"][self.str_now] = lambda: self._sync_sides_from_host("state_is_wrap", self.obj.state_is_wrap)


    def on_changed(self, _v=0, is_commit=True, is_only_smooth=False):
        for item in self.sliders.values():
            item.sync_to_state(self.state)

        if is_commit:
            self.commit(is_only_smooth=is_only_smooth)
            

    def commit(self, is_only_smooth=False):
        
        if is_only_smooth:
            self.owner.opts.window_length = int(self.state["window_length"])
            return
        
        # ---- radius ----
        current_radius = self.host._impl_opts_backup[self.str_now]["radius"]
        scale = float(self.state["radius_rescale"])
        if callable(current_radius):
            radius_now = lambda x: scale * current_radius(x)
        else:
            radius_now = scale * float(current_radius)
            
        # ---- color (controlled or restore) ----
        if bool(self.state.get("is_use_control_color", False)):
            color_now = (
                float(self.state["color_r"]),
                float(self.state["color_g"]),
                float(self.state["color_b"]),
            )
            paint_by_now = 'color'
        else:
            color_now = self.host._impl_opts_backup[self.str_now]["color"]
            paint_by_now = self.host._impl_opts_backup[self.str_now]["paint_by"]
            
        # ---- opacity (controlled or restore) ----
        if bool(self.state.get("is_use_control_opacity", False)):
            opacity_now = self.state["opacity"]
        else:
            opacity_now = self.host._impl_opts_backup[self.str_now]["opacity"]

        self.obj.act_commit(
            radius=radius_now,
            color=color_now,
            opacity=opacity_now,
            paint_by=paint_by_now,
            sides=int(self.state["sides"]),
            is_silhouette=False,
        )
        
    def _on_toggle_is_wrap(self, _state: int):
        is_wrap = bool(self.chk_is_wrap.isChecked())
        self.state["is_wrap"] = is_wrap
        self.spheres.act_commit(coords=self._helper_create_sphere_coords(is_wrap))
        self.obj.act_commit(is_wrap=is_wrap)
        
    def _on_toggle_is_smooth(self, _state: int):
        is_smooth = bool(self.chk_is_smooth.isChecked())
        self.state["is_smooth"] = is_smooth
        self.sliders['window_length'].set_enabled(is_smooth)
        self.obj.act_commit(is_smooth=is_smooth)

    def _on_toggle_use_color(self, _state: int):
        is_color = bool(self.chk_use_color.isChecked())
        self.state["is_use_control_color"] = is_color
        for k in ("color_r", "color_g", "color_b"):
            self.sliders[k].set_enabled(is_color)
        self.commit()
        
    def _on_toggle_use_opacity(self, _state: int):
        is_opacity = bool(self.chk_use_opacity.isChecked())
        self.state["is_use_control_opacity"] = is_opacity
        self.sliders["opacity"].set_enabled(is_opacity)
        self.commit()
        
        
    def on_close(self):
        super().on_close()
        sync = getattr(self.owner.opts, "_impl_sync_func", None)
        for k, sub in sync.items():
            sub.pop(self.str_now, None)
        sync = getattr(self.obj, "_impl_sync_func", None)
        for k, sub in sync.items():
            sub.pop(self.str_now, None)
        object.__setattr__(self.owner, "_state_is_silhouette", True)
        self.spheres.act_remove()