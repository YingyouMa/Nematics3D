from qtpy import QtWidgets

from .panel_base import PanelBase, make_labeled_slider_row, make_RGB_slider, LogTickMapper

class InteractRod(PanelBase):
    def __init__(self, host):
        super().__init__(host, title=f"Rod Controls of {host.name!r}")

    def build_ui(self):

        self.state = {
            "radius_rescale":           1.0,
            "length_rescale":           1.0,
            "sides":                    int(self.host.opts.sides),
            "is_use_control_color":     False,
            "is_use_control_opacity":   False,
            "color":                    self.host._calc_color[0],
            "opacity":                  self.host._calc_opacity[0],
            }

        # ----------------------------
        # Geometry group
        # ----------------------------
        group_geometry = QtWidgets.QGroupBox("Geometry", self)
        gl_geometry = QtWidgets.QVBoxLayout(group_geometry)
        self.layout.addWidget(group_geometry)
        
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
            value_min=log_mapper.value_min,
            value_max=log_mapper.value_max,
            value_init=1,
            tick_to_value=log_mapper.tick_to_value,
            value_to_tick=log_mapper.value_to_tick
        )
        
        self.sliders["length_rescale"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="length_rescale",
            state_key="length_rescale",
            value_min=log_mapper.value_min,
            value_max=log_mapper.value_max,
            value_init=1,
            tick_to_value=log_mapper.tick_to_value,
            value_to_tick=log_mapper.value_to_tick
        )

    
        self.sliders["sides"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="sides",
            state_key="sides",
            value_min=4,
            value_max=30,
            value_init=self.state["sides"],
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
            init_rgb=self.state["color"],
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
            value_min=0,
            value_max=1,
            value_init=self.state["opacity"],
            tick_to_value=lambda t: float(t / 100.0),
            value_to_tick=lambda v: int(v * 100)
        )
        
        self.chk_use_opacity = QtWidgets.QCheckBox("Use controlled opacity", group_opacity)
        self.chk_use_opacity.setChecked(self.state["is_use_control_opacity"])
        gl_opacity.addWidget(self.chk_use_opacity)
        self.chk_use_opacity.stateChanged.connect(self._on_toggle_use_opacity)
        self.sliders["opacity"].set_enabled(self.chk_use_opacity.isChecked())


        for item in self.sliders.values():
            item.slider.valueChanged.connect(self.on_changed)
            item.slider.sliderPressed.connect(self.host._helper_clear_silhouette)
            item.slider.sliderReleased.connect(self.host._helper_add_silhouette)

        self.on_changed(0, is_commit=False)
        
        self.host.opts._impl_sync_func["sides"][self.str_now] = (
            lambda: self._sync_from_host("sides", self.host.opts.sides)
        )


    def commit(self):
        # ---- radius ----
        current_radius = self.host._opts_backup[self.str_now]["radius"]
        scale = float(self.state["radius_rescale"])
        if callable(current_radius):
            radius_now = lambda x: scale * current_radius(x)
        else:
            radius_now = scale * float(current_radius)
            
        # ---- length ----
        current_length = self.host._opts_backup[self.str_now]["length"]
        scale = float(self.state["length_rescale"])
        if callable(current_length):
            length_now = lambda x: scale * current_length(x)
        else:
            length_now = scale * float(current_length)
            
        # ---- color (controlled or restore) ----
        if bool(self.state.get("is_use_control_color", False)):
            color_now = (
                float(self.state["color_r"]),
                float(self.state["color_g"]),
                float(self.state["color_b"]),
            )
            paint_by_now = 'color'
        else:
            color_now = self.host._opts_backup[self.str_now]["color"]
            paint_by_now = self.host._opts_backup[self.str_now]["paint_by"]
            
        # ---- opacity (controlled or restore) ----
        if bool(self.state.get("is_use_control_opacity", False)):
            opacity_now = self.state["opacity"]
        else:
            opacity_now = self.host._opts_backup[self.str_now]["opacity"]

        self.host.act_commit(
            radius=radius_now,
            length=length_now,
            color=color_now,
            opacity=opacity_now,
            paint_by=paint_by_now,
            sides=int(self.state["sides"]),
            is_silhouette=False,
        )

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
