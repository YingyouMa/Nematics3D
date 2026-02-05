from qtpy import QtWidgets

from .panel_base import PanelBase, make_labeled_slider_row, make_RGB_slider

class InteractRod(PanelBase):
    def __init__(self, host):
        super().__init__(host, title="Rod Controls")

    def build_ui(self):
        # ----------------------------
        # initial state
        # ----------------------------
        self.state = {
            "radius_rescale":           1.0,
            "length_rescale":           1.0,
            "sides":                    int(self.host.opts.sides),
            "is_use_control_color":     False,
            "is_use_control_opacity":   False,
            "color":                    (1,1,1),
            "opacity":                  1,
            }

        # ----------------------------
        # Geometry group
        # ----------------------------
        group_geometry = QtWidgets.QGroupBox("Geometry", self)
        gl_geometry = QtWidgets.QVBoxLayout(group_geometry)
        self.layout.addWidget(group_geometry)

        
        self.sliders["radius_rescale"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="radius_rescale",
            tick_min=20,
            tick_max=500,
            tick_init=100,     
            tick_to_value=lambda t: t / 100.0,
            value_fmt="{:.4g}",
        )
        
        self.sliders["length_rescale"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="length_rescale",
            tick_min=50,
            tick_max=200,
            tick_init=100,     
            tick_to_value=lambda t: t / 100.0,
            value_fmt="{:.4g}",
        )

    
        self.sliders["sides"] = make_labeled_slider_row(
            parent=group_geometry,
            layout=gl_geometry,
            name="sides",
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
        
        init_rgb = self.host._calc_color[0]
        make_RGB_slider(
            parent=group_RGB,
            layout=gl_RGB,
            sliders=self.sliders,
            prefix="color",
            init_rgb=init_rgb
            )
        
        self.chk_use_color = QtWidgets.QCheckBox("Use controlled color", group_RGB)
        self.chk_use_color.setChecked(self.state["is_use_control_color"])
        gl_RGB.addWidget(self.chk_use_color)
        self.chk_use_color.stateChanged.connect(self._on_toggle_use_color)
        self._apply_color_enabled()
        
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
            tick_min=0,
            tick_max=100,
            tick_init=int(self.host._calc_opacity[0]*100),   
            tick_to_value=lambda t: float(t/100.0),
            value_fmt="{:.2f}",
        )
        
        self.chk_use_opacity = QtWidgets.QCheckBox("Use controlled opacity", group_opacity)
        self.chk_use_opacity.setChecked(self.state["is_use_control_opacity"])
        gl_opacity.addWidget(self.chk_use_opacity)
        self.chk_use_opacity.stateChanged.connect(self._on_toggle_use_opacity)
        self._apply_opacity_enabled()


        for item in self.sliders.values():
            item.slider.valueChanged.connect(self.on_changed)
            item.slider.sliderPressed.connect(self.host._helper_clear_silhouette)
            item.slider.sliderReleased.connect(self.host._helper_add_silhouette)

        self.on_changed(0, is_commit=False)
        
        self.host.opts._impl_sync_func["sides"][self.str_now] = lambda: self._sync_sides_from_host("sides", self.host.opts.sides)

    def on_changed(self, _v: int = 0, is_commit: bool = True):
        # ---- radius_rescale ----
        rr = float(self.sliders["radius_rescale"].get_value())
        self.sliders["radius_rescale"].label.setText(f"{rr:.4g}")
        self.state["radius_rescale"] = rr
        
        # ---- length_rescale ----
        lr = float(self.sliders["length_rescale"].get_value())
        self.sliders["length_rescale"].label.setText(f"{lr:.4g}")
        self.state["length_rescale"] = lr
    
        # ---- sides ----
        sides_f = float(self.sliders["sides"].get_value())
        sides = int(round(sides_f))
        self.sliders["sides"].label.setText(f"{sides:d}")
        self.state["sides"] = sides
    
        # ---- color (RGB) + checkbox ----
        self.state["is_use_control_color"] = bool(self.chk_use_color.isChecked())
    
        r = float(self.sliders["color_r"].get_value())
        g = float(self.sliders["color_g"].get_value())
        b = float(self.sliders["color_b"].get_value())
    
        self.sliders["color_r"].label.setText(f"{r:.4g}")
        self.sliders["color_g"].label.setText(f"{g:.4g}")
        self.sliders["color_b"].label.setText(f"{b:.4g}")
        
        self.state["color"] = (r, g, b)
        
        # ---- opacity + checkbox ----
        self.state["is_use_control_opacity"] = bool(self.chk_use_opacity.isChecked())
        opacity = float(self.sliders["opacity"].get_value())
        self.sliders["opacity"].label.setText(f"{opacity:4g}")
        self.state["opacity"] = opacity

        if is_commit:
            self.commit()

    def commit(self):
        # ---- radius ----
        current_radius = self.host._impl_opts_backup[self.str_now]["radius"]
        scale = float(self.state["radius_rescale"])
        if callable(current_radius):
            radius_now = lambda x: scale * current_radius(x)
        else:
            radius_now = scale * float(current_radius)
            
        # ---- length ----
        current_length = self.host._impl_opts_backup[self.str_now]["length"]
        scale = float(self.state["length_rescale"])
        if callable(current_length):
            length_now = lambda x: scale * current_length(x)
        else:
            length_now = scale * float(current_length)
            
        # ---- color (controlled or restore) ----
        if bool(self.state.get("is_use_control_color", False)):
            color_now = tuple(float(x) for x in self.state["color"])
            paint_by_now = 'color'
        else:
            color_now = self.host._impl_opts_backup[self.str_now]["color"]
            paint_by_now = self.host._impl_opts_backup[self.str_now]["paint_by"]
            
        # ---- opacity (controlled or restore) ----
        if bool(self.state.get("is_use_control_opacity", False)):
            opacity_now = self.state["opacity"]
        else:
            opacity_now = self.host._impl_opts_backup[self.str_now]["opacity"]

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
        self.state["is_use_control_color"] = self.chk_use_color.isChecked()
        self._apply_color_enabled()
        self.commit()
        
    def _on_toggle_use_opacity(self, _state: int):
        self.state["is_use_control_opacity"] = self.chk_use_opacity.isChecked()
        self._apply_opacity_enabled()
        self.commit()
        
    def _apply_color_enabled(self):
        color_enabled = bool(self.chk_use_color.isChecked())
        for k in ["color_r", "color_g", "color_b"]:
            item = self.sliders[k]
            item.slider.setEnabled(color_enabled)
            item.label.setEnabled(color_enabled)
            
    def _apply_opacity_enabled(self):
        opacity_enabled = bool(self.chk_use_opacity.isChecked())
        item = self.sliders['opacity']
        item.slider.setEnabled(opacity_enabled)
        item.label.setEnabled(opacity_enabled)