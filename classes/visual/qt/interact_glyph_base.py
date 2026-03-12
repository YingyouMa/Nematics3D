from qtpy import QtWidgets
from .panel_base import PanelBase, make_labeled_slider_row, make_RGB_slider, LogTickMapper

class InteractGlyphBase(PanelBase):
    def __init__(self, host, figure, title=None, 
                 is_radius=True, is_sides=True, is_geometry=False,
                 is_color=True, is_opacity=True):
        
        self.config = {
            "is_radius": is_radius,
            "is_sides": is_sides,
            "is_color": is_color,
            "is_opacity": is_opacity,
            "is_geometry": is_geometry
        }
        self._custom_sliders = []
        
        display_title = title or f"Control of {host}"
        super().__init__(host, figure, title=display_title)
        


    def build_ui(self):
        # ----------------------------
        # Geometry Group
        # ----------------------------

        if self.config["is_radius"] or self.config["is_sides"] or self.config["is_geometry"]:
            self.group_geometry = QtWidgets.QGroupBox("Geometry", self)
            self.gl_geometry = QtWidgets.QVBoxLayout(self.group_geometry)
            self.layout.addWidget(self.group_geometry)

            if self.config["is_radius"]:
                self.state["radius_rescale"] = 1.0
                log_mapper = LogTickMapper(value_min=0.2, value_max=5, base=10.0)
                self.sliders["radius_rescale"] = make_labeled_slider_row(
                    parent=self.group_geometry, layout=self.gl_geometry,
                    name="radius_rescale", state_key="radius_rescale",
                    value_min=log_mapper.value_min, value_max=log_mapper.value_max,
                    value_init=1.0, tick_to_value=log_mapper.tick_to_value,
                    value_to_tick=log_mapper.value_to_tick
                )
                self.lbl_radius = QtWidgets.QLabel(self.group_geometry)
                self.gl_geometry.addWidget(self.lbl_radius)
                self._update_radius_label()

            if self.config["is_sides"]:
                self.state["sides"] = int(self.host.opts.sides)
                self.sliders["sides"] = make_labeled_slider_row(
                    parent=self.group_geometry, layout=self.gl_geometry,
                    name="sides", state_key="sides",
                    value_min=3, value_max=30,
                    value_init=self.state["sides"], value_fmt="{:.0f}"
                )

            self._build_extra_geometry(self.group_geometry, self.gl_geometry)

        # ----------------------------
        # RGB Group
        # ----------------------------
        if self.config["is_color"]:
            
            init_rgb = self.host._calc_color[0]
            self.state["color_r"] = init_rgb[0]
            self.state["color_g"] = init_rgb[1]
            self.state["color_b"] = init_rgb[2]
            
            group_RGB = QtWidgets.QGroupBox("Color (RGB 0..1)", self)
            gl_RGB = QtWidgets.QVBoxLayout(group_RGB)
            self.layout.addWidget(group_RGB)
            
            make_RGB_slider(group_RGB, gl_RGB, self.sliders, "color", init_rgb)
            
            self.chk_use_color = QtWidgets.QCheckBox("Use controlled color", group_RGB)
            self.chk_use_color.setChecked(False)
            gl_RGB.addWidget(self.chk_use_color)
            self.chk_use_color.stateChanged.connect(self._on_toggle_use_color)
            
            for k in ("color_r", "color_g", "color_b"):
                self.sliders[k].set_enabled(False)

        # ----------------------------
        # Opacity Group
        # ----------------------------
        if self.config["is_opacity"]:
            self.state["opacity"] = self.host._calc_opacity[0]
            group_opacity = QtWidgets.QGroupBox("Opacity (0..1)", self)
            gl_opacity = QtWidgets.QVBoxLayout(group_opacity)
            self.layout.addWidget(group_opacity)

            self.sliders["opacity"] = make_labeled_slider_row(
                parent=group_opacity, layout=gl_opacity,
                name="opacity", state_key="opacity",
                value_min=0, value_max=1, value_init=self.state["opacity"],
                tick_to_value=lambda t: float(t / 100.0),
                value_to_tick=lambda v: int(v * 100)
            )
            self.chk_use_opacity = QtWidgets.QCheckBox("Use controlled opacity", group_opacity)
            self.chk_use_opacity.setChecked(False)
            gl_opacity.addWidget(self.chk_use_opacity)
            self.chk_use_opacity.stateChanged.connect(self._on_toggle_use_opacity)
            self.sliders["opacity"].set_enabled(False)
            
        self._build_extra_group()

        for item in self.sliders.values():
            if item not in self._custom_sliders:
                item.slider.valueChanged.connect(self.on_changed)
            item.slider.sliderPressed.connect(self.host._helper_clear_silhouette)
            item.slider.sliderReleased.connect(self.host._helper_add_silhouette)
            
        

    def _build_extra_geometry(self, parent, layout):
        pass
    
    def _build_extra_group(self):
        pass

    def _update_radius_label(self):
        if hasattr(self, 'lbl_radius') and hasattr(self.host, '_calc_radius'):
            self.lbl_radius.setText(f"The first radius is {self.host._calc_radius[0]:.2f}")

    def commit(self):
        params = {}
        
        if self.config["is_radius"]:
            current_radius = self.host._opts_backup[self.str_now_live]["radius"]
            scale = float(self.state["radius_rescale"])
            params['radius'] = (lambda x: scale * current_radius(x)) if callable(current_radius) else scale * float(current_radius)

        if self.config["is_color"]:
            if self.state.get("is_use_control_color"):
                params['color'] = (float(self.state["color_r"]), float(self.state["color_g"]), float(self.state["color_b"]))
                params['paint_by'] = 'color'
            else:
                params['color'] = self.host._opts_backup[self.str_now_live]["color"]
                params['paint_by'] = self.host._opts_backup[self.str_now_live]["paint_by"]

        if self.config["is_opacity"]:
            params['opacity'] = self.state["opacity"] if self.state.get("is_use_control_opacity") else self.host._opts_backup[self.str_now_live]["opacity"]
        
        if self.config["is_sides"]:
            params['sides'] = int(self.state["sides"])

        self._extra_commit(params)

        self._is_gui_updating = True
        try:
            with self.host._helper_temporarily_set_silhouette(False):
                self.host.act_commit(**params)
        finally:
            self._is_gui_updating = False

    def _extra_commit(self, params):
        pass

    def _on_toggle_use_color(self, _):
        is_color = self.chk_use_color.isChecked()
        self.state["is_use_control_color"] = is_color
        for k in ("color_r", "color_g", "color_b"): 
            self.sliders[k].set_enabled(is_color)
        if not self._is_block_chk_commit:
            self.commit()

    def _on_toggle_use_opacity(self, _):
        is_opacity = self.chk_use_opacity.isChecked()
        self.state["is_use_control_opacity"] = is_opacity
        self.sliders["opacity"].set_enabled(is_opacity)
        if not self._is_block_chk_commit:
            self.commit()

    def _sync_func(self, **kwargs):
        if not getattr(self, "_is_gui_updating", False):
            self.host._opts_backup[self.str_now_live].update(kwargs)
            if 'sides' in kwargs and self.config["is_sides"]:
                self._sync_from_host_slider("sides", kwargs["sides"])
            if 'radius' in kwargs and self.config["is_radius"]:
                self.sliders["radius_rescale"].set_tick(1, is_block_signals=True)
                self._update_radius_label()
            if 'color' in kwargs and self.config["is_color"]:
                self._is_block_chk_commit = True
                self.chk_use_color.setChecked(False)
                self._is_block_chk_commit = False
            if 'opacity' in kwargs and self.config["is_opacity"]:
                self._is_block_chk_commit = True
                self.chk_use_opacity.setChecked(False)
                self._is_block_chk_commit = False
