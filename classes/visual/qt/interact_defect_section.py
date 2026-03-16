import numpy as np
from qtpy import QtWidgets

from .panel_base import PanelBase, make_labeled_slider_row, LogTickMapper
from ..plot_rod import PlotRod
from Nematics3D.general import is_equal, is_given_str


class InteractDefectSection(PanelBase):
    def __init__(self, field, figure):
        self.field = field
        self.defect_plane = field.grid.wrapper
        object.__setattr__(self.field, "_state_is_interactable", False)
        
        self.visual_normal = PlotRod(
            coords=field.grid.opts.origin,
            orient=field.grid.opts.normal,
            radius=field.grid.opts.dr/4,
            length=field.grid.opts.layers * field.grid.opts.dr * 2.5,
            color=(1,0,0),
            figure=figure,
            name=f"The normal of {field.grid.name!r}",
            category="Interaction",
            is_reset_camera=False,
            is_visible=False
        )
        object.__setattr__(self.visual_normal, "_state_is_interactable", False)
        object.__setattr__(self, "_is_continuous_interacting", False)

        super().__init__(field.grid, figure, title=f"Controls of {field.grid.name!r}")
        self.defect_plane.act_save_opts(self.str_now)
        self.defect_plane.act_save_opts(self.str_now_live)
        self._section_sync_name = self.str_now_live + "_section"
        self.defect_plane.act_attach_sync_task(self._section_sync_name, self._sync_func_defect_plane)

    def _iter_silhouette_targets(self):
        targets = [self.visual_normal]
        for name in (
            "_entity_visual_nb",
            "_entity_visual_nd",
            "_entity_visual_defect",
        ):
            visual = getattr(self.field, name, None)
            if visual is not None:
                targets.append(visual)

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
            if not hasattr(visual, "_state_is_silhouette"):
                continue
            object.__setattr__(visual, "_state_is_silhouette", False)
            if hasattr(visual, "_helper_clear_silhouette"):
                visual._helper_clear_silhouette()

    def _helper_end_continuous_interaction(self, *_args):
        if not self._is_continuous_interacting:
            return
        object.__setattr__(self, "_is_continuous_interacting", False)
        for visual in self._iter_silhouette_targets():
            if not hasattr(visual, "_state_is_silhouette"):
                continue
            object.__setattr__(visual, "_state_is_silhouette", True)
            if getattr(visual, "_entity", None) is not None and hasattr(visual, "_helper_add_silhouette"):
                visual._helper_add_silhouette()

    def _update_normal_visual(self, is_visible=True):
        origin = np.asarray(self.host.opts.origin, dtype=float).reshape(1, 3)
        normal = np.asarray(self.host.opts.normal, dtype=float).reshape(1, 3)
        self.visual_normal.act_commit(
            coords=origin,
            orient=normal,
            radius=float(self.host.opts.dr) / 4,
            length=float(self.host.opts.layers * self.host.opts.dr * 2.5),
            is_visible=is_visible,
        )

    def build_ui(self):
        def _on_toggle_show_axes():
            checked = self.chk_is_show_axes.isChecked()
            if checked:
                self._update_normal_visual(is_visible=True)
            else:
                self.visual_normal.opts.is_visible = False
            
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
            else self.host.opts.dr
        )
        
        # fmt: off
        self.state = {
            "u_percent":                            self.defect_plane.opts.u_percent,
            "dr":                                   self.host.opts.dr,
            "arc_dist":                             arc_dist_init,
            "layers":                               self.host.opts.layers,
            "is_use_control_arc_dist":              self.host.opts.arc_dist is not None,
            "is_use_control_normal":                not (
                                                        isinstance(self.defect_plane.state_normal, str)
                                                        and self.defect_plane.state_normal == "tangent"
                                                    ),
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
        
        self.sliders["u_percent"] = make_labeled_slider_row(
            parent=group_vector,
            layout=gl_vector,
            name="u_percent",
            state_key="u_percent",
            value_min=0,
            value_max=100,
            value_init=self.state["u_percent"],
            tick_to_value=float,
            value_to_tick=int,
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

        log_arc_dist = LogTickMapper(
            value_min=0.2 * self.state["arc_dist"],
            value_max=5 * self.state["arc_dist"],
            base=10.0,
        )
        self.sliders["arc_dist"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="arc_dist",
            value_min=log_arc_dist.value_min,
            value_max=log_arc_dist.value_max,
            value_init=self.state["arc_dist"],
            tick_to_value=log_arc_dist.tick_to_value,
            value_to_tick=log_arc_dist.value_to_tick,
        )

        self.chk_use_arc_dist = QtWidgets.QCheckBox(
            "Use controlled arc_dist", group_scalar
        )
        self.chk_use_arc_dist.setChecked(self.state["is_use_control_arc_dist"])
        gl_scalar.addWidget(self.chk_use_arc_dist)
        self.chk_use_arc_dist.stateChanged.connect(self._on_toggle_use_arc_dist)
        self.sliders["arc_dist"].set_enabled(self.state["is_use_control_arc_dist"])

        layers_max = max(4, int(np.ceil(self.state["layers"] * 3.0)))
        self.sliders["layers"] = make_labeled_slider_row(
            parent=group_scalar,
            layout=gl_scalar,
            name="layers",
            state_key="layers",
            value_min=1,
            value_max=layers_max,
            value_init=int(self.state["layers"]),
            value_fmt="{:.0f}",
        )
        
        
        
        
        
        
        for key, item in self.sliders.items():
            item.slider.valueChanged.connect(self.on_changed)
            item.slider.sliderPressed.connect(self._helper_begin_continuous_interaction)
            item.slider.sliderReleased.connect(self._helper_end_continuous_interaction)
                    
        self.on_changed(0, is_commit=False)
        
        
    def commit(self):
        # ---- normal ----
        if self.state["is_use_control_normal"]:
            normal_azimuth = np.deg2rad(self.state["normal_azimuth"])
            normal_polar_angle = np.deg2rad(self.state["normal_polar_angle"])
            normal_now = self._helper_calc_vec(normal_azimuth, normal_polar_angle)
        else:
            normal_now = "tangent"

        arc_dist_now = (
            float(self.state["arc_dist"])
            if bool(self.state.get("is_use_control_arc_dist", False))
            else None
        )

        params = {"u_percent": self.state["u_percent"]}
        if not is_equal(self.host.opts.dr, self.state["dr"]):
            params["dr"] = self.state["dr"]
        if not is_equal(self.host.opts.layers, int(self.state["layers"])):
            params["layers"] = int(self.state["layers"])
        if not is_equal(self.host.opts.arc_dist, arc_dist_now):
            params["arc_dist"] = arc_dist_now
        if self.state["is_use_control_normal"]:
            if not is_equal(self.defect_plane.state_normal, normal_now):
                params["state_normal"] = normal_now
        else:
            if not is_given_str(self.defect_plane.state_normal, "tangent"):
                params["state_normal"] = "tangent"

        self._is_gui_updating = True
        try:
            self.defect_plane.act_commit(**params)
        finally:
            self._is_gui_updating = False
        
        self.normal_info.setText(
            self._vect_text(self.host.opts.normal, "normal")
        )
        self.origin_info.setText(
            self._vect_text(self.host.opts.origin, "origin")
        )
        
        if self.chk_is_show_axes.isChecked():
            object.__setattr__(self.visual_normal, "_state_is_silhouette", False)
            try:
                self._update_normal_visual(is_visible=True)
            finally:
                object.__setattr__(
                    self.visual_normal,
                    "_state_is_silhouette",
                    not self._is_continuous_interacting,
                )
        
        
        
    def _on_toggle_use_normal(self, _state: int):
        result = self.chk_use_normal.isChecked()
        self.state["is_use_control_normal"] = result
        self.sliders["normal_azimuth"].set_enabled(result)
        self.sliders["normal_polar_angle"].set_enabled(result)
        self.commit()

    def _on_toggle_use_arc_dist(self, _state: int):
        result = self.chk_use_arc_dist.isChecked()
        self.state["is_use_control_arc_dist"] = result
        self.sliders["arc_dist"].set_enabled(result)
        self.commit()

    def _sync_func(self, **kwargs):
        if not getattr(self, "_is_gui_updating", False):
            self.host._opts_backup[self.str_now_live].update(kwargs)

            if "origin" in kwargs:
                self.origin_info.setText(self._vect_text(self.host.opts.origin, "origin"))
            if "normal" in kwargs:
                if self.state["is_use_control_normal"]:
                    self._sync_from_host_slider(
                        "normal_azimuth",
                        self.get_azimuth(self.host.opts.normal),
                    )
                    self._sync_from_host_slider(
                        "normal_polar_angle",
                        self.get_polar_angle(self.host.opts.normal),
                    )
                self.normal_info.setText(self._vect_text(self.host.opts.normal, "normal"))
            if "dr" in kwargs:
                self._sync_from_host_slider("dr", self.host.opts.dr)
            if "arc_dist" in kwargs:
                is_controlled = self.host.opts.arc_dist is not None
                self.state["is_use_control_arc_dist"] = is_controlled
                self.chk_use_arc_dist.blockSignals(True)
                try:
                    self.chk_use_arc_dist.setChecked(is_controlled)
                finally:
                    self.chk_use_arc_dist.blockSignals(False)
                self.sliders["arc_dist"].set_enabled(is_controlled)
                if is_controlled:
                    self._sync_from_host_slider("arc_dist", self.host.opts.arc_dist)
            if "layers" in kwargs:
                self._sync_from_host_slider("layers", int(self.host.opts.layers))

        if self.chk_is_show_axes.isChecked():
            self._update_normal_visual(is_visible=True)

    def _sync_func_defect_plane(self, **kwargs):
        if not getattr(self, "_is_gui_updating", False):
            self.defect_plane._opts_backup[self.str_now_live].update(kwargs)

            if "u_percent" in kwargs:
                self._sync_from_host_slider("u_percent", self.defect_plane.opts.u_percent)
            if "state_normal" in kwargs:
                is_controlled = not (
                    isinstance(self.defect_plane.state_normal, str)
                    and self.defect_plane.state_normal == "tangent"
                )
                self.state["is_use_control_normal"] = is_controlled
                self.chk_use_normal.blockSignals(True)
                try:
                    self.chk_use_normal.setChecked(is_controlled)
                finally:
                    self.chk_use_normal.blockSignals(False)
                self.sliders["normal_azimuth"].set_enabled(is_controlled)
                self.sliders["normal_polar_angle"].set_enabled(is_controlled)
                if is_controlled:
                    self._sync_from_host_slider(
                        "normal_azimuth",
                        self.get_azimuth(self.host.opts.normal),
                    )
                    self._sync_from_host_slider(
                        "normal_polar_angle",
                        self.get_polar_angle(self.host.opts.normal),
                    )
                self.normal_info.setText(self._vect_text(self.host.opts.normal, "normal"))

    def _on_reset_to_live(self):
        defect_live = {
            k: v
            for k, v in self.defect_plane._opts_backup[self.str_now_live].items()
            if k not in self.defect_plane.attrs_forbidden
        }
        host_live = {
            k: v
            for k, v in self.host._opts_backup[self.str_now_live].items()
            if k not in self.host.attrs_forbidden
        }
        self.defect_plane.act_commit(**defect_live)
        self.host.act_commit(**host_live)

    def _on_reset_to_original(self):
        defect_original = {
            k: v
            for k, v in self.defect_plane._opts_backup[self.str_now].items()
            if k not in self.defect_plane.attrs_forbidden
        }
        host_original = {
            k: v
            for k, v in self.host._opts_backup[self.str_now].items()
            if k not in self.host.attrs_forbidden
        }
        self.defect_plane.act_commit(**defect_original)
        self.host.act_commit(**host_original)
        self.defect_plane._opts_backup[self.str_now_live] = dict(defect_original)
        self.host._opts_backup[self.str_now_live] = dict(host_original)

        
    def on_close(self):
        self._helper_end_continuous_interaction()
        self.defect_plane.act_detach_sync_task(self._section_sync_name)
        super().on_close()
        object.__setattr__(self.field, "_state_is_interactable", True)
        self.visual_normal.act_remove()

