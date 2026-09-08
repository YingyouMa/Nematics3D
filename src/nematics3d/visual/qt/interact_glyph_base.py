import numpy as np
from qtpy import QtWidgets

from nematics3d.visual.qt.lighting_console import LightingConsole
from nematics3d.visual.qt.panel_base import (
    LogTickMapper,
    PanelBase,
    make_RGB_slider,
    make_labeled_slider_row,
)


class InteractGlyphBase(PanelBase):
    """Shared Qt controls for PlotGlyph-like visualization hosts."""

    def _helper_panel_marker_key(self):
        return f"{self.str_now}::calc0"

    def _helper_update_first_point_marker(self):
        pm = getattr(self.fig, "pick_manager", None) if self.fig is not None else None
        if pm is None:
            return
        coords = getattr(self.host, "calc_coords", None)
        if coords is None or len(coords) == 0:
            pm.act_remove_helper_marker(self._helper_panel_marker_key())
            return
        pm.act_set_helper_marker(
            self._helper_panel_marker_key(),
            np.asarray(coords[0], dtype=float),
            marker_id=0,
        )

    def _helper_remove_first_point_marker(self):
        pm = getattr(self.fig, "pick_manager", None) if self.fig is not None else None
        if pm is not None:
            pm.act_remove_helper_marker(self._helper_panel_marker_key())

    def __init__(
        self,
        host,
        figure,
        title=None,
        is_radius=True,
        is_sides=True,
        is_geometry=False,
        is_color=True,
        is_opacity=True,
    ):
        self.config = {
            "is_radius": is_radius,
            "is_sides": is_sides,
            "is_color": is_color,
            "is_opacity": is_opacity,
            "is_geometry": is_geometry,
        }
        self._custom_sliders = []
        self._lighting_console = None
        if title is None:
            host_kind = type(host).__name__.removeprefix("Plot")
            display_title = f"{host_kind} Controls of {host.name!r}"
        else:
            display_title = title
        super().__init__(host, figure, title=display_title)
        self._helper_update_first_point_marker()

    def build_ui(self):
        if (
            self.config["is_radius"]
            or self.config["is_sides"]
            or self.config["is_geometry"]
            or hasattr(self.host, "act_bounds_enable")
        ):
            self.group_geometry = QtWidgets.QGroupBox("Geometry", self)
            self.gl_geometry = QtWidgets.QVBoxLayout(self.group_geometry)
            self.layout.addWidget(self.group_geometry)

            if self.config["is_radius"]:
                self.state["radius_rescale"] = 1.0
                self._radius_rescale_base = self._helper_clone_resolver_value(
                    self.host.opts.radius
                )
                log_mapper = LogTickMapper(value_min=0.2, value_max=5, base=10.0)
                self.sliders["radius_rescale"] = make_labeled_slider_row(
                    parent=self.group_geometry,
                    layout=self.gl_geometry,
                    name="radius_rescale",
                    state_key="radius_rescale",
                    value_min=log_mapper.value_min,
                    value_max=log_mapper.value_max,
                    value_init=1.0,
                    tick_to_value=log_mapper.tick_to_value,
                    value_to_tick=log_mapper.value_to_tick,
                    input_out_of_range="expand_max",
                )
                self.lbl_radius = QtWidgets.QLabel(self.group_geometry)
                self.gl_geometry.addWidget(self.lbl_radius)
                self._update_radius_label()

            if self.config["is_sides"]:
                self.state["sides"] = int(self.host.opts.sides)
                self.sliders["sides"] = make_labeled_slider_row(
                    parent=self.group_geometry,
                    layout=self.gl_geometry,
                    name="sides",
                    state_key="sides",
                    value_min=3,
                    value_max=30,
                    value_init=self.state["sides"],
                    value_fmt="{:.0f}",
                )

            if hasattr(self.host, "act_bounds_enable") and hasattr(
                self.host, "act_bounds_disable"
            ):
                self.state["is_bounds_enabled"] = bool(
                    getattr(self.host, "impl_is_bounds_enabled", True)
                )
                self.chk_is_bounds_enabled = QtWidgets.QCheckBox(
                    "Enable bounds effect", self.group_geometry
                )
                self.chk_is_bounds_enabled.setChecked(self.state["is_bounds_enabled"])
                self.gl_geometry.addWidget(self.chk_is_bounds_enabled)
                self.chk_is_bounds_enabled.stateChanged.connect(
                    self._on_toggle_bounds_enabled
                )
                self.lbl_bounds_restore_note = QtWidgets.QLabel(
                    "Note: this option does not currently support Restore Original.",
                    self.group_geometry,
                )
                self.lbl_bounds_restore_note.setWordWrap(True)
                self.gl_geometry.addWidget(self.lbl_bounds_restore_note)

            self._build_extra_geometry(self.group_geometry, self.gl_geometry)

        if self.config["is_color"]:
            init_rgb = self.host.calc_color[0]
            self.state["color_r"] = init_rgb[0]
            self.state["color_g"] = init_rgb[1]
            self.state["color_b"] = init_rgb[2]
            group_rgb = QtWidgets.QGroupBox("Color (RGB 0..1)", self)
            gl_rgb = QtWidgets.QVBoxLayout(group_rgb)
            self.layout.addWidget(group_rgb)
            make_RGB_slider(group_rgb, gl_rgb, self.sliders, "color", init_rgb)
            self.chk_use_color = QtWidgets.QCheckBox(
                "Override with uniform color", group_rgb
            )
            self.chk_use_color.setChecked(False)
            gl_rgb.addWidget(self.chk_use_color)
            self.chk_use_color.stateChanged.connect(self._on_toggle_use_color)
            for key in ("color_r", "color_g", "color_b"):
                self.sliders[key].set_enabled(False)

        if self.config["is_opacity"]:
            self.state["opacity"] = self.host.calc_opacity[0]
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
                value_to_tick=lambda v: int(v * 100),
            )
            self.chk_use_opacity = QtWidgets.QCheckBox(
                "Override with uniform opacity", group_opacity
            )
            self.chk_use_opacity.setChecked(False)
            gl_opacity.addWidget(self.chk_use_opacity)
            self.chk_use_opacity.stateChanged.connect(self._on_toggle_use_opacity)
            self.sliders["opacity"].set_enabled(False)

        self._build_extra_group()
        self.btn_lighting_console = QtWidgets.QPushButton(
            "Open Lighting Controls", self
        )
        self.layout.addWidget(self.btn_lighting_console)
        self.btn_lighting_console.clicked.connect(self._open_lighting_console)

    def _build_extra_geometry(self, parent, layout):
        pass

    def _build_extra_group(self):
        pass

    def _open_lighting_console(self):
        if self._lighting_console is None:
            self._lighting_console = LightingConsole(self.host, self)
        self._lighting_console.show()
        self._lighting_console.raise_()
        self._lighting_console.activateWindow()

    def _helper_get_first_used_point_radius(self):
        if not hasattr(self.host, "calc_radius"):
            return None, None
        radius_all = np.asarray(self.host.calc_radius, dtype=float)
        if radius_all.size == 0:
            return None, None
        keep_index = getattr(self.host, "calc_keep_index", None)
        if keep_index is not None:
            keep_index = np.asarray(keep_index, dtype=int)
            if keep_index.size == 0:
                return None, None
            source_index = int(keep_index[0])
            return float(radius_all[source_index]), source_index
        return float(radius_all[0]), 0

    @staticmethod
    def _helper_clone_resolver_value(value):
        if callable(value):
            return value
        if np.isscalar(value):
            return float(value)
        return np.asarray(value, dtype=float).copy()

    @staticmethod
    def _helper_scale_resolver_value(value, scale):
        """Scale one scalar/array/callable resolver without changing its input form."""
        scale = float(scale)
        if callable(value):
            return lambda x: scale * value(x)
        if np.isscalar(value):
            return scale * float(value)
        return scale * np.asarray(value, dtype=float)

    def _helper_reset_rescale_control(self, slider_name, opts_attr, base_attr):
        """Adopt a host-side resolver value as the new scale=1 baseline."""
        object.__setattr__(
            self,
            base_attr,
            self._helper_clone_resolver_value(getattr(self.host.opts, opts_attr)),
        )
        slider = self.sliders.get(slider_name)
        if slider is not None:
            slider.set_tick(1, is_block_signals=True)

    def _update_radius_label(self):
        if not hasattr(self, "lbl_radius"):
            return
        radius, _source_index = self._helper_get_first_used_point_radius()
        if radius is None:
            self.lbl_radius.setText("No currently used point is available.")
            return
        self.lbl_radius.setText(f"Radius at the red helper marker: {radius:.2f}")

    def _helper_build_commit_params(self):
        params = {}
        if self.config["is_radius"]:
            radius_base = self._radius_rescale_base
            scale = float(self.state["radius_rescale"])
            if callable(radius_base):
                params["radius"] = lambda x: scale * radius_base(x)
            elif np.isscalar(radius_base):
                params["radius"] = scale * float(radius_base)
            else:
                params["radius"] = scale * np.asarray(radius_base, dtype=float)
        if self.config["is_color"]:
            if self.state.get("is_uniform_color_override"):
                params["color"] = (
                    float(self.state["color_r"]),
                    float(self.state["color_g"]),
                    float(self.state["color_b"]),
                )
                params["paint_by"] = "color"
            else:
                params["color"] = self.host.opts.color
                params["paint_by"] = self.host.opts.paint_by
        if self.config["is_opacity"]:
            params["opacity"] = (
                self.state["opacity"]
                if self.state.get("is_uniform_opacity_override")
                else self.host.opts.opacity
            )
        if self.config["is_sides"]:
            params["sides"] = int(self.state["sides"])
        self._extra_commit(params)
        return params

    def _helper_run_commit(self, params):
        self.host.act_commit(**params)

    def commit(self):
        params = self._helper_build_commit_params()
        self._is_gui_updating = True
        try:
            self._helper_run_commit(params)
        finally:
            self._is_gui_updating = False

    def _extra_commit(self, params):
        pass

    def _on_toggle_use_color(self, _):
        is_color = self.chk_use_color.isChecked()
        self.state["is_uniform_color_override"] = is_color
        for key in ("color_r", "color_g", "color_b"):
            self.sliders[key].set_enabled(is_color)
        if not self._is_block_chk_commit:
            self.commit()

    def _on_toggle_use_opacity(self, _):
        is_opacity = self.chk_use_opacity.isChecked()
        self.state["is_uniform_opacity_override"] = is_opacity
        self.sliders["opacity"].set_enabled(is_opacity)
        if not self._is_block_chk_commit:
            self.commit()

    def _on_toggle_bounds_enabled(self, _):
        is_enabled = self.chk_is_bounds_enabled.isChecked()
        self.state["is_bounds_enabled"] = is_enabled
        if self._is_block_chk_commit:
            return
        if is_enabled:
            self.host.act_bounds_enable()
        else:
            self.host.act_bounds_disable()

    def _sync_func(self, **kwargs):
        is_gui_updating = getattr(self, "_is_gui_updating", False)
        is_bounds_enabled = kwargs.get("is_bounds_enabled", None)
        if is_bounds_enabled is not None and hasattr(self, "chk_is_bounds_enabled"):
            self._is_block_chk_commit = True
            self.chk_is_bounds_enabled.setChecked(bool(is_bounds_enabled))
            self._is_block_chk_commit = False
            self.state["is_bounds_enabled"] = bool(is_bounds_enabled)
        if not is_gui_updating:
            if "sides" in kwargs and self.config["is_sides"]:
                self._sync_from_host_slider("sides", kwargs["sides"])
            if "color" in kwargs and self.config["is_color"]:
                self._is_block_chk_commit = True
                self.chk_use_color.setChecked(False)
                self._is_block_chk_commit = False
            if "opacity" in kwargs and self.config["is_opacity"]:
                self._is_block_chk_commit = True
                self.chk_use_opacity.setChecked(False)
                self._is_block_chk_commit = False
        if "radius" in kwargs and self.config["is_radius"]:
            if not is_gui_updating:
                self._helper_reset_rescale_control(
                    "radius_rescale", "radius", "_radius_rescale_base"
                )
            self._update_radius_label()
        update_length_label = getattr(self, "_update_length_label", None)
        if callable(update_length_label) and "length" in kwargs:
            if not is_gui_updating and "length_rescale" in self.sliders:
                self.sliders["length_rescale"].set_tick(1, is_block_signals=True)
            update_length_label()
        self._helper_update_first_point_marker()

    def on_close(self):
        if self._lighting_console is not None:
            self._lighting_console.close()
        self._helper_remove_first_point_marker()
        super().on_close()
