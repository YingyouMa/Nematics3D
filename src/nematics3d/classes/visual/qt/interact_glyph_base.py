import numpy as np
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import QSignalBlocker
from .panel_base import (
    PanelBase,
    make_labeled_slider_row,
    make_RGB_slider,
    LogTickMapper,
)

# NOTE: This panel is intentionally coupled to PlotGlyph-like hosts.
# The host is expected to provide glyph internals such as:
# `opts`, `calc_color`, `calc_opacity`, `calc_radius`, `opts_backup`,
# `state_is_silhouette`, `_helper_clear_silhouette`, and `_helper_add_silhouette`.


class LightingConsole(QtWidgets.QWidget):
    def __init__(self, host, parent_panel):
        super().__init__(parent_panel)
        self.host = host
        self.parent_panel = parent_panel
        self.sliders = {}
        self.state = {}
        self._is_gui_updating = False
        self._sync_name = f"{parent_panel.str_now}::lighting"

        self.setWindowTitle(f"Lighting Controls of {host.name!r}")
        self.setObjectName(f"{parent_panel.objectName()}_lighting")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)

        self._build_ui()
        self._wire_default_connections()
        self.host.act_attach_sync_task(self._sync_name, self._sync_func)

    def _build_ui(self):
        self.state = {
            "shading_type": str(self.host.opts.shading_type),
            "ambient": float(self.host.opts.ambient),
            "diffuse": float(self.host.opts.diffuse),
            "specular": float(self.host.opts.specular),
            "specular_power": float(self.host.opts.specular_power),
            "specular_color_r": float(self.host.opts.specular_color[0]),
            "specular_color_g": float(self.host.opts.specular_color[1]),
            "specular_color_b": float(self.host.opts.specular_color[2]),
            "metallic": float(self.host.opts.metallic),
            "roughness": float(self.host.opts.roughness),
        }

        group_mode = QtWidgets.QGroupBox("Shading Type", self)
        hl_mode = QtWidgets.QHBoxLayout(group_mode)
        self.layout.addWidget(group_mode)

        self.chk_phong = QtWidgets.QCheckBox("Phong", group_mode)
        self.chk_pbr = QtWidgets.QCheckBox("PBR", group_mode)
        hl_mode.addWidget(self.chk_phong)
        hl_mode.addWidget(self.chk_pbr)

        self.group_phong = QtWidgets.QGroupBox("Phong Lighting", self)
        gl_phong = QtWidgets.QVBoxLayout(self.group_phong)
        self.layout.addWidget(self.group_phong)

        make_RGB_slider(
            self.group_phong,
            gl_phong,
            self.sliders,
            "specular_color",
            init_rgb=tuple(np.asarray(self.host.opts.specular_color, dtype=float)),
            value_fmt="{:.3f}",
        )

        for key in ("ambient", "diffuse", "specular"):
            self.sliders[key] = make_labeled_slider_row(
                parent=self.group_phong,
                layout=gl_phong,
                name=key,
                state_key=key,
                value_min=0,
                value_max=1,
                value_init=self.state[key],
                tick_to_value=lambda t: float(t / 1000.0),
                value_to_tick=lambda v: int(v * 1000.0),
                value_fmt="{:.3f}",
            )

        self.sliders["specular_power"] = make_labeled_slider_row(
            parent=self.group_phong,
            layout=gl_phong,
            name="specular_power",
            state_key="specular_power",
            value_min=1,
            value_max=100,
            value_init=self.state["specular_power"],
            tick_to_value=lambda t: float(t / 10.0),
            value_to_tick=lambda v: int(v * 10.0),
            value_fmt="{:.1f}",
        )

        self.group_pbr = QtWidgets.QGroupBox("PBR Lighting", self)
        gl_pbr = QtWidgets.QVBoxLayout(self.group_pbr)
        self.layout.addWidget(self.group_pbr)

        for key in ("metallic", "roughness"):
            self.sliders[key] = make_labeled_slider_row(
                parent=self.group_pbr,
                layout=gl_pbr,
                name=key,
                state_key=key,
                value_min=0,
                value_max=1,
                value_init=self.state[key],
                tick_to_value=lambda t: float(t / 1000.0),
                value_to_tick=lambda v: int(v * 1000.0),
                value_fmt="{:.3f}",
            )

        self._set_shading_type(self.state["shading_type"], is_commit=False)

    def _wire_default_connections(self):
        self.chk_phong.stateChanged.connect(
            lambda _state: self._on_toggle_shading_type("phong")
        )
        self.chk_pbr.stateChanged.connect(
            lambda _state: self._on_toggle_shading_type("pbr")
        )
        for item in self.sliders.values():
            item.slider.valueChanged.connect(self.on_changed)

    def on_changed(self, _value=0):
        for item in self.sliders.values():
            item.sync_to_state(self.state)
        self.commit()

    def commit(self):
        params = {
            "shading_type": self.state["shading_type"],
            "ambient": float(self.state["ambient"]),
            "diffuse": float(self.state["diffuse"]),
            "specular": float(self.state["specular"]),
            "specular_power": float(self.state["specular_power"]),
            "specular_color": (
                float(self.state["specular_color_r"]),
                float(self.state["specular_color_g"]),
                float(self.state["specular_color_b"]),
            ),
            "metallic": float(self.state["metallic"]),
            "roughness": float(self.state["roughness"]),
        }
        self._is_gui_updating = True
        try:
            self.host.act_commit(**params)
        finally:
            self._is_gui_updating = False

    def _set_shading_type(self, shading_type, *, is_commit):
        shading_type = "pbr" if str(shading_type) == "pbr" else "phong"
        self.state["shading_type"] = shading_type

        with QSignalBlocker(self.chk_phong):
            self.chk_phong.setChecked(shading_type == "phong")
        with QSignalBlocker(self.chk_pbr):
            self.chk_pbr.setChecked(shading_type == "pbr")

        is_phong = shading_type == "phong"
        self.group_phong.setEnabled(is_phong)
        self.group_pbr.setEnabled(not is_phong)
        for key in (
            "specular_color_r",
            "specular_color_g",
            "specular_color_b",
        ):
            self.sliders[key].set_enabled(is_phong)

        if is_commit:
            self.commit()

    def _on_toggle_shading_type(self, shading_type):
        sender = self.sender()
        if sender is self.chk_phong and not self.chk_phong.isChecked():
            self._set_shading_type(self.state["shading_type"], is_commit=False)
            return
        if sender is self.chk_pbr and not self.chk_pbr.isChecked():
            self._set_shading_type(self.state["shading_type"], is_commit=False)
            return
        self._set_shading_type(shading_type, is_commit=True)

    def _sync_func(self, **kwargs):
        if self._is_gui_updating:
            return

        if "shading_type" in kwargs:
            self._set_shading_type(self.host.opts.shading_type, is_commit=False)

        for key in (
            "ambient",
            "diffuse",
            "specular",
            "specular_power",
            "metallic",
            "roughness",
        ):
            if key not in kwargs:
                continue
            self.state[key] = float(getattr(self.host.opts, key))
            self.sliders[key].set_tick(self.state[key], is_block_signals=True)

        if "specular_color" in kwargs:
            specular_color = np.asarray(self.host.opts.specular_color, dtype=float)
            for ch, value in zip(("r", "g", "b"), specular_color, strict=True):
                key = f"specular_color_{ch}"
                self.state[key] = float(value)
                self.sliders[key].set_tick(self.state[key], is_block_signals=True)

    def closeEvent(self, event):
        try:
            self.host.act_detach_sync_task(self._sync_name)
            self.parent_panel._lighting_console = None
        finally:
            event.accept()


class InteractGlyphBase(PanelBase):
    # -------------------------------
    # Helper marker management
    # -------------------------------

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
        if pm is None:
            return
        pm.act_remove_helper_marker(self._helper_panel_marker_key())

    # -------------------------------
    # Initialization
    # -------------------------------

    # ==================== OVERRIDE ====================
    # InteractGlyphBase overrides PanelBase.__init__ to install glyph-specific
    # configuration flags and the first-point helper marker workflow.
    # ==================================================

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

        display_title = title or f"Control of {host}"
        super().__init__(host, figure, title=display_title)
        self._helper_update_first_point_marker()

    # -------------------------------
    # UI construction hooks
    # -------------------------------

    # ==================== OVERRIDE ====================
    # InteractGlyphBase overrides PanelBase.build_ui to construct the shared
    # glyph-control widgets before subclass-specific extra groups are added.
    # ==================================================

    def build_ui(self):
        # ----------------------------
        # Geometry Group
        # ----------------------------

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

        # ----------------------------
        # RGB Group
        # ----------------------------
        if self.config["is_color"]:

            init_rgb = self.host.calc_color[0]
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
                "Use controlled opacity", group_opacity
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

        for item in self.sliders.values():
            if item not in self._custom_sliders:
                item.slider.valueChanged.connect(self.on_changed)
            item.slider.sliderPressed.connect(self._on_slider_pressed)
            item.slider.sliderReleased.connect(self._on_slider_released)

    def _build_extra_geometry(self, parent, layout):
        pass

    def _build_extra_group(self):
        pass

    def _open_lighting_console(self):
        if getattr(self, "_lighting_console", None) is None:
            self._lighting_console = LightingConsole(self.host, self)
        self._lighting_console.show()
        self._lighting_console.raise_()
        self._lighting_console.activateWindow()

    # -------------------------------
    # Radius and silhouette helpers
    # -------------------------------

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
            radius_index = source_index
            raw_coords = getattr(self.host, "raw_coords", None)
            if raw_coords is not None and len(radius_all) == 2 * len(raw_coords):
                radius_index = 2 * source_index
            return float(radius_all[radius_index]), source_index

        return float(radius_all[0]), 0

    def _update_radius_label(self):
        if not hasattr(self, "lbl_radius"):
            return

        radius, source_index = self._helper_get_first_used_point_radius()
        if radius is None:
            self.lbl_radius.setText("No currently used point is available.")
            return

        self.lbl_radius.setText(f"Radius at the red helper marker: {radius:.2f}")

    def _set_host_silhouette_enabled(self, is_enabled):
        if hasattr(self.host, "state_is_silhouette"):
            object.__setattr__(self.host, "state_is_silhouette", bool(is_enabled))

    def _on_slider_pressed(self):
        self._set_host_silhouette_enabled(False)
        if hasattr(self.host, "_helper_clear_silhouette"):
            self.host._helper_clear_silhouette()

    def _on_slider_released(self):
        self._set_host_silhouette_enabled(True)
        if hasattr(self.host, "_helper_add_silhouette"):
            self.host._helper_add_silhouette()

    # -------------------------------
    # Commit pipeline
    # -------------------------------

    def _helper_build_commit_params(self):
        params = {}
        if self.config["is_radius"]:
            current_radius = self.host.opts.radius
            scale = float(self.state["radius_rescale"])
            if callable(current_radius):
                params["radius"] = lambda x: scale * current_radius(x)
            elif np.isscalar(current_radius):
                params["radius"] = scale * float(current_radius)
            else:
                params["radius"] = scale * np.asarray(current_radius, dtype=float)
        if self.config["is_color"]:
            if self.state.get("is_use_control_color"):
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
                if self.state.get("is_use_control_opacity")
                else self.host.opts.opacity
            )
        if self.config["is_sides"]:
            params["sides"] = int(self.state["sides"])
        self._extra_commit(params)
        return params

    def _helper_run_commit(self, params):
        self.host.act_commit(**params)

    # ==================== OVERRIDE ====================
    # InteractGlyphBase overrides PanelBase.commit to translate UI state into
    # glyph-specific commit parameters for the host itself. Any follow-up UI
    # refresh is handled in `_sync_func`, regardless of whether the change
    # came from this panel or from an external command-line commit.
    # ==================================================

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

    def _on_toggle_bounds_enabled(self, _):
        is_enabled = self.chk_is_bounds_enabled.isChecked()
        self.state["is_bounds_enabled"] = is_enabled
        if self._is_block_chk_commit:
            return
        if is_enabled:
            self.host.act_bounds_enable()
        else:
            self.host.act_bounds_disable()

    # ==================== OVERRIDE ====================
    # InteractGlyphBase overrides PanelBase._sync_func so live host updates can
    # be reflected back into the panel widgets and helper marker state.
    # ==================================================

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
                self.sliders["radius_rescale"].set_tick(1, is_block_signals=True)
            self._update_radius_label()

        update_length_label = getattr(self, "_update_length_label", None)
        if callable(update_length_label) and "length" in kwargs:
            if not is_gui_updating and "length_rescale" in self.sliders:
                self.sliders["length_rescale"].set_tick(1, is_block_signals=True)
            update_length_label()

        self._helper_update_first_point_marker()

    # -------------------------------
    # Panel lifecycle
    # -------------------------------

    # ==================== OVERRIDE ====================
    # InteractGlyphBase overrides PanelBase.on_close to remove the helper
    # marker before the shared panel cleanup runs.
    # ==================================================

    def on_close(self):
        if getattr(self, "_lighting_console", None) is not None:
            self._lighting_console.close()
        self._helper_remove_first_point_marker()
        super().on_close()
