import numpy as np
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import QSignalBlocker

from .panel_base import make_labeled_slider_row, make_RGB_slider
from .ui_throttle import UIThrottle


class LightingConsole(QtWidgets.QWidget):
    def __init__(self, host, parent_panel):
        super().__init__(parent_panel)
        self.host = host
        self.parent_panel = parent_panel
        self.sliders = {}
        self.state = {}
        self._is_gui_updating = False
        self._sync_name = f"{parent_panel.str_now}::lighting"
        self._slider_throttle = UIThrottle(
            interval_ms=int(getattr(parent_panel, "slider_throttle_ms", 20)),
            parent=self,
        )

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
            item.slider.valueChanged.connect(self._schedule_on_changed)
            item.slider.sliderReleased.connect(self._flush_on_changed)

    def on_changed(self, _value=0):
        for item in self.sliders.values():
            item.sync_to_state(self.state)
        self.commit()

    def _schedule_on_changed(self, _value=0):
        self._slider_throttle.schedule(self.on_changed)

    def _flush_on_changed(self, _value=0):
        self._slider_throttle.flush()

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
            self._slider_throttle.cancel()
            self.host.act_detach_sync_task(self._sync_name)
            self.parent_panel._lighting_console = None
        finally:
            event.accept()
