"""Qt controls for live windowed-sinc surface smoothing."""

from __future__ import annotations

from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from nematics3d.geometry.smoothing.surface import SmoothedSurface
from nematics3d.visual.qt.panel_base import PanelBase, make_labeled_slider_row


class InteractSmoothedSurface(PanelBase):
    """Live controls for a :class:`SmoothedSurface`."""

    def __init__(self, host, figure):
        if not isinstance(host, SmoothedSurface):
            raise TypeError(
                "InteractSmoothedSurface requires a SmoothedSurface host. "
                f"Got {type(host).__name__!r}."
            )
        super().__init__(
            host,
            figure,
            title=f"Smoothed Surface Controls of {host.name!r}",
        )

    def build_ui(self):
        self.state = {
            "pass_band": float(self.host.opts.pass_band),
            "feature_angle": float(self.host.opts.feature_angle),
            "edge_angle": float(self.host.opts.edge_angle),
        }

        group_main = QtWidgets.QGroupBox("Smoothing", self)
        layout_main = QtWidgets.QVBoxLayout(group_main)
        self.layout.addWidget(group_main)

        self.sliders["pass_band"] = make_labeled_slider_row(
            parent=group_main,
            layout=layout_main,
            name="pass_band",
            value_min=0.0,
            value_max=2.0,
            value_init=self.state["pass_band"],
            tick_to_value=lambda t: t / 1000.0,
            value_to_tick=lambda v: int(round(v * 1000.0)),
            value_fmt="{:.3f}",
            single_step=1,
            page_step=20,
        )

        self.input_n_iter = QtWidgets.QSpinBox(group_main)
        self.input_n_iter.setRange(0, 1000000)
        self.input_n_iter.setValue(int(self.host.opts.n_iter))
        self.input_n_iter.setKeyboardTracking(False)
        row_n_iter = QtWidgets.QFormLayout()
        row_n_iter.addRow("n_iter:", self.input_n_iter)
        layout_main.addLayout(row_n_iter)
        self.input_n_iter.valueChanged.connect(self._on_n_iter_changed)

        group_features = QtWidgets.QGroupBox("Edges and Features", self)
        layout_features = QtWidgets.QVBoxLayout(group_features)
        self.layout.addWidget(group_features)

        self.chk_boundary_smoothing = QtWidgets.QCheckBox(
            "Smooth boundary vertices", group_features
        )
        self.chk_boundary_smoothing.setChecked(bool(self.host.opts.boundary_smoothing))
        self.chk_boundary_smoothing.stateChanged.connect(self._on_checkbox_changed)
        layout_features.addWidget(self.chk_boundary_smoothing)

        self.chk_feature_smoothing = QtWidgets.QCheckBox(
            "Enable feature-edge smoothing", group_features
        )
        self.chk_feature_smoothing.setChecked(bool(self.host.opts.feature_smoothing))
        self.chk_feature_smoothing.stateChanged.connect(self._on_checkbox_changed)
        layout_features.addWidget(self.chk_feature_smoothing)

        for key in ("feature_angle", "edge_angle"):
            self.sliders[key] = make_labeled_slider_row(
                parent=group_features,
                layout=layout_features,
                name=key,
                value_min=0.0,
                value_max=180.0,
                value_init=self.state[key],
                tick_to_value=lambda t: t / 10.0,
                value_to_tick=lambda v: int(round(v * 10.0)),
                value_fmt="{:.1f}",
                single_step=1,
                page_step=10,
            )

        self.chk_normalize_coordinates = QtWidgets.QCheckBox(
            "Normalize coordinates during smoothing", group_features
        )
        self.chk_normalize_coordinates.setChecked(
            bool(self.host.opts.normalize_coordinates)
        )
        self.chk_normalize_coordinates.stateChanged.connect(self._on_checkbox_changed)
        layout_features.addWidget(self.chk_normalize_coordinates)

        hint = QtWidgets.QLabel(
            "Sliders update the smoothed mesh live. Updates are throttled while dragging.",
            self,
        )
        hint.setWordWrap(True)
        self.layout.addWidget(hint)

    def commit(self):
        self.host.act_commit(
            pass_band=float(self.state["pass_band"]),
            n_iter=int(self.input_n_iter.value()),
            boundary_smoothing=self.chk_boundary_smoothing.isChecked(),
            feature_smoothing=self.chk_feature_smoothing.isChecked(),
            feature_angle=float(self.state["feature_angle"]),
            edge_angle=float(self.state["edge_angle"]),
            normalize_coordinates=self.chk_normalize_coordinates.isChecked(),
        )

    def _on_n_iter_changed(self, _value: int):
        self.commit()

    def _on_checkbox_changed(self, _state: int):
        self.commit()

    def _sync_func(self, **kwargs):
        for key in ("pass_band", "feature_angle", "edge_angle"):
            if key in kwargs:
                self._sync_from_host_slider(key, float(getattr(self.host.opts, key)))

        if "n_iter" in kwargs:
            with QSignalBlocker(self.input_n_iter):
                self.input_n_iter.setValue(int(self.host.opts.n_iter))

        checkbox_attrs = {
            "boundary_smoothing": self.chk_boundary_smoothing,
            "feature_smoothing": self.chk_feature_smoothing,
            "normalize_coordinates": self.chk_normalize_coordinates,
        }
        for key, checkbox in checkbox_attrs.items():
            if key in kwargs:
                with QSignalBlocker(checkbox):
                    checkbox.setChecked(bool(getattr(self.host.opts, key)))


__all__ = ["InteractSmoothedSurface"]
