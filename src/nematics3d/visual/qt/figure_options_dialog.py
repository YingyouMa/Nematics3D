"""Qt dialog for live PlotFigure camera/background options."""

import datetime

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from nematics3d.visual.qt.panel_base import (
    make_RGB_slider,
    make_labeled_slider_row,
)


class FigureOptionsDialog(QtWidgets.QDialog):
    """Non-modal live editor and snapshot browser for PlotFigure options."""

    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self.figure = figure
        self._is_gui_updating = False
        self._sliders: dict[str, object] = {}
        self._snapshots: dict[str, dict[str, object]] = {}
        self._snapshot_names_saved_from_dialog: list[str] = []
        self._opts_snapshot_dialog = None
        self._sync_task_name = self._helper_make_snapshot_name("figure_opts_sync")
        self._original_snapshot_name = self._helper_make_snapshot_name(
            "figure_opts_initial"
        )

        self.setWindowTitle("PlotFigure Options")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(10)
        self._build_ui()
        self.adjustSize()
        self._helper_save_snapshot(self._original_snapshot_name, is_user_snapshot=False)
        self._sync_from_opts()
        if self.figure is not None:
            self.figure.act_attach_sync_task(
                self._sync_task_name,
                self._sync_from_figure,
            )

    def _build_ui(self):
        group_camera = QtWidgets.QGroupBox("Camera", self)
        layout_camera = QtWidgets.QVBoxLayout(group_camera)
        self.layout.addWidget(group_camera)

        slider_specs = {
            "azimuth": (0, 360, "Azimuth"),
            "elevation": (-90, 90, "Elevation"),
            "roll": (-180, 180, "Roll"),
        }
        for key, (value_min, value_max, label) in slider_specs.items():
            self._sliders[key] = make_labeled_slider_row(
                parent=group_camera,
                layout=layout_camera,
                name=label,
                state_key=key,
                value_min=value_min,
                value_max=value_max,
                value_init=0,
                tick_to_value=lambda t: t / 10,
                value_to_tick=lambda v: int(round(v * 10)),
                value_fmt="{:.2f}",
            )
            self._sliders[key].slider.valueChanged.connect(
                self._apply_camera_slider_changes
            )

        self.panel_distance, self.distance_input = self._make_scalar_apply_row(
            parent=group_camera,
            layout=layout_camera,
            title="Distance",
            value=0.0,
            value_min=0.0,
            value_max=1.0e12,
            decimals=2,
        )
        self.distance_input.valueChanged.connect(self._apply_distance)

        (
            self.panel_focal_point,
            self.focal_inputs,
            self.btn_focal_apply,
        ) = self._make_vector_apply_row(
            parent=group_camera,
            layout=layout_camera,
            title="Focal Point",
            values=(0.0, 0.0, 0.0),
            callback=self._apply_focal_point,
        )

        group_background = QtWidgets.QGroupBox("Background", self)
        layout_background = QtWidgets.QVBoxLayout(group_background)
        self.layout.addWidget(group_background)
        make_RGB_slider(
            parent=group_background,
            layout=layout_background,
            sliders=self._sliders,
            prefix="bg_color",
            init_rgb=(1.0, 1.0, 1.0),
            value_fmt="{:.2f}",
        )
        for key in ("bg_color_r", "bg_color_g", "bg_color_b"):
            self._sliders[key].slider.valueChanged.connect(self._apply_bg_color_changes)

        group_snapshot = QtWidgets.QGroupBox("Save / Load", self)
        layout_snapshot = QtWidgets.QGridLayout(group_snapshot)
        self.layout.addWidget(group_snapshot)
        buttons = (
            ("Save Current", self._on_save_current_snapshot, 0, 0),
            ("Restore Original", self._on_restore_original_snapshot, 0, 1),
            ("Load Latest Save", self._on_load_latest_snapshot, 1, 0),
            ("Load Saved...", self._on_choose_snapshot_to_restore, 1, 1),
        )
        for label, callback, row, col in buttons:
            button = QtWidgets.QPushButton(label, group_snapshot)
            button.clicked.connect(callback)
            layout_snapshot.addWidget(button, row, col)

        self.btn_show_opts_snapshot = QtWidgets.QPushButton("Show Current Opts", self)
        self.btn_show_opts_snapshot.clicked.connect(self._open_opts_snapshot_dialog)
        self.layout.addWidget(self.btn_show_opts_snapshot)

    @staticmethod
    def _make_scalar_apply_row(
        *, parent, layout, title, value, value_min, value_max, decimals
    ):
        panel = QtWidgets.QWidget(parent)
        panel_layout = QtWidgets.QHBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        panel_layout.addWidget(QtWidgets.QLabel(f"{title}:", panel))
        box = QtWidgets.QDoubleSpinBox(panel)
        box.setDecimals(int(decimals))
        box.setKeyboardTracking(False)
        box.setRange(float(value_min), float(value_max))
        box.setValue(float(value))
        panel_layout.addWidget(box)
        layout.addWidget(panel)
        return panel, box

    @staticmethod
    def _make_vector_apply_row(*, parent, layout, title, values, callback):
        panel = QtWidgets.QWidget(parent)
        panel_layout = QtWidgets.QHBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        panel_layout.addWidget(QtWidgets.QLabel(f"{title}:", panel))
        inputs = []
        for value in np.asarray(values, dtype=float):
            box = QtWidgets.QDoubleSpinBox(panel)
            box.setDecimals(2)
            box.setKeyboardTracking(False)
            box.setRange(-1.0e12, 1.0e12)
            box.setValue(float(value))
            panel_layout.addWidget(box)
            inputs.append(box)
        button = QtWidgets.QPushButton("Apply", panel)
        button.clicked.connect(lambda: callback(inputs))
        panel_layout.addWidget(button)
        layout.addWidget(panel)
        return panel, inputs, button

    @staticmethod
    def _helper_make_snapshot_name(prefix: str = "figure_opts") -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"{prefix}_{stamp}"

    def _helper_collect_current_payload(self) -> dict[str, object]:
        if self.figure is None or not self.figure.is_alive:
            return {}
        opts = self.figure.opts
        return {
            "azimuth": float(opts.azimuth),
            "elevation": float(opts.elevation),
            "roll": float(opts.roll),
            "distance": float(opts.distance),
            "focal_point": np.asarray(opts.focal_point, dtype=float).copy(),
            "bg_color": np.asarray(opts.bg_color, dtype=float).copy(),
        }

    def _helper_save_snapshot(self, name: str, *, is_user_snapshot: bool):
        payload = self._helper_collect_current_payload()
        if not payload:
            return
        self._snapshots[name] = payload
        if is_user_snapshot:
            self._snapshot_names_saved_from_dialog.append(name)

    def _helper_restore_snapshot(self, name: str):
        payload = self._snapshots.get(name)
        if payload is None:
            raise KeyError(f"Snapshot {name!r} is not available.")
        self.figure.act_commit(**payload)

    def _helper_get_snapshot_choice_entries(self):
        return [
            (
                f"{name} (initial)" if name == self._original_snapshot_name else name,
                name,
            )
            for name in self._snapshots
        ]

    def _sync_slider_value(self, key, value):
        slider = self._sliders[key]
        if slider.slider.isSliderDown() or slider.value_box.hasFocus():
            return
        slider.set_tick(float(value), is_block_signals=True)

    @staticmethod
    def _sync_scalar_box(box, value):
        if box.hasFocus():
            return
        box.blockSignals(True)
        try:
            box.setValue(float(value))
        finally:
            box.blockSignals(False)

    @staticmethod
    def _sync_vector_boxes(boxes, values):
        for box, value in zip(boxes, np.asarray(values, dtype=float), strict=True):
            if box.hasFocus():
                continue
            box.blockSignals(True)
            try:
                box.setValue(float(value))
            finally:
                box.blockSignals(False)

    def _sync_from_opts(self):
        if self._is_gui_updating:
            return
        payload = self._helper_collect_current_payload()
        if not payload:
            return
        self._is_gui_updating = True
        try:
            for key in ("azimuth", "elevation", "roll"):
                self._sync_slider_value(key, payload[key])
            self._sync_scalar_box(self.distance_input, payload["distance"])
            self._sync_vector_boxes(self.focal_inputs, payload["focal_point"])
            for key, value in zip(
                ("bg_color_r", "bg_color_g", "bg_color_b"),
                np.asarray(payload["bg_color"], dtype=float),
                strict=True,
            ):
                self._sync_slider_value(key, value)
        finally:
            self._is_gui_updating = False

    def _sync_from_figure(self, **kwargs):
        relevant = {
            "azimuth",
            "elevation",
            "roll",
            "distance",
            "focal_point",
            "bg_color",
        }
        if kwargs and any(key in kwargs for key in relevant):
            self._sync_from_opts()

    def _apply_camera_slider_changes(self, *_args):
        if self._is_gui_updating:
            return
        for key in ("azimuth", "elevation", "roll"):
            self._sliders[key].set_label()
        self.figure.act_commit(
            **{
                key: float(self._sliders[key].get_value())
                for key in ("azimuth", "elevation", "roll")
            }
        )

    def _apply_bg_color_changes(self, *_args):
        if self._is_gui_updating:
            return
        keys = ("bg_color_r", "bg_color_g", "bg_color_b")
        for key in keys:
            self._sliders[key].set_label()
        self.figure.act_commit(
            bg_color=tuple(float(self._sliders[key].get_value()) for key in keys)
        )

    def _apply_distance(self, *_args):
        if not self._is_gui_updating:
            self.figure.act_commit(distance=float(self.distance_input.value()))

    def _apply_focal_point(self, boxes):
        self.figure.act_commit(
            focal_point=np.array([box.value() for box in boxes], dtype=float)
        )

    def _helper_build_opts_snapshot_text(self):
        if self.figure is None or not self.figure.is_alive:
            return "PlotFigure is no longer available."
        return repr(self.figure.opts)

    def _open_opts_snapshot_dialog(self):
        if self._opts_snapshot_dialog is not None:
            self._opts_snapshot_dialog.close()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Current PlotFigure Opts Snapshot")
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(
            "This text is a one-time snapshot when the window opens. It does not update live.",
            dialog,
        )
        label.setWordWrap(True)
        label_font = QtGui.QFont(label.font())
        label_font.setPointSize(max(label_font.pointSize(), 12))
        label.setFont(label_font)
        layout.addWidget(label)
        text = QtWidgets.QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setPointSize(13)
        text.setFont(font)
        text.setPlainText(self._helper_build_opts_snapshot_text())
        layout.addWidget(text)
        dialog.destroyed.connect(
            lambda *_: setattr(self, "_opts_snapshot_dialog", None)
        )
        self._opts_snapshot_dialog = dialog
        dialog.resize(760, 540)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_save_current_snapshot(self):
        self._helper_save_snapshot(
            self._helper_make_snapshot_name(), is_user_snapshot=True
        )

    def _on_restore_original_snapshot(self):
        self._helper_restore_snapshot(self._original_snapshot_name)

    def _on_load_latest_snapshot(self):
        if not self._snapshot_names_saved_from_dialog:
            QtWidgets.QMessageBox.information(
                self,
                "No Saved Snapshot",
                "No dialog-created snapshot is available yet.",
            )
            return
        self._helper_restore_snapshot(self._snapshot_names_saved_from_dialog[-1])

    def _on_choose_snapshot_to_restore(self):
        entries = self._helper_get_snapshot_choice_entries()
        if not entries:
            QtWidgets.QMessageBox.information(
                self, "No Saved Snapshot", "No snapshot is available for this dialog."
            )
            return
        labels = [label for label, _ in entries]
        selected, is_ok = QtWidgets.QInputDialog.getItem(
            self,
            "Load Saved Snapshot",
            "Choose one snapshot to restore:",
            labels,
            0,
            False,
        )
        if is_ok:
            self._helper_restore_snapshot(dict(entries)[str(selected)])

    def closeEvent(self, event):
        try:
            if self.figure is not None:
                self.figure.act_detach_sync_task(self._sync_task_name)
            if self._opts_snapshot_dialog is not None:
                self._opts_snapshot_dialog.close()
        finally:
            super().closeEvent(event)


__all__ = ["FigureOptionsDialog"]
