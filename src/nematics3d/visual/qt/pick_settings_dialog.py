"""Qt dialog for configuring :class:`PickManager` interaction options."""

from qtpy import QtWidgets


class PickSettingsDialog(QtWidgets.QDialog):
    """Non-modal editor for one PickManager's live options."""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Interaction Settings")

        layout = QtWidgets.QVBoxLayout(self)

        group_panel = QtWidgets.QGroupBox("Panel", self)
        form_panel = QtWidgets.QFormLayout(group_panel)
        layout.addWidget(group_panel)

        self.spin_throttle = QtWidgets.QSpinBox(self)
        self.spin_throttle.setRange(1, 1000)
        self.spin_throttle.setSingleStep(5)
        self.spin_throttle.setValue(int(manager.opts.slider_throttle_ms))
        form_panel.addRow("Slider throttle (ms)", self.spin_throttle)

        self.spin_double_click = QtWidgets.QDoubleSpinBox(self)
        self.spin_double_click.setRange(0.01, 10.0)
        self.spin_double_click.setSingleStep(0.05)
        self.spin_double_click.setDecimals(3)
        self.spin_double_click.setValue(float(manager.opts.double_click_threshold))
        form_panel.addRow("Double click threshold", self.spin_double_click)

        group_marker = QtWidgets.QGroupBox("Marker", self)
        form_marker = QtWidgets.QFormLayout(group_marker)
        layout.addWidget(group_marker)

        self.spin_marker_proximity = QtWidgets.QDoubleSpinBox(self)
        self.spin_marker_proximity.setRange(0.0, 1000.0)
        self.spin_marker_proximity.setSingleStep(0.05)
        self.spin_marker_proximity.setDecimals(3)
        self.spin_marker_proximity.setValue(
            float(manager.opts.marker_proximity_threshold)
        )
        form_marker.addRow("Proximity threshold", self.spin_marker_proximity)

        self.spin_marker_size = QtWidgets.QSpinBox(self)
        self.spin_marker_size.setRange(1, 200)
        self.spin_marker_size.setValue(int(manager.opts.marker_size))
        form_marker.addRow("Size", self.spin_marker_size)

        self.spin_marker_font_size = QtWidgets.QSpinBox(self)
        self.spin_marker_font_size.setRange(1, 200)
        self.spin_marker_font_size.setValue(int(manager.opts.marker_font_size))
        form_marker.addRow("Font size", self.spin_marker_font_size)

        self.marker_color_inputs = self._make_rgb_row(
            form_marker, "Color (r g b)", manager.opts.marker_color
        )

        group_silhouette = QtWidgets.QGroupBox("Silhouette", self)
        form_silhouette = QtWidgets.QFormLayout(group_silhouette)
        layout.addWidget(group_silhouette)

        self.spin_sil_opacity = QtWidgets.QDoubleSpinBox(self)
        self.spin_sil_opacity.setRange(0.0, 1.0)
        self.spin_sil_opacity.setSingleStep(0.05)
        self.spin_sil_opacity.setDecimals(3)
        self.spin_sil_opacity.setValue(float(manager.opts.sil_opacity))
        form_silhouette.addRow("Opacity", self.spin_sil_opacity)

        self.spin_sil_width = QtWidgets.QDoubleSpinBox(self)
        self.spin_sil_width.setRange(0.0, 1000.0)
        self.spin_sil_width.setSingleStep(0.5)
        self.spin_sil_width.setDecimals(3)
        self.spin_sil_width.setValue(float(manager.opts.sil_width))
        form_silhouette.addRow("Width", self.spin_sil_width)

        self.sil_color_inputs = self._make_rgb_row(
            form_silhouette, "Color (r g b)", manager.opts.sil_color
        )

        buttons = QtWidgets.QDialogButtonBox(parent=self)
        btn_ok = buttons.addButton(QtWidgets.QDialogButtonBox.Ok)
        btn_apply = buttons.addButton(QtWidgets.QDialogButtonBox.Apply)
        btn_cancel = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        btn_ok.clicked.connect(lambda: (self.apply_settings(), self.accept()))
        btn_apply.clicked.connect(self.apply_settings)
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(buttons)

        self.setModal(False)

    def _make_rgb_row(self, form, title, values):
        widget = QtWidgets.QWidget(self)
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        inputs = []
        for value in values:
            spin = QtWidgets.QDoubleSpinBox(widget)
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(3)
            spin.setMinimumWidth(80)
            spin.setValue(float(value))
            row.addWidget(spin)
            inputs.append(spin)
        form.addRow(title, widget)
        return tuple(inputs)

    def apply_settings(self):
        """Commit current widget values to the manager's live options."""
        opts = self.manager.opts
        opts.slider_throttle_ms = int(self.spin_throttle.value())
        opts.double_click_threshold = float(self.spin_double_click.value())
        opts.marker_proximity_threshold = float(self.spin_marker_proximity.value())
        opts.marker_size = int(self.spin_marker_size.value())
        opts.marker_font_size = int(self.spin_marker_font_size.value())
        opts.marker_color = tuple(
            float(spin.value()) for spin in self.marker_color_inputs
        )
        opts.sil_opacity = float(self.spin_sil_opacity.value())
        opts.sil_width = float(self.spin_sil_width.value())
        opts.sil_color = tuple(float(spin.value()) for spin in self.sil_color_inputs)


__all__ = ["PickSettingsDialog"]
