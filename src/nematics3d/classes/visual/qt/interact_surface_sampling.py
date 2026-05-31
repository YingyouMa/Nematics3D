import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QSignalBlocker

from ...surface_sampling import (
    SurfaceSampling,
    _helper_resolve_spacing_for_target_count,
)
from .panel_base import PanelBase, LogTickMapper, make_labeled_slider_row


class InteractSurfaceSampling(PanelBase):
    # ==================== OVERRIDE ====================
    # InteractSurfaceSampling overrides PanelBase.__init__ because this panel
    # is specialized for SurfaceSampling hosts and also manages a helper marker
    # showing the first currently sampled point.
    # ==================================================
    def __init__(self, host, figure):
        if not isinstance(host, SurfaceSampling):
            raise TypeError(
                "InteractSurfaceSampling requires a SurfaceSampling host. "
                f"Got {type(host).__name__!r}."
            )

        self._is_block_chk_commit = False
        self._is_gui_updating = False
        super().__init__(
            host,
            figure,
            title=f"Surface Sampling Controls of {host.name!r}",
        )
        self._helper_update_first_point_marker()

    # -------------------------------
    # Helper marker management
    # -------------------------------

    def _helper_panel_marker_key(self):
        return f"{self.str_now}::calc0"

    def _helper_update_first_point_marker(self):
        pm = getattr(self.fig, "pick_manager", None) if self.fig is not None else None
        if pm is None:
            return

        coords = np.asarray(getattr(self.host, "calc_sample_points", ()), dtype=float)
        if coords.size == 0:
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
    # UI construction
    # -------------------------------

    def build_ui(self):
        spacing_current = self._helper_get_display_spacing()
        spacing_mapper = LogTickMapper(value_min=1.0e-3, value_max=1.0e3, base=10.0)

        self._custom_sliders = []
        self.state = {
            "spacing": spacing_current,
            "is_auto_spacing": self.host.opts.spacing is None,
        }

        group_spacing = QtWidgets.QGroupBox("Spacing", self)
        layout_spacing = QtWidgets.QVBoxLayout(group_spacing)
        self.layout.addWidget(group_spacing)

        self.chk_is_auto_spacing = QtWidgets.QCheckBox(
            "Automatic spacing from default_sample_count_target",
            group_spacing,
        )
        self.chk_is_auto_spacing.setChecked(self.state["is_auto_spacing"])
        layout_spacing.addWidget(self.chk_is_auto_spacing)
        self.chk_is_auto_spacing.stateChanged.connect(self._on_toggle_auto_spacing)

        self.sliders["spacing"] = make_labeled_slider_row(
            parent=group_spacing,
            layout=layout_spacing,
            name="spacing",
            state_key="spacing",
            value_min=spacing_mapper.value_min,
            value_max=spacing_mapper.value_max,
            value_init=spacing_current,
            tick_to_value=spacing_mapper.tick_to_value,
            value_to_tick=spacing_mapper.value_to_tick,
            value_fmt="{:.4f}",
            input_out_of_range="expand_max",
        )
        self._custom_sliders.append(self.sliders["spacing"])
        self.sliders["spacing"].set_enabled(not self.state["is_auto_spacing"])
        self.sliders["spacing"].slider.valueChanged.connect(
            lambda _value=0: self._on_spacing_slider_changed()
        )
        self.sliders["spacing"].value_box.editingFinished.connect(
            self._update_spacing_mode_label
        )

        self.lbl_spacing_mode = QtWidgets.QLabel(group_spacing)
        self.lbl_spacing_mode.setWordWrap(True)
        layout_spacing.addWidget(self.lbl_spacing_mode)

        group_params = QtWidgets.QGroupBox("Sampling Parameters", self)
        form = QtWidgets.QFormLayout(group_params)
        self.layout.addWidget(group_params)

        self.input_seed = QtWidgets.QSpinBox(group_params)
        self.input_seed.setRange(-2147483648, 2147483647)
        self.input_seed.setValue(int(self.host.opts.seed))
        form.addRow("seed:", self.input_seed)

        self.input_oversample = QtWidgets.QSpinBox(group_params)
        self.input_oversample.setRange(1, 1000000)
        self.input_oversample.setValue(int(self.host.opts.oversample))
        form.addRow("oversample:", self.input_oversample)

        self.input_relax_steps = QtWidgets.QSpinBox(group_params)
        self.input_relax_steps.setRange(0, 1000000)
        self.input_relax_steps.setValue(int(self.host.opts.relax_steps))
        form.addRow("relax_steps:", self.input_relax_steps)

        self.input_k_neighbors = QtWidgets.QSpinBox(group_params)
        self.input_k_neighbors.setRange(1, 1000000)
        self.input_k_neighbors.setValue(int(self.host.opts.k_neighbors))
        form.addRow("k_neighbors:", self.input_k_neighbors)

        self.input_default_target = QtWidgets.QSpinBox(group_params)
        self.input_default_target.setRange(1, 100000000)
        self.input_default_target.setValue(
            int(self.host.opts.default_sample_count_target)
        )
        form.addRow("default_sample_count_target:", self.input_default_target)

        group_apply = QtWidgets.QGroupBox("Apply", self)
        layout_apply = QtWidgets.QVBoxLayout(group_apply)
        self.layout.addWidget(group_apply)

        self.lbl_apply_hint = QtWidgets.QLabel(
            "Surface sampling is not updated live. Adjust parameters first, "
            "then click Apply once.",
            group_apply,
        )
        self.lbl_apply_hint.setWordWrap(True)
        layout_apply.addWidget(self.lbl_apply_hint)

        self.btn_apply = QtWidgets.QPushButton("Apply Sampling Changes", group_apply)
        self.btn_apply.clicked.connect(self.commit)
        layout_apply.addWidget(self.btn_apply)

        group_summary = QtWidgets.QGroupBox("Current Result", self)
        layout_summary = QtWidgets.QVBoxLayout(group_summary)
        self.layout.addWidget(group_summary)

        self.lbl_surface_area = QtWidgets.QLabel(group_summary)
        layout_summary.addWidget(self.lbl_surface_area)
        self.lbl_target_count = QtWidgets.QLabel(group_summary)
        layout_summary.addWidget(self.lbl_target_count)
        self.lbl_sample_count = QtWidgets.QLabel(group_summary)
        layout_summary.addWidget(self.lbl_sample_count)

        self._update_spacing_mode_label()
        self._update_summary_labels()

    # -------------------------------
    # Helpers
    # -------------------------------

    def _helper_get_display_spacing(self) -> float:
        if self.host.opts.spacing is not None:
            return max(float(self.host.opts.spacing), 1.0e-12)

        area = float(getattr(self.host, "calc_surface_area", 0.0))
        target = int(getattr(self.host.opts, "default_sample_count_target", 1))
        if area > 0.0:
            return max(
                float(
                    _helper_resolve_spacing_for_target_count(
                        area,
                        target,
                    )
                ),
                1.0e-12,
            )

        return 1.0

    def _update_spacing_mode_label(self):
        if hasattr(self, "chk_is_auto_spacing"):
            is_auto = bool(self.chk_is_auto_spacing.isChecked())
        else:
            is_auto = self.host.opts.spacing is None

        if hasattr(self, "sliders") and "spacing" in self.sliders:
            spacing_display = float(self.sliders["spacing"].value_box.value())
        else:
            spacing_display = self._helper_get_display_spacing()

        if is_auto:
            self.lbl_spacing_mode.setText(
                "Automatic mode is selected. Apply will resample using "
                "`default_sample_count_target`; the displayed spacing is only the "
                f"current estimate: {spacing_display:.4f}."
            )
            return

        self.lbl_spacing_mode.setText(
            "Manual mode is selected. Apply will resample using "
            f"spacing = {spacing_display:.4f}."
        )

    def _update_summary_labels(self):
        surface_area = float(getattr(self.host, "calc_surface_area", 0.0))
        target_count = int(getattr(self.host, "calc_sample_count_target", 0))
        sample_count = len(np.asarray(getattr(self.host, "calc_sample_points", ())))

        self.lbl_surface_area.setText(f"Surface area: {surface_area:.6g}")
        self.lbl_target_count.setText(f"Resolved target sample count: {target_count}")
        self.lbl_sample_count.setText(f"Current sampled point count: {sample_count}")

    def _sync_spinbox(self, box: QtWidgets.QSpinBox, value: int):
        with QSignalBlocker(box):
            box.setValue(int(value))

    # -------------------------------
    # Commit and synchronization
    # -------------------------------

    def _on_spacing_slider_changed(self):
        self.sliders["spacing"].set_label()
        self._update_spacing_mode_label()

    def _capture_ui_state(self):
        self.sliders["spacing"].apply_value_box_edit()
        self.state["spacing"] = self.sliders["spacing"].get_value()
        self.state["is_auto_spacing"] = self.chk_is_auto_spacing.isChecked()

    def commit(self):
        self._capture_ui_state()
        spacing = None
        if not bool(self.state.get("is_auto_spacing", False)):
            spacing = float(self.state["spacing"])

        self._is_gui_updating = True
        try:
            self.host.act_commit(
                spacing=spacing,
                seed=int(self.input_seed.value()),
                oversample=int(self.input_oversample.value()),
                relax_steps=int(self.input_relax_steps.value()),
                k_neighbors=int(self.input_k_neighbors.value()),
                default_sample_count_target=int(self.input_default_target.value()),
            )
        finally:
            self._is_gui_updating = False

    # ==================== OVERRIDE ====================
    # InteractSurfaceSampling overrides PanelBase._sync_func because this
    # panel must keep checkboxes, manual input widgets, result labels, and the
    # first-sample helper marker synchronized with host-side resampling.
    # ==================================================
    def _sync_func(self, **kwargs):
        is_auto = self.host.opts.spacing is None
        with QSignalBlocker(self.chk_is_auto_spacing):
            self.chk_is_auto_spacing.setChecked(is_auto)
        self.state["is_auto_spacing"] = is_auto

        spacing_display = self._helper_get_display_spacing()
        self._sync_from_host_slider("spacing", spacing_display)
        self.sliders["spacing"].set_enabled(not is_auto)

        if "seed" in kwargs:
            self._sync_spinbox(self.input_seed, int(self.host.opts.seed))
        if "oversample" in kwargs:
            self._sync_spinbox(self.input_oversample, int(self.host.opts.oversample))
        if "relax_steps" in kwargs:
            self._sync_spinbox(
                self.input_relax_steps,
                int(self.host.opts.relax_steps),
            )
        if "k_neighbors" in kwargs:
            self._sync_spinbox(
                self.input_k_neighbors,
                int(self.host.opts.k_neighbors),
            )
        if "default_sample_count_target" in kwargs:
            self._sync_spinbox(
                self.input_default_target,
                int(self.host.opts.default_sample_count_target),
            )

        self._update_spacing_mode_label()
        self._update_summary_labels()
        self._helper_update_first_point_marker()

    # -------------------------------
    # UI callbacks
    # -------------------------------

    def _on_toggle_auto_spacing(self, _state: int):
        is_auto = self.chk_is_auto_spacing.isChecked()
        self.state["is_auto_spacing"] = is_auto
        self.sliders["spacing"].set_enabled(not is_auto)
        self._update_spacing_mode_label()

    # ==================== OVERRIDE ====================
    # InteractSurfaceSampling overrides PanelBase.on_close because it must
    # remove the helper marker that highlights the first sampled point.
    # ==================================================
    def on_close(self):
        self._helper_remove_first_point_marker()
        super().on_close()
