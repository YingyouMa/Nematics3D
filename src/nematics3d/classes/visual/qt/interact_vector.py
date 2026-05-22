import numpy as np
from qtpy import QtWidgets

from .interact_glyph_base import InteractGlyphBase
from .panel_base import make_labeled_slider_row, LogTickMapper


class InteractVector(InteractGlyphBase):
    # ==================== OVERRIDE ====================
    # InteractVector overrides InteractGlyphBase.__init__ to enable
    # standard glyph controls while extending geometry with vector-specific
    # length and arrow-tip controls.
    # ==================================================
    def __init__(self, host, figure):
        super().__init__(
            host,
            figure,
            title=f"Vector Controls of {host.name!r}",
            is_radius=True,
            is_sides=True,
            is_geometry=True,
            is_color=True,
            is_opacity=True,
        )

    def _build_extra_geometry(self, parent, layout):
        self.state["length_rescale"] = 1.0
        self._length_rescale_base = self._helper_clone_resolver_value(
            self.host.opts.length
        )
        log_mapper = LogTickMapper(value_min=0.2, value_max=5, base=10.0)
        self.sliders["length_rescale"] = make_labeled_slider_row(
            parent=parent,
            layout=layout,
            name="length_rescale",
            state_key="length_rescale",
            value_min=log_mapper.value_min,
            value_max=log_mapper.value_max,
            value_init=1.0,
            tick_to_value=log_mapper.tick_to_value,
            value_to_tick=log_mapper.value_to_tick,
            input_out_of_range="expand_max",
        )

        self.state["tip_length_fraction"] = float(self.host.opts.tip_length_fraction)
        self.sliders["tip_length_fraction"] = make_labeled_slider_row(
            parent=parent,
            layout=layout,
            name="tip_length_fraction",
            state_key="tip_length_fraction",
            value_min=0.01,
            value_max=0.8,
            value_init=self.state["tip_length_fraction"],
            tick_to_value=lambda t: float(t / 1000.0),
            value_to_tick=lambda v: int(round(v * 1000.0)),
            value_fmt="{:.2f}",
        )

        self.state["tip_radius_ratio"] = float(self.host.opts.tip_radius_ratio)
        self.sliders["tip_radius_ratio"] = make_labeled_slider_row(
            parent=parent,
            layout=layout,
            name="tip_radius_ratio",
            state_key="tip_radius_ratio",
            value_min=1.0,
            value_max=8.0,
            value_init=self.state["tip_radius_ratio"],
            tick_to_value=lambda t: float(t / 100.0),
            value_to_tick=lambda v: int(round(v * 100.0)),
            value_fmt="{:.2f}",
        )

        self.state["is_anchor_tail"] = self.host.opts.anchor == "tail"
        self.chk_is_anchor_tail = QtWidgets.QCheckBox(
            "Use tail as vector anchor",
            parent,
        )
        self.chk_is_anchor_tail.setChecked(self.state["is_anchor_tail"])
        layout.addWidget(self.chk_is_anchor_tail)
        self.chk_is_anchor_tail.stateChanged.connect(self._on_toggle_anchor_tail)

        self.lbl_length = QtWidgets.QLabel(parent)
        layout.addWidget(self.lbl_length)
        self.lbl_tip = QtWidgets.QLabel(parent)
        layout.addWidget(self.lbl_tip)
        self._update_length_label()
        self._update_tip_label()

    def _helper_get_first_used_point_index(self):
        length_all = getattr(self.host, "calc_length", None)
        if length_all is None:
            return None
        length_all = np.asarray(length_all, dtype=float)
        if length_all.size == 0:
            return None

        keep_index = getattr(self.host, "calc_keep_index", None)
        if keep_index is not None:
            keep_index = np.asarray(keep_index, dtype=int)
            if keep_index.size == 0:
                return None
            return int(keep_index[0])

        return 0

    def _update_length_label(self):
        if not hasattr(self, "lbl_length"):
            return

        idx = self._helper_get_first_used_point_index()
        if idx is None:
            self.lbl_length.setText("No currently used point is available.")
            return

        length = float(np.asarray(self.host.calc_length, dtype=float)[idx])
        shaft_length = float(np.asarray(self.host.calc_shaft_length, dtype=float)[idx])
        tip_length = float(np.asarray(self.host.calc_tip_length, dtype=float)[idx])
        self.lbl_length.setText(
            "Length at the red helper marker:\n"
            f"Total: {length:.2f}, Shaft: {shaft_length:.2f}, Tip: {tip_length:.2f}"
        )

    def _update_tip_label(self):
        if not hasattr(self, "lbl_tip"):
            return

        idx = self._helper_get_first_used_point_index()
        if idx is None:
            self.lbl_tip.setText("No currently used point is available.")
            return

        tip_radius = float(np.asarray(self.host.calc_tip_radius, dtype=float)[idx])
        self.lbl_tip.setText(
            "Tip controls:\n"
            f"Length fraction: {float(self.host.opts.tip_length_fraction):.2f}, "
            f"Radius ratio: {float(self.host.opts.tip_radius_ratio):.2f}, "
            f"Tip radius: {tip_radius:.2f}"
        )

    def _extra_commit(self, params):
        length_base = self._length_rescale_base
        scale = float(self.state["length_rescale"])
        if callable(length_base):
            params["length"] = lambda x: scale * length_base(x)
        elif np.isscalar(length_base):
            params["length"] = scale * float(length_base)
        else:
            params["length"] = scale * np.asarray(length_base, dtype=float)

        params["tip_length_fraction"] = float(self.state["tip_length_fraction"])
        params["tip_radius_ratio"] = float(self.state["tip_radius_ratio"])
        params["anchor"] = "tail" if self.state.get("is_anchor_tail") else "center"

    def _on_toggle_anchor_tail(self, _):
        is_tail = self.chk_is_anchor_tail.isChecked()
        self.state["is_anchor_tail"] = is_tail
        if not self._is_block_chk_commit:
            self.commit()

    # ==================== OVERRIDE ====================
    # InteractVector extends the shared sync handler so scalar tip controls and
    # anchor state stay aligned with host-side commits.
    # ==================================================
    def _sync_func(self, **kwargs):
        super()._sync_func(**kwargs)

        if "length" in kwargs and not getattr(self, "_is_gui_updating", False):
            self._length_rescale_base = self._helper_clone_resolver_value(
                self.host.opts.length
            )

        if "tip_length_fraction" in kwargs:
            self._sync_from_host_slider(
                "tip_length_fraction",
                kwargs["tip_length_fraction"],
            )
        if "tip_radius_ratio" in kwargs:
            self._sync_from_host_slider("tip_radius_ratio", kwargs["tip_radius_ratio"])
        if "anchor" in kwargs and hasattr(self, "chk_is_anchor_tail"):
            self._is_block_chk_commit = True
            try:
                self.chk_is_anchor_tail.setChecked(kwargs["anchor"] == "tail")
            finally:
                self._is_block_chk_commit = False
            self.state["is_anchor_tail"] = kwargs["anchor"] == "tail"

        self._update_tip_label()
