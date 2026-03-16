import numpy as np
from qtpy import QtWidgets

from .interact_glyph_base import InteractGlyphBase
from .panel_base import make_labeled_slider_row, LogTickMapper


class InteractRod(InteractGlyphBase):
    # ==================== OVERRIDE ====================
    # InteractRod overrides InteractGlyphBase.__init__ to enable
    # the standard glyph controls while extending geometry with
    # rod-specific length scaling.
    # ==================================================
    def __init__(self, host, figure):
        super().__init__(
            host,
            figure,
            title=f"Rod Controls of {host.name!r}",
            is_radius=True,
            is_sides=True,
            is_geometry=True,
            is_color=True,
            is_opacity=True,
        )

    def _build_extra_geometry(self, parent, layout):
        self.state["length_rescale"] = 1.0
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
        )
        self.lbl_length = QtWidgets.QLabel(parent)
        layout.addWidget(self.lbl_length)
        self._update_length_label()

    def _update_length_label(self):
        if hasattr(self, "lbl_length") and hasattr(self.host, "_calc_length"):
            self.lbl_length.setText(
                f"The first length is {self.host._calc_length[0]:.2f}"
            )

    def _extra_commit(self, params):
        current_length = self.host._opts_backup[self.str_now_live]["length"]
        scale = float(self.state["length_rescale"])
        if callable(current_length):
            params["length"] = lambda x: scale * current_length(x)
        elif np.isscalar(current_length):
            params["length"] = scale * float(current_length)
        else:
            params["length"] = scale * np.asarray(current_length, dtype=float)

    # ==================== OVERRIDE ====================
    # InteractRod extends InteractGlyphBase._sync_func to keep the
    # rod-specific length slider and label in sync with host updates.
    # ==================================================
    def _sync_func(self, **kwargs):
        super()._sync_func(**kwargs)
        if not getattr(self, "_is_gui_updating", False) and "length" in kwargs:
            self.sliders["length_rescale"].set_tick(1, is_block_signals=True)
            self._update_length_label()