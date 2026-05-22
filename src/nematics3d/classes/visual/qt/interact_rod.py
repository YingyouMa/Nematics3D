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
            input_out_of_range="expand_max",
        )
        self.lbl_length = QtWidgets.QLabel(parent)
        layout.addWidget(self.lbl_length)
        self._update_length_label()

    def _helper_get_first_used_point_length(self):
        if not hasattr(self.host, "calc_length"):
            return None

        length_all = np.asarray(self.host.calc_length, dtype=float)
        if length_all.size == 0:
            return None

        keep_index = getattr(self.host, "calc_keep_index", None)
        if keep_index is not None:
            keep_index = np.asarray(keep_index, dtype=int)
            if keep_index.size == 0:
                return None
            return float(length_all[int(keep_index[0])])

        return float(length_all[0])

    def _update_length_label(self):
        if not hasattr(self, "lbl_length"):
            return

        length = self._helper_get_first_used_point_length()
        if length is None:
            self.lbl_length.setText("No currently used point is available.")
            return

        self.lbl_length.setText(f"Length at the red helper marker: {length:.2f}")

    def _extra_commit(self, params):
        current_length = self.host.opts.length
        scale = float(self.state["length_rescale"])
        if callable(current_length):
            params["length"] = lambda x: scale * current_length(x)
        elif np.isscalar(current_length):
            params["length"] = scale * float(current_length)
        else:
            params["length"] = scale * np.asarray(current_length, dtype=float)
