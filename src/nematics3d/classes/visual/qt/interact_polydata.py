from qtpy import QtWidgets
from .interact_glyph_base import InteractGlyphBase
from .panel_base import make_labeled_slider_row, make_RGB_slider


class InteractPolyData(InteractGlyphBase):
    # ==================== OVERRIDE ====================
    # InteractPolyData overrides InteractGlyphBase.__init__ because PolyData
    # mesh control only needs color and opacity, while radius, sides, and
    # extra geometry controls are not meaningful here.
    # ==================================================
    def __init__(self, host, figure):
        super().__init__(
            host,
            figure,
            title=f"PolyData Controls of {host.name!r}",
            is_radius=False,
            is_sides=False,
            is_geometry=False,
            is_color=True,
            is_opacity=True,
        )

    # -------------------------------
    # Extra UI: Edges group
    # -------------------------------

    def _build_extra_group(self):
        opts = self.host.opts

        self.state["is_show_edges"] = bool(opts.is_show_edges)
        init_edge_color = tuple(opts.edge_color)
        self.state["edge_color_r"] = init_edge_color[0]
        self.state["edge_color_g"] = init_edge_color[1]
        self.state["edge_color_b"] = init_edge_color[2]
        self.state["edge_width"] = float(opts.edge_width)

        group_edges = QtWidgets.QGroupBox("Edges", self)
        gl_edges = QtWidgets.QVBoxLayout(group_edges)
        self.layout.addWidget(group_edges)

        # --- show edges checkbox ---
        self.chk_is_show_edges = QtWidgets.QCheckBox("Show edges", group_edges)
        self.chk_is_show_edges.setChecked(self.state["is_show_edges"])
        gl_edges.addWidget(self.chk_is_show_edges)

        # --- edge color sliders ---
        make_RGB_slider(group_edges, gl_edges, self.sliders, "edge_color", init_edge_color)

        # --- edge width slider ---
        self.sliders["edge_width"] = make_labeled_slider_row(
            parent=group_edges,
            layout=gl_edges,
            name="edge_width",
            state_key="edge_width",
            value_min=0.5,
            value_max=10.0,
            value_init=self.state["edge_width"],
            tick_to_value=lambda t: t / 10.0,
            value_to_tick=lambda v: int(v * 10),
        )

        # disable edge controls unless show_edges is on
        self._set_edge_controls_enabled(self.state["is_show_edges"])

        self.chk_is_show_edges.stateChanged.connect(self._on_toggle_show_edges)

    def _set_edge_controls_enabled(self, enabled: bool):
        for k in ("edge_color_r", "edge_color_g", "edge_color_b", "edge_width"):
            self.sliders[k].set_enabled(enabled)

    def _on_toggle_show_edges(self, _):
        is_on = self.chk_is_show_edges.isChecked()
        self.state["is_show_edges"] = is_on
        self._set_edge_controls_enabled(is_on)
        if not self._is_block_chk_commit:
            self.commit()

    # -------------------------------
    # Commit
    # -------------------------------

    def _extra_commit(self, params):
        params["is_show_edges"] = self.state["is_show_edges"]
        if self.state["is_show_edges"]:
            params["edge_color"] = (
                float(self.state["edge_color_r"]),
                float(self.state["edge_color_g"]),
                float(self.state["edge_color_b"]),
            )
            params["edge_width"] = float(self.state["edge_width"])
