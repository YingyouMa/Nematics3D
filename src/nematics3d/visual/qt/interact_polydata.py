from qtpy import QtWidgets

from nematics3d.visual.qt.interact_glyph_base import InteractGlyphBase
from nematics3d.visual.qt.panel_base import make_labeled_slider_row, make_RGB_slider


class InteractPolyData(InteractGlyphBase):
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

        self.chk_is_show_edges = QtWidgets.QCheckBox("Show edges", group_edges)
        self.chk_is_show_edges.setChecked(self.state["is_show_edges"])
        gl_edges.addWidget(self.chk_is_show_edges)

        make_RGB_slider(
            group_edges, gl_edges, self.sliders, "edge_color", init_edge_color
        )
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

        self._set_edge_controls_enabled(self.state["is_show_edges"])
        self.chk_is_show_edges.stateChanged.connect(self._on_toggle_show_edges)

    def _set_edge_controls_enabled(self, enabled: bool):
        for key in ("edge_color_r", "edge_color_g", "edge_color_b", "edge_width"):
            self.sliders[key].set_enabled(enabled)

    def _on_toggle_show_edges(self, _):
        is_on = self.chk_is_show_edges.isChecked()
        self.state["is_show_edges"] = is_on
        self._set_edge_controls_enabled(is_on)
        if not self._is_block_chk_commit:
            self.commit()

    def _extra_commit(self, params):
        params["is_show_edges"] = self.state["is_show_edges"]
        if self.state["is_show_edges"]:
            params["edge_color"] = (
                float(self.state["edge_color_r"]),
                float(self.state["edge_color_g"]),
                float(self.state["edge_color_b"]),
            )
            params["edge_width"] = float(self.state["edge_width"])

    def _sync_func(self, **kwargs):
        super()._sync_func(**kwargs)

        if "is_show_edges" in kwargs and hasattr(self, "chk_is_show_edges"):
            is_on = bool(kwargs["is_show_edges"])
            self._is_block_chk_commit = True
            try:
                self.chk_is_show_edges.setChecked(is_on)
            finally:
                self._is_block_chk_commit = False
            self.state["is_show_edges"] = is_on
            self._set_edge_controls_enabled(is_on)

        if "edge_width" in kwargs and "edge_width" in self.sliders:
            self._sync_from_host_slider("edge_width", kwargs["edge_width"])

        if "edge_color" in kwargs:
            for channel, value in zip(("r", "g", "b"), tuple(kwargs["edge_color"])):
                key = f"edge_color_{channel}"
                if key in self.sliders:
                    self._sync_from_host_slider(key, value)


__all__ = ["InteractPolyData"]
