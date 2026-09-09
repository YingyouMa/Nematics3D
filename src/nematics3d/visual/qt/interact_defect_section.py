"""Interactive controls for defect-centered polar Q-plane sections."""

from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

from nematics3d.format import is_equal, is_given_str
from nematics3d.visual.plot_rod import PlotRod
from nematics3d.visual.qt.interact_plane import InteractPlane
from nematics3d.visual.qt.panel_base import (
    LogTickMapper,
    PanelBase,
    make_labeled_slider_row,
)


class InteractDefectSection(PanelBase):
    """Control a defect-section wrapper and its bound polar sampling grid."""

    def __init__(self, field, figure):
        self.field = field
        self.defect_plane = field.grid.wrapper
        object.__setattr__(field, "state_is_interactable", False)
        object.__setattr__(field.grid, "impl_is_warn_orthogonal", False)
        self.visual_normal = PlotRod(
            coords=field.grid.opts.origin,
            orient=field.grid.opts.normal,
            radius=field.grid.opts.dr / 4,
            length=field.grid.opts.layers * field.grid.opts.dr * 2.5,
            color=(1, 0, 0),
            figure=figure,
            name=f"The normal of {field.grid.name!r}",
            category="Interaction",
            is_reset_camera=False,
            is_visible=False,
        )
        object.__setattr__(self.visual_normal, "state_is_interactable", False)
        self._is_continuous_interacting = False
        super().__init__(field.grid, figure, title=f"Controls of {field.grid.name!r}")
        self._section_sync_name = self.str_now + "_section"
        self.defect_plane.act_attach_sync_task(
            self._section_sync_name, self._sync_func_defect_plane
        )

    def _helper_list_snapshot_hosts(self):
        return [self.defect_plane, self.host]

    def _iter_silhouette_targets(self):
        targets = [self.visual_normal] + [
            getattr(self.field, name, None)
            for name in ("visual_nb", "visual_nd", "visual_defect", "visual_S")
        ]
        seen = set()
        for visual in targets:
            if visual is None or id(visual) in seen:
                continue
            seen.add(id(visual))
            yield visual

    def _helper_begin_continuous_interaction(self, *_args):
        if self._is_continuous_interacting:
            return
        self._is_continuous_interacting = True
        for visual in self._iter_silhouette_targets():
            if hasattr(visual, "state_is_silhouette"):
                object.__setattr__(visual, "state_is_silhouette", False)
                if hasattr(visual, "_helper_clear_silhouette"):
                    visual._helper_clear_silhouette()

    def _helper_end_continuous_interaction(self, *_args):
        if not self._is_continuous_interacting:
            return
        self._is_continuous_interacting = False
        for visual in self._iter_silhouette_targets():
            if hasattr(visual, "state_is_silhouette"):
                object.__setattr__(visual, "state_is_silhouette", True)
                if getattr(visual, "entity_actor", None) is not None and hasattr(
                    visual, "_helper_add_silhouette"
                ):
                    visual._helper_add_silhouette()

    def _update_normal_visual(self, is_visible=True):
        self.visual_normal.act_commit(
            coords=np.asarray(self.host.opts.origin, dtype=float).reshape(1, 3),
            orient=np.asarray(self.host.opts.normal, dtype=float).reshape(1, 3),
            radius=float(self.host.opts.dr) / 4,
            length=float(self.host.opts.layers * self.host.opts.dr * 2.5),
            is_visible=is_visible,
        )

    def _show_dataset_grid_spacing_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Source Dataset Grid Axes")
        layout = QtWidgets.QVBoxLayout(dialog)
        for line in InteractPlane._helper_dataset_grid_spacing_lines(self.field):
            layout.addWidget(QtWidgets.QLabel(line, dialog))
        close = QtWidgets.QPushButton("Close", dialog)
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.show()
        self._dataset_grid_info_dialog = dialog

    def build_ui(self):
        row = QtWidgets.QWidget(self)
        row_layout = QtWidgets.QHBoxLayout(row)
        self.chk_is_show_axes = QtWidgets.QCheckBox("Show section normal", row)
        self.chk_is_show_axes.stateChanged.connect(self._on_toggle_show_axes)
        self.btn_dataset_grid_info = QtWidgets.QPushButton("Show grid axes", row)
        self.btn_dataset_grid_info.clicked.connect(
            self._show_dataset_grid_spacing_dialog
        )
        row_layout.addWidget(self.chk_is_show_axes)
        row_layout.addWidget(self.btn_dataset_grid_info)
        self.layout.addWidget(row)

        arc_dist = self.host.opts.arc_dist or self.host.opts.dr
        controlled_normal = not is_given_str(self.defect_plane.state_normal, "tangent")
        self.state = {
            "u_percent": self.defect_plane.opts.u_percent,
            "dr": float(self.host.opts.dr),
            "arc_dist": float(arc_dist),
            "layers": int(self.host.opts.layers),
            "is_use_control_arc_dist": self.host.opts.arc_dist is not None,
            "is_use_control_normal": controlled_normal,
            "normal_azimuth": self.get_azimuth(self.host.opts.normal),
            "normal_polar_angle": self.get_polar_angle(self.host.opts.normal),
        }

        placement = QtWidgets.QGroupBox("Placement", self)
        placement_layout = QtWidgets.QVBoxLayout(placement)
        self.layout.addWidget(placement)
        self.sliders["u_percent"] = make_labeled_slider_row(
            parent=placement,
            layout=placement_layout,
            name="u_percent",
            state_key="u_percent",
            value_min=0,
            value_max=100,
            value_init=self.state["u_percent"],
            tick_to_value=lambda t: t / 1000,
            value_to_tick=lambda v: int(round(v * 1000)),
            value_fmt="{:.3f}",
        )
        self.sliders["normal_azimuth"] = make_labeled_slider_row(
            parent=placement,
            layout=placement_layout,
            name="Azimuth of normal",
            state_key="normal_azimuth",
            value_min=0,
            value_max=360,
            value_init=self.state["normal_azimuth"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.3f}",
        )
        self.sliders["normal_polar_angle"] = make_labeled_slider_row(
            parent=placement,
            layout=placement_layout,
            name="Polar angle of normal",
            state_key="normal_polar_angle",
            value_min=0,
            value_max=180,
            value_init=self.state["normal_polar_angle"],
            tick_to_value=lambda t: t / 10,
            value_to_tick=lambda v: int(v * 10),
            value_fmt="{:.3f}",
        )
        self.chk_use_normal = QtWidgets.QCheckBox("Use controlled normal", placement)
        self.chk_use_normal.setChecked(controlled_normal)
        self.chk_use_normal.stateChanged.connect(self._on_toggle_use_normal)
        placement_layout.addWidget(self.chk_use_normal)
        self.sliders["normal_azimuth"].set_enabled(controlled_normal)
        self.sliders["normal_polar_angle"].set_enabled(controlled_normal)

        scalar = QtWidgets.QGroupBox("Sampling", self)
        scalar_layout = QtWidgets.QVBoxLayout(scalar)
        self.layout.addWidget(scalar)
        mapper_dr = LogTickMapper(0.2 * self.state["dr"], 5 * self.state["dr"], base=10)
        self.sliders["dr"] = make_labeled_slider_row(
            parent=scalar,
            layout=scalar_layout,
            name="dr",
            value_min=mapper_dr.value_min,
            value_max=mapper_dr.value_max,
            value_init=self.state["dr"],
            tick_to_value=mapper_dr.tick_to_value,
            value_to_tick=mapper_dr.value_to_tick,
            input_out_of_range="expand_max",
        )
        mapper_arc = LogTickMapper(
            0.2 * self.state["arc_dist"], 5 * self.state["arc_dist"], base=10
        )
        self.sliders["arc_dist"] = make_labeled_slider_row(
            parent=scalar,
            layout=scalar_layout,
            name="arc_dist",
            value_min=mapper_arc.value_min,
            value_max=mapper_arc.value_max,
            value_init=self.state["arc_dist"],
            tick_to_value=mapper_arc.tick_to_value,
            value_to_tick=mapper_arc.value_to_tick,
            input_out_of_range="expand_max",
        )
        self.chk_use_arc_dist = QtWidgets.QCheckBox("Use controlled arc_dist", scalar)
        self.chk_use_arc_dist.setChecked(self.state["is_use_control_arc_dist"])
        self.chk_use_arc_dist.stateChanged.connect(self._on_toggle_use_arc_dist)
        scalar_layout.addWidget(self.chk_use_arc_dist)
        self.sliders["arc_dist"].set_enabled(self.state["is_use_control_arc_dist"])
        self.sliders["layers"] = make_labeled_slider_row(
            parent=scalar,
            layout=scalar_layout,
            name="layers",
            state_key="layers",
            value_min=1,
            value_max=max(4, int(np.ceil(self.state["layers"] * 3))),
            value_init=self.state["layers"],
            value_fmt="{:.0f}",
            input_out_of_range="expand_max",
        )
        self.on_changed(0, is_commit=False)

    def _on_toggle_show_axes(self):
        if self.chk_is_show_axes.isChecked():
            self._update_normal_visual(True)
        else:
            self.visual_normal.opts.is_visible = False

    def commit(self):
        normal = (
            self._helper_calc_vec(
                np.deg2rad(self.state["normal_azimuth"]),
                np.deg2rad(self.state["normal_polar_angle"]),
            )
            if self.state["is_use_control_normal"]
            else "tangent"
        )
        arc_dist = (
            float(self.state["arc_dist"])
            if self.state["is_use_control_arc_dist"]
            else None
        )
        params = {"u_percent": self.state["u_percent"]}
        if not is_equal(self.host.opts.dr, self.state["dr"]):
            params["dr"] = self.state["dr"]
        if not is_equal(self.host.opts.layers, int(self.state["layers"])):
            params["layers"] = int(self.state["layers"])
        if not is_equal(self.host.opts.arc_dist, arc_dist):
            params["arc_dist"] = arc_dist
        if not is_equal(self.defect_plane.state_normal, normal):
            params["state_normal"] = normal
        self.defect_plane.act_commit(**params)

    def _on_toggle_use_normal(self, _state):
        value = self.chk_use_normal.isChecked()
        self.state["is_use_control_normal"] = value
        self.sliders["normal_azimuth"].set_enabled(value)
        self.sliders["normal_polar_angle"].set_enabled(value)
        self.commit()

    def _on_toggle_use_arc_dist(self, _state):
        value = self.chk_use_arc_dist.isChecked()
        self.state["is_use_control_arc_dist"] = value
        self.sliders["arc_dist"].set_enabled(value)
        self.commit()

    @staticmethod
    def get_azimuth(vec):
        return InteractPlane.get_azimuth(vec)

    @staticmethod
    def get_polar_angle(vec):
        return InteractPlane.get_polar_angle(vec)

    def _sync_func(self, **kwargs):
        if "normal" in kwargs and self.state["is_use_control_normal"]:
            self._sync_from_host_slider(
                "normal_azimuth", self.get_azimuth(self.host.opts.normal)
            )
            self._sync_from_host_slider(
                "normal_polar_angle", self.get_polar_angle(self.host.opts.normal)
            )
        if "dr" in kwargs:
            self._sync_from_host_slider("dr", self.host.opts.dr)
        if "arc_dist" in kwargs:
            controlled = self.host.opts.arc_dist is not None
            self.state["is_use_control_arc_dist"] = controlled
            self.chk_use_arc_dist.blockSignals(True)
            self.chk_use_arc_dist.setChecked(controlled)
            self.chk_use_arc_dist.blockSignals(False)
            self.sliders["arc_dist"].set_enabled(controlled)
            if controlled:
                self._sync_from_host_slider("arc_dist", self.host.opts.arc_dist)
        if "layers" in kwargs:
            self._sync_from_host_slider("layers", int(self.host.opts.layers))
        if self.chk_is_show_axes.isChecked():
            self._update_normal_visual(True)

    def _sync_func_defect_plane(self, **kwargs):
        if "u_percent" in kwargs:
            self._sync_from_host_slider("u_percent", self.defect_plane.opts.u_percent)
        if "state_normal" in kwargs:
            controlled = not is_given_str(self.defect_plane.state_normal, "tangent")
            self.state["is_use_control_normal"] = controlled
            self.chk_use_normal.blockSignals(True)
            self.chk_use_normal.setChecked(controlled)
            self.chk_use_normal.blockSignals(False)
            self.sliders["normal_azimuth"].set_enabled(controlled)
            self.sliders["normal_polar_angle"].set_enabled(controlled)

    def on_close(self):
        self._helper_end_continuous_interaction()
        self.defect_plane.act_detach_sync_task(self._section_sync_name)
        super().on_close()
        object.__setattr__(self.host, "impl_is_warn_orthogonal", True)
        object.__setattr__(self.field, "state_is_interactable", True)
        self.visual_normal.act_remove()


__all__ = ["InteractDefectSection"]
