from qtpy import QtWidgets
import numpy as np

from .panel_base import make_labeled_slider_row
from nematics3d.datatypes import boundary_periodic_size_to_flag
from nematics3d.grid import apply_linear_transform
from ..plot_sphere import PlotSphere
from .interact_glyph_base import InteractGlyphBase
from nematics3d.logging_decorator import logging_and_warning_decorator


class InteractDisclinationLine(InteractGlyphBase):

    @logging_and_warning_decorator()
    def __init__(self, host, logger=None):
        # host is the PlotTube
        self.wrapper = host.wrapper  # DisclinationLineSmoothPlot
        self.smooth = host.wrapper.owner  # DisclinationLineSmooth
        self._impl_silhouette_state_backup = bool(
            getattr(host, "state_is_silhouette", True)
        )
        self._impl_min_line_length_backup = self.smooth.opts.min_line_length

        object.__setattr__(host, "state_is_silhouette", False)

        super().__init__(
            host=host,
            figure=host.fig,
            title="Smoothed disclination line control",
            is_radius=True,
            is_sides=True,
            is_color=True,
            is_opacity=True,
        )

        console = getattr(self.fig, "console", None)
        if console is not None and self.name != "panel_unregistered":
            console.println(
                "In the command line, the corresponding smoothed-line object is also "
                f"available as the current figure's interacts[{self.name!r}].host.wrapper.owner."
            )
        object.__setattr__(self.smooth.opts, "min_line_length", 2)
        logger.warning(
            "Opening this panel temporarily forces smooth.opts.min_line_length to 2 so short disclination lines can still adjust window_length here."
        )

        self.spheres = PlotSphere(
            self._helper_create_sphere_coords(self.wrapper.opts.is_wrap),
            figure=self.host.fig,
            bounds=self.host.bounds,
            name=f"raw defect points of {self.smooth!r}",
            color=(0, 0, 0),
            is_reset_camera=False,
        )
        if bool(getattr(self.host, "impl_is_bounds_enabled", True)):
            self.spheres.act_bounds_enable()
        else:
            self.spheres.act_bounds_disable()

    def _helper_list_snapshot_hosts(self):
        return [self.smooth, self.wrapper, self.host]

    def _helper_after_restore_snapshot(self, name: str) -> None:
        self.spheres.act_commit(
            coords=self._helper_create_sphere_coords(self.wrapper.opts.is_wrap)
        )

    def _build_extra_group(self):

        group_smooth = QtWidgets.QGroupBox("Smooth", self)
        gl_smooth = QtWidgets.QVBoxLayout(group_smooth)
        self.layout.insertWidget(0, group_smooth)

        self.state["window_length"] = int(self.smooth.opts.window_length)
        self.state["is_smooth"] = bool(self.wrapper.opts.is_smooth)

        self.sliders["window_length"] = make_labeled_slider_row(
            parent=group_smooth,
            layout=gl_smooth,
            name="window_length",
            state_key="window_length",
            value_min=5,
            value_max=np.min([100, self.smooth.owner.calc_defect_num - 1]),
            value_init=self.state["window_length"],
            value_fmt="{:.0f}",
        )

        self.sliders["window_length"].slider.valueChanged.connect(
            lambda _value=0: self._slider_throttle.schedule(
                self.on_changed,
                is_only_smooth=True,
            )
        )
        self.sliders["window_length"].slider.sliderPressed.connect(
            self._helper_begin_slider_interaction
        )
        self.sliders["window_length"].slider.sliderReleased.connect(
            self._helper_end_slider_interaction
        )
        self._custom_sliders.append(self.sliders["window_length"])

        self.chk_is_smooth = QtWidgets.QCheckBox(
            "Use smoothed coordinates", group_smooth
        )
        self.chk_is_smooth.setChecked(self.state["is_smooth"])
        gl_smooth.addWidget(self.chk_is_smooth)
        self.chk_is_smooth.stateChanged.connect(self._on_toggle_is_smooth)
        self.sliders["window_length"].set_enabled(self.state["is_smooth"])

        self.smooth.act_attach_sync_task(
            name=self.str_now,
            func=self._sync_func_smooth,
        )

    def _build_extra_geometry(self, parent, layout):

        self.state["is_wrap"] = bool(self.wrapper.opts.is_wrap)

        self.chk_is_wrap = QtWidgets.QCheckBox("Use wrapped coordinates", parent)
        self.chk_is_wrap.setChecked(self.state["is_wrap"])
        layout.addWidget(self.chk_is_wrap)
        self.chk_is_wrap.stateChanged.connect(self._on_toggle_is_wrap)

        self.wrapper.act_attach_sync_task(
            name=self.str_now,
            func=self._sync_func_wrapper,
        )

    def on_changed(self, _v=0, is_commit=True, is_only_smooth=False):
        for item in self.sliders.values():
            item.sync_to_state(self.state)
        if is_commit:
            if is_only_smooth:
                self._is_gui_updating = True
                try:
                    self.smooth.act_commit(
                        window_length=int(self.state["window_length"])
                    )
                finally:
                    self._is_gui_updating = False
            else:
                self.commit()

    def _sync_func_wrapper(self, **kwargs):
        if "is_smooth" in kwargs:
            self._is_block_chk_commit = True
            self.chk_is_smooth.setChecked(bool(kwargs["is_smooth"]))
            self._is_block_chk_commit = False
        if "is_wrap" in kwargs:
            self._is_block_chk_commit = True
            self.chk_is_wrap.setChecked(bool(kwargs["is_wrap"]))
            self._is_block_chk_commit = False
            self.spheres.act_commit(
                coords=self._helper_create_sphere_coords(bool(kwargs["is_wrap"]))
            )

    def _sync_func_smooth(self, **kwargs):
        if "window_length" in kwargs:
            self._sync_from_host_slider("window_length", kwargs["window_length"])

    def _helper_create_sphere_coords(self, is_wrap):
        owner = self.smooth.owner
        if is_wrap:
            boundary_flag = boundary_periodic_size_to_flag(
                owner.raw_box_size_periodic_index
            )
            coords_index = np.where(
                boundary_flag,
                owner.raw_defect_indices % owner.raw_box_size_periodic_index,
                owner.raw_defect_indices,
            )
            coords = apply_linear_transform(
                coords_index,
                transform=owner.raw_grid_transform,
                offset=owner.raw_grid_offset,
            )
        else:
            coords = owner.calc_defect_coords
        return coords

    # ==================== OVERRIDE ====================
    # InteractDisclinationLine overrides InteractGlyphBase commit
    # execution so that shared glyph-style controls are submitted
    # to the visualization wrapper first, then forwarded to PlotTube.
    # ==================================================
    def _helper_run_commit(self, params):
        self.wrapper.act_commit(**params)

    # ==================== OVERRIDE ====================
    # InteractDisclinationLine overrides the shared bounds checkbox handler so
    # the helper defect-point spheres follow the same bounds effect as the tube.
    # ==================================================
    def _on_toggle_bounds_enabled(self, _state):
        super()._on_toggle_bounds_enabled(_state)
        is_enabled = self.chk_is_bounds_enabled.isChecked()
        if is_enabled:
            self.spheres.act_bounds_enable()
        else:
            self.spheres.act_bounds_disable()

    # ==================== OVERRIDE ====================
    # InteractDisclinationLine extends the glyph sync handler so backend bounds
    # toggles on the tube also keep the helper defect-point spheres in sync.
    # ==================================================
    def _sync_func(self, **kwargs):
        super()._sync_func(**kwargs)
        if "is_bounds_enabled" in kwargs:
            if bool(kwargs["is_bounds_enabled"]):
                self.spheres.act_bounds_enable()
            else:
                self.spheres.act_bounds_disable()

    def _on_toggle_is_wrap(self, _state):
        is_wrap = self.chk_is_wrap.isChecked()
        self.state["is_wrap"] = is_wrap
        self._is_gui_updating = True
        try:
            self.wrapper.act_commit(is_wrap=is_wrap)
        finally:
            self._is_gui_updating = False

    def _on_toggle_is_smooth(self, _state):
        is_smooth = self.chk_is_smooth.isChecked()
        self.state["is_smooth"] = is_smooth
        self.sliders["window_length"].set_enabled(is_smooth)
        self._is_gui_updating = True
        try:
            self.wrapper.act_commit(is_smooth=is_smooth)
        finally:
            self._is_gui_updating = False

    # ==================== OVERRIDE ====================
    # InteractDisclinationLine overrides PanelBase.on_close
    # because it must detach the extra sync tasks created for
    # wrapper/smooth control and remove the helper sphere markers.
    # ==================================================
    def on_close(self):
        super().on_close()
        self.wrapper.act_detach_sync_task(self.str_now)
        self.smooth.act_detach_sync_task(self.str_now)
        object.__setattr__(
            self.host,
            "state_is_silhouette",
            getattr(self, "_impl_silhouette_state_backup", True),
        )
        object.__setattr__(
            self.smooth.opts,
            "min_line_length",
            getattr(
                self, "_impl_min_line_length_backup", self.smooth.opts.min_line_length
            ),
        )
        self.spheres.act_unbind_bounds(is_apply=False)
        self.spheres.act_remove()
