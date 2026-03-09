from qtpy import QtWidgets
import numpy as np

from .panel_base import make_labeled_slider_row
from Nematics3D.datatypes import boundary_periodic_size_to_flag
from ..plot_sphere import PlotSphere
from .interact_glyph_base import InteractGlyphBase 
from Nematics3D.logging_decorator import logging_and_warning_decorator


class InteractDisclinationLine(InteractGlyphBase):
    
    # @logging_and_warning_decorator()
    def __init__(self, host, logger=None):
        # host is the PlotTube
        self.wrapper = host.wrapper          # DisclinationLineSmoothPlot
        self.smooth = host.wrapper.owner     # DisclinationLineSmooth

        object.__setattr__(self.smooth, "_state_is_silhouette", False)
        
        super().__init__(
            host=host, 
            figure=host.fig, 
            title="Smoothed disclination line control",
            is_radius=True, is_sides=True, is_color=True, is_opacity=True
        )
        
        self.wrapper.act_save_opts(name=self.str_now)
        self.smooth.act_save_opts(name=self.str_now)
        self.wrapper.act_save_opts(name=self.str_now_live)
        self.smooth.act_save_opts(name=self.str_now_live)
        
        # logger.warning(
        #     "To tune smoothing parameter for short lines, "
        #     "The parameter `min_line_length` is set to 2."
        #     "You could find the original settings in _opts_backup of ths line."
        # )
        object.__setattr__(self.smooth.opts, "min_line_length", 2)
        
        self.spheres = PlotSphere(
            self._helper_create_sphere_coords(self.wrapper.opts.is_wrap),
            figure=self.host.fig,
            name="raw defect points of {self.smooth!r}",
            color=(0, 0, 0),
            is_reset_camera=False
        )

    def _build_extra_group(self):

        group_smooth = QtWidgets.QGroupBox("Smooth", self)
        gl_smooth = QtWidgets.QVBoxLayout(group_smooth)
        self.layout.insertWidget(0, group_smooth)

        self.state["window_length"] = int(self.smooth.opts.window_length)
        self.state["is_smooth"] = bool(self.wrapper.opts.is_smooth)

        self.sliders["window_length"] = make_labeled_slider_row(
            parent=group_smooth, layout=gl_smooth,
            name="window_length", state_key="window_length",
            value_min=5,
            value_max=np.min([100, self.smooth.owner._calc_defect_num - 1]),
            value_init=self.state["window_length"],
            value_fmt="{:.0f}",
        )
        
        self.sliders["window_length"].slider.valueChanged.connect(
            lambda: self.on_changed(is_only_smooth=True)
        )
        self._custom_sliders.append(self.sliders["window_length"])

        self.chk_is_smooth = QtWidgets.QCheckBox("Use smoothed coordinates", group_smooth)
        self.chk_is_smooth.setChecked(self.state["is_smooth"])
        gl_smooth.addWidget(self.chk_is_smooth)
        self.chk_is_smooth.stateChanged.connect(self._on_toggle_is_smooth)
        self.sliders["window_length"].set_enabled(self.state["is_smooth"])
    
        
        self.smooth.act_attach_sync_task(
            name = self.str_now_live,
            func = self._sync_func_smooth
        )
        
    def _build_extra_geometry(self, parent, layout):

        self.state["is_wrap"] = bool(self.wrapper.opts.is_wrap)
        
        self.chk_is_wrap = QtWidgets.QCheckBox("Use wrapped coordinates", parent)
        self.chk_is_wrap.setChecked(self.state["is_wrap"])
        layout.addWidget(self.chk_is_wrap) 
        self.chk_is_wrap.stateChanged.connect(self._on_toggle_is_wrap)
        
        self.wrapper.act_attach_sync_task(
            name = self.str_now_live,
            func = self._sync_func_wrapper
        )

    def on_changed(self, _v=0, is_commit=True, is_only_smooth=False):
        for item in self.sliders.values():
            item.sync_to_state(self.state)
        if is_commit:
            if is_only_smooth:            
                self.smooth.opts.window_length = int(self.state["window_length"])
            else:
                self.commit()
        
                
    def _sync_func_wrapper(self, **kwargs):
        if not getattr(self, "_is_gui_updating", False):
            if 'is_smooth' in kwargs:
                self._is_block_chk_commit = True
                self.chk_is_smooth.setChecked(bool(kwargs["is_smooth"]))
                self._is_block_chk_commit = False
            if 'is_wrap' in kwargs:
                self._is_block_chk_commit = True
                self.chk_is_wrap.setChecked(bool(kwargs["is_wrap"]))
                self._is_block_chk_commit = False

    def _sync_func_smooth(self, **kwargs):
        if not getattr(self, "_is_gui_updating", False):
            self._sync_from_host_slider("window_length", kwargs["window_length"])
        

    def _helper_create_sphere_coords(self, is_wrap):
        if is_wrap:
            boundary_flag = boundary_periodic_size_to_flag(self.smooth.owner._raw_box_size_periodic_index)
            coords = np.where(
                boundary_flag,
                self.smooth.owner._calc_defect_coords % self.smooth.owner._raw_box_size_periodic_index,
                self.smooth.owner._calc_defect_coords
            )
        else:
            coords = self.smooth.owner._calc_defect_coords
        return coords

    def _on_toggle_is_wrap(self, _state):
        is_wrap = self.chk_is_wrap.isChecked()
        self.state["is_wrap"] = is_wrap
        self.wrapper.act_commit(is_wrap=is_wrap)

    def _on_toggle_is_smooth(self, _state):
        is_smooth = self.chk_is_smooth.isChecked()
        self.state["is_smooth"] = is_smooth
        self.sliders['window_length'].set_enabled(is_smooth)
        self.wrapper.act_commit(is_smooth=is_smooth)

    def on_close(self):
        super().on_close()
        self.wrapper.act_detach_sync_task(self.str_now_live)
        self.smooth.act_detach_sync_task(self.str_now_live)
        object.__setattr__(self.smooth, "_state_is_silhouette", True)
        self.spheres.act_remove()

