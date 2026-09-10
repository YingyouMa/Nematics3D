"""Visualization adapters for :mod:`nematics3d.sample.q_surface`."""

import numpy as np

from ..core.opts import merge_opts_all
from .plot_figure import OptsFigure, as_plotfigure
from .plot_rod import OptsRod, PlotRod
from .plot_sphere import OptsSphere, PlotSphere


def _bind_surface_sampling_interaction(field, visual):
    default_func = getattr(visual, "impl_interact_func", None)
    def interact():
        if callable(default_func):
            default_func()
        from .qt.interact_surface_sampling import InteractSurfaceSampling
        InteractSurfaceSampling.show_once(field.sampling, visual.fig)
    visual.act_set_interact_func(interact)


def update_q_surface_visuals(field):
    coords = field.sampling.result
    near = field.calc_is_near_defect
    bulk = ~near
    visual_nb = getattr(field, "visual_nb", None)
    visual_nd = getattr(field, "visual_nd", None)
    visual_defect = getattr(field, "visual_defect", None)
    if visual_nb is not None:
        if np.any(bulk):
            visual_nb.act_commit(coords=coords[bulk], orient=field.calc_n[bulk], is_visible=True)
        else:
            visual_nb.opts.is_visible = False
    if visual_nd is not None:
        if np.any(near):
            visual_nd.act_commit(coords=coords[near], orient=field.calc_n[near], is_visible=True)
        else:
            visual_nd.opts.is_visible = False
    if visual_defect is not None:
        if field.calc_defect_pos is not None and len(field.calc_defect_pos):
            visual_defect.act_commit(coords=field.calc_defect_pos)
        else:
            visual_defect.opts.is_visible = False


def visualize_q_surface_n(field, figure=None, opts_figure: OptsFigure | None = None, opts_nb: OptsRod | None = None, opts_nd: OptsRod | None = None, opts_defect: OptsSphere | None = None, is_defect: bool = False, **kwargs):
    opts_figure = OptsFigure() if opts_figure is None else opts_figure
    opts_nb = OptsRod() if opts_nb is None else opts_nb
    opts_nd = OptsRod() if opts_nd is None else opts_nd
    opts_defect = OptsSphere() if opts_defect is None else opts_defect
    merge = merge_opts_all({"figure_": opts_figure, "nb_": opts_nb, "nd_": opts_nd, "defect_": opts_defect}, kwargs, type(field).__name__)
    figure = as_plotfigure(figure, merge["figure_"])
    coords = field.sampling.result
    near = field.calc_is_near_defect
    bulk = ~near
    fallback = np.arange(min(2, len(coords)))
    visual_nb = PlotRod(coords=coords[bulk] if np.any(bulk) else coords[fallback], orient=field.calc_n[bulk] if np.any(bulk) else field.calc_n[fallback], name=f"n bulk of surface {field.name!r}", category="surface analysis", opts=merge["nb_"], figure=figure, opts_defaults_override=field.default_visual_opts["nb"], is_visible=bool(np.any(bulk)))
    visual_nb.act_bind_relation_base("owner", field, is_weak=True)
    field.act_bind_relation_base("visual_nb", visual_nb, is_weak=False)
    _bind_surface_sampling_interaction(field, visual_nb)
    visual_nd = PlotRod(coords=coords[near] if np.any(near) else coords[fallback], orient=field.calc_n[near] if np.any(near) else field.calc_n[fallback], name=f"n near defect of surface {field.name!r}", category="surface analysis", opts=merge["nd_"], figure=figure, opts_defaults_override=field.default_visual_opts["nd"], is_visible=bool(np.any(near)))
    visual_nd.act_bind_relation_base("owner", field, is_weak=True)
    field.act_bind_relation_base("visual_nd", visual_nd, is_weak=False)
    _bind_surface_sampling_interaction(field, visual_nd)
    has_defects = field.calc_defect_pos is not None and len(field.calc_defect_pos) > 0
    visual_defect = PlotSphere(coords=field.calc_defect_pos if has_defects else coords[fallback], name=f"defects of surface {field.name!r}", category="surface analysis", opts=merge["defect_"], figure=figure, is_visible=bool(is_defect and has_defects))
    visual_defect.act_bind_relation_base("owner", field, is_weak=True)
    field.act_bind_relation_base("visual_defect", visual_defect, is_weak=False)
    return visual_nb


__all__ = ["update_q_surface_visuals", "visualize_q_surface_n"]
