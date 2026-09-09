"""Visualization adapters for :mod:`nematics3d.sample.q_plane`."""

from __future__ import annotations

import numpy as np

from nematics3d.datatypes import as_bool
from nematics3d.core.opts import merge_opts_all
from nematics3d.visual.plot_delaunay import OptsDelaunay, PlotDelaunay
from nematics3d.visual.plot_figure import OptsFigure, as_plotfigure
from nematics3d.visual.plot_rod import OptsRod, PlotRod
from nematics3d.visual.plot_sphere import OptsSphere, PlotSphere


def _bind_plane_interaction(field, visual):
    from nematics3d.sample.q_plane import QPlanePolar

    default_func = getattr(visual, "impl_interact_func", None)

    def interact():
        if callable(default_func):
            default_func()
        if isinstance(field, QPlanePolar):
            from nematics3d.visual.qt.interact_defect_section import (
                InteractDefectSection,
            )

            InteractDefectSection.show_once(field, visual.fig)
        else:
            from nematics3d.visual.qt.interact_plane import InteractPlane

            InteractPlane.show_once(field, visual.fig)

    visual.act_set_interact_func(interact)


def update_q_plane_visuals(field):
    """Refresh any Q-plane visuals already attached to ``field``."""
    visual_nb = getattr(field, "visual_nb", None)
    visual_nd = getattr(field, "visual_nd", None)
    visual_defect = getattr(field, "visual_defect", None)
    visual_S = getattr(field, "visual_S", None)

    if visual_nb is not None and visual_nd is not None:
        bulk = ~field.calc_is_near_defect
        near = field.calc_is_near_defect
        if np.any(bulk):
            visual_nb.act_commit(
                coords=field.grid()[bulk],
                orient=field.calc_n[bulk],
                is_visible=True,
            )
        else:
            visual_nb.opts.is_visible = False

        if np.any(near):
            visual_nd.act_commit(
                coords=field.grid()[near],
                orient=field.calc_n[near],
                is_visible=True,
            )
        else:
            visual_nd.opts.is_visible = False

        if visual_defect is not None:
            if field.calc_defect_pos is not None and len(field.calc_defect_pos) > 0:
                visual_defect.act_commit(
                    coords=field.calc_defect_pos,
                    is_visible=visual_defect.is_show_defect,
                )
            else:
                visual_defect.opts.is_visible = False

    if visual_S is not None:
        visual_S.act_commit(coords=field.grid(), scalars=field.calc_S)


def visualize_q_plane_n(
    field,
    figure=None,
    opts_figure: OptsFigure | None = None,
    opts_nb: OptsRod | None = None,
    opts_nd: OptsRod | None = None,
    opts_defect: OptsSphere | None = None,
    is_defect: bool = False,
    **kwargs,
):
    """Create director and optional defect visuals for one sampled Q plane."""
    is_defect = as_bool(is_defect, replace=True)
    opts_nb = OptsRod() if opts_nb is None else opts_nb
    opts_nd = OptsRod() if opts_nd is None else opts_nd
    opts_figure = OptsFigure() if opts_figure is None else opts_figure
    opts_defect = OptsSphere() if opts_defect is None else opts_defect

    merge = merge_opts_all(
        {
            "figure_": opts_figure,
            "defect_": opts_defect,
            "nb_": opts_nb,
            "nd_": opts_nd,
        },
        kwargs,
        type(field).__name__,
    )
    figure = as_plotfigure(figure, merge["figure_"])
    opts_defect = merge["defect_"]
    opts_nb = merge["nb_"]
    opts_nd = merge["nd_"]

    bulk = ~field.calc_is_near_defect
    near = field.calc_is_near_defect
    fallback = np.flatnonzero(bulk)[:2]
    if len(fallback) == 0:
        fallback = np.arange(min(2, len(field.grid())))

    visual_nb = PlotRod(
        coords=field.grid()[bulk] if np.any(bulk) else field.grid()[fallback],
        orient=field.calc_n[bulk] if np.any(bulk) else field.calc_n[fallback],
        name=f"n bulk of plane {field.name!r}",
        category="plane analysis",
        opts=opts_nb,
        figure=figure,
        bounds=field.grid.bounds,
        is_subscribe_bounds=True,
        is_passive_bounds_sync=True,
        opts_defaults_override=field.default_visual_opts["nb"],
        is_visible=bool(np.any(bulk)),
    )
    visual_nb.act_bind_relation_base("owner", field, is_weak=True)
    _bind_plane_interaction(field, visual_nb)
    field.act_bind_relation_base("visual_nb", visual_nb, is_weak=False)

    visual_nd = PlotRod(
        coords=field.grid()[near] if np.any(near) else field.grid()[fallback],
        orient=field.calc_n[near] if np.any(near) else field.calc_n[fallback],
        name=f"n near defect of plane {field.name!r}",
        category="plane analysis",
        opts=opts_nd,
        figure=figure,
        bounds=field.grid.bounds,
        is_subscribe_bounds=True,
        is_passive_bounds_sync=True,
        opts_defaults_override=field.default_visual_opts["nd"],
        is_visible=bool(np.any(near)),
    )
    visual_nd.act_bind_relation_base("owner", field, is_weak=True)
    _bind_plane_interaction(field, visual_nd)
    field.act_bind_relation_base("visual_nd", visual_nd, is_weak=False)

    defect_coords = (
        field.calc_defect_pos
        if field.calc_defect_pos is not None and len(field.calc_defect_pos) > 0
        else field.grid()[fallback]
    )
    visual_defect = PlotSphere(
        coords=defect_coords,
        name=f"defects of plane {field.name!r}",
        category="plane analysis",
        opts=opts_defect,
        figure=figure,
        bounds=field.grid.bounds,
        is_subscribe_bounds=True,
        is_passive_bounds_sync=True,
        is_visible=bool(is_defect and field.calc_defect_pos is not None),
    )
    visual_defect.act_bind_relation_base("owner", field, is_weak=True)
    field.act_bind_relation_base("visual_defect", visual_defect, is_weak=False)
    visual_defect.act_add_attr(
        "is_show_defect",
        f"Whether to plot defect points during director visualization of {field.name}.",
        default=is_defect,
    )
    visual_defect.opts.is_visible = bool(
        visual_defect.is_show_defect
        and field.calc_defect_pos is not None
        and len(field.calc_defect_pos) > 0
    )
    return visual_nb, visual_nd, visual_defect


def visualize_q_plane_S(
    field,
    figure=None,
    opts_figure: OptsFigure | None = None,
    opts_S: OptsDelaunay | None = None,
    **kwargs,
):
    """Create a scalar-order surface visual for one sampled Q plane."""
    opts_figure = OptsFigure() if opts_figure is None else opts_figure
    opts_S = OptsDelaunay() if opts_S is None else opts_S
    merge = merge_opts_all(
        {"figure_": opts_figure, "S_": opts_S},
        kwargs,
        type(field).__name__,
    )
    figure = as_plotfigure(figure, merge["figure_"])
    visual_S = PlotDelaunay(
        coords=field.grid(),
        scalars=field.calc_S,
        figure=figure,
        name=f"S of plane {field.name!r}",
        category="plane analysis",
        opts=merge["S_"],
        bounds=field.grid.bounds,
        is_subscribe_bounds=True,
        is_passive_bounds_sync=True,
        opts_defaults_override=field.default_visual_opts["S"],
    )
    visual_S.act_bind_relation_base("owner", field, is_weak=True)
    _bind_plane_interaction(field, visual_S)
    field.act_bind_relation_base("visual_S", visual_S, is_weak=False)
    return visual_S


__all__ = ["update_q_plane_visuals", "visualize_q_plane_n", "visualize_q_plane_S"]
