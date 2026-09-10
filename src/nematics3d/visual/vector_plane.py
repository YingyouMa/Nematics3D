"""Visualization adapter for :mod:`nematics3d.sample.vector_plane`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvistaqt import BackgroundPlotter

from nematics3d.core.opts import merge_opts_all
from nematics3d.visual.plot_figure import OptsFigure, PlotFigure, as_plotfigure
from nematics3d.visual.plot_vector import OptsVector, PlotVector


def _bind_plane_interaction(owner, visual) -> None:
    default_func = getattr(visual, "impl_interact_func", None)

    def _interact():
        if callable(default_func):
            default_func()
        from nematics3d.visual.qt.interact_plane import InteractPlane

        InteractPlane.show_once(owner, visual.fig)

    visual.act_set_interact_func(_interact)


def update_vector_plane_visual(owner) -> None:
    """Refresh an existing vector-plane visual, if present."""
    visual = getattr(owner, "visual", None)
    if visual is not None:
        visual.act_commit(coords=owner.grid(), orient=owner.result)


def visualize_vector_plane(
    owner,
    figure: PlotFigure | BackgroundPlotter | None = None,
    opts_figure: OptsFigure | None = None,
    opts_vector: OptsVector | None = None,
    **kwargs,
):
    """Create or replace the PlotVector representation of one VectorPlane."""
    if opts_figure is None:
        opts_figure = OptsFigure()
    if opts_vector is None:
        opts_vector = OptsVector()

    merge = merge_opts_all(
        {"figure_": opts_figure, "": opts_vector},
        kwargs,
        type(owner).__name__,
    )
    figure = as_plotfigure(figure, merge["figure_"])

    visual = PlotVector(
        coords=owner.grid(),
        orient=owner.result,
        name=f"vector field of plane {owner.name!r}",
        category="plane analysis",
        opts=merge[""],
        figure=figure,
        bounds=owner.grid.bounds,
        is_subscribe_bounds=True,
        is_passive_bounds_sync=True,
        opts_defaults_override=owner.default_visual_opts["vector"],
    )
    visual.act_bind_relation_base("owner", owner, is_weak=True)
    _bind_plane_interaction(owner, visual)
    owner.act_bind_relation_base("visual", visual, is_weak=False)
    return visual


__all__ = ["update_vector_plane_visual", "visualize_vector_plane"]
