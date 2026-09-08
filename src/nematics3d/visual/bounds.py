"""Visualization support for Bounds objects."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import weakref

import numpy as np

from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.plot_tube import PlotTube


_VISUAL_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 6),
    (4, 7),
    (3, 5),
    (3, 6),
    (5, 7),
    (6, 7),
)
_VISUAL_DEFAULTS = MappingProxyType(
    {"color": (0.0, 0.0, 0.0), "radius": 0.35, "is_pickable": True}
)


@dataclass(slots=True)
class _BoundsVisualEntry:
    figure_ref: weakref.ReferenceType
    tube_ref: weakref.ReferenceType
    sync_name: str

    @property
    def figure(self):
        return self.figure_ref()

    @property
    def tube(self):
        return self.tube_ref()


def _build_visual_edges(bounds) -> tuple[np.ndarray, np.ndarray]:
    coords = []
    line_index = []
    for index, (corner_a, corner_b) in enumerate(_VISUAL_EDGES):
        coords.extend([bounds.corners[corner_a], bounds.corners[corner_b]])
        line_index.extend([index, index])
    return np.asarray(coords, dtype=float), np.asarray(line_index, dtype=int)


def _is_visual_entry_alive(entry: _BoundsVisualEntry) -> bool:
    figure = entry.figure
    tube = entry.tube
    return (
        figure is not None
        and tube is not None
        and figure.is_alive
        and tube in figure.glyphs
    )


def _find_visual_entry(bounds, *, figure=None, tube=None, sync_name=None):
    for entry in bounds.entity_visuals:
        if sync_name is not None and entry.sync_name == sync_name:
            return entry
        if figure is not None and entry.figure is figure:
            return entry
        if tube is not None and entry.tube is tube:
            return entry
    return None


def _prune_visuals(bounds):
    visuals_alive = []
    sync_to_detach = []
    for entry in bounds.entity_visuals:
        if _is_visual_entry_alive(entry):
            visuals_alive.append(entry)
        else:
            sync_to_detach.append(entry.sync_name)

    for sync_name in sync_to_detach:
        bounds.act_detach_sync_task(sync_name)

    if len(visuals_alive) != len(bounds.entity_visuals):
        object.__setattr__(bounds, "entity_visuals", visuals_alive)


def _unregister_visual_sync(bounds, sync_name=None, *, tube=None):
    visuals_alive = []
    sync_to_detach = []
    for entry in bounds.entity_visuals:
        is_match = (sync_name is not None and entry.sync_name == sync_name) or (
            tube is not None and entry.tube is tube
        )
        if is_match:
            sync_to_detach.append(entry.sync_name)
        else:
            visuals_alive.append(entry)

    for sync_name_to_detach in sync_to_detach:
        bounds.act_detach_sync_task(sync_name_to_detach)

    if sync_to_detach:
        object.__setattr__(bounds, "entity_visuals", visuals_alive)


def _refresh_visual(bounds, sync_name: str):
    entry = _find_visual_entry(bounds, sync_name=sync_name)
    if entry is None or not _is_visual_entry_alive(entry):
        _unregister_visual_sync(bounds, sync_name)
        return
    coords, line_index = _build_visual_edges(bounds)
    entry.tube.act_commit(coords=coords, line_index=line_index, is_reapply_opts=True)


def _open_interact_panels(bounds, tube, figure):
    from nematics3d.classes.visual.qt.interact_tube import InteractTube
    from nematics3d.visual.qt.interact_bounds import InteractBounds

    InteractTube.show_once(tube, figure)
    InteractBounds.show_once(bounds, figure)


def _as_plot_figure(figure):
    if figure is None:
        return PlotFigure()
    if isinstance(figure, PlotFigure):
        return figure
    try:
        return PlotFigure(plotter=figure)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return PlotFigure()


def visualize_bounds(
    bounds,
    *,
    figure=None,
    opts=None,
    opts_defaults_override=None,
    name=None,
    category="bounds",
    is_reset_camera=False,
    is_replace=False,
    **kwargs,
):
    """Visualize one Bounds object as a synchronized tube frame."""
    _prune_visuals(bounds)
    figure = _as_plot_figure(figure)

    entry_old = _find_visual_entry(bounds, figure=figure)
    if entry_old is not None:
        tube_old = entry_old.tube
        if tube_old is not None and _is_visual_entry_alive(entry_old):
            if not is_replace:
                if opts is not None:
                    tube_old.act_commit(opts=opts, **kwargs)
                elif kwargs:
                    tube_old.act_commit(**kwargs)
                return tube_old
            tube_old.act_remove()

    coords, line_index = _build_visual_edges(bounds)
    if opts_defaults_override is None:
        opts_defaults_override = dict(_VISUAL_DEFAULTS)
    else:
        opts_defaults_override = dict(_VISUAL_DEFAULTS) | dict(opts_defaults_override)

    tube = PlotTube(
        coords=coords,
        line_index=line_index,
        figure=figure,
        opts=opts,
        opts_defaults_override=opts_defaults_override,
        name=bounds.name if name is None else name,
        category=category,
        is_reset_camera=is_reset_camera,
        **kwargs,
    )

    sync_name = f"{tube.impl_name_pv}__bounds_sync"
    tube.act_bind_relation_base(
        "bounds_visual_source",
        bounds,
        doc="Bounds source driving this visualized frame.",
        is_weak=True,
    )
    tube.act_set_interact_func(
        lambda: _open_interact_panels(bounds, tube=tube, figure=figure)
    )
    bounds.act_attach_sync_task(
        sync_name, lambda **_kwargs: _refresh_visual(bounds, sync_name)
    )
    bounds.entity_visuals.append(
        _BoundsVisualEntry(
            figure_ref=weakref.ref(figure),
            tube_ref=weakref.ref(tube),
            sync_name=sync_name,
        )
    )
    return tube


__all__ = ["visualize_bounds"]
