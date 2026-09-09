"""Visualization wrapper for smoothed disclination lines."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping

import numpy as np

from ..analysis.disclination.smooth import DisclinationLineSmooth
from ..core.class_base import AttrDef
from ..core.host_base import HostBase, OptsBase
from ..core.opts import cover_value
from ..datatypes import UNSET, Unset, as_bool
from ..grid import apply_linear_transform
from .plot_figure import PlotFigure
from .plot_tube import OptsTube, PlotTube


@dataclass(slots=True, repr=False)
class OptsDefectLinePlot(OptsBase):
    is_smooth: bool | Unset = UNSET
    is_wrap: bool | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "is_smooth": "Whether to display smoothed rather than raw coordinates.",
        "is_wrap": "Whether to wrap periodic coordinates into the principal box.",
    }
    impl_validators: ClassVar[Mapping[str, Callable[[Any, str], Any]]] = {
        **OptsBase.impl_validators,
        "is_smooth": lambda v, d: as_bool(v, name=d),
        "is_wrap": lambda v, d: as_bool(v, name=d),
    }
    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(getattr(OptsBase, "impl_defaults_frozen", {})),
            "is_smooth": True,
            "is_wrap": True,
        }
    )


class DisclinationLineSmoothPlot(HostBase):
    __attr_defs__ = {
        "owner": AttrDef(
            doc="Smoothed disclination line.", kind="relation", is_weak_by_default=True
        ),
        "wrapped": AttrDef(
            doc="Underlying PlotTube.", kind="relation", is_weak_by_default=False
        ),
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in HostBase.__slots__
    )

    def __init__(
        self,
        line: DisclinationLineSmooth,
        figure: PlotFigure | None = None,
        opts: OptsDefectLinePlot | None = None,
        opts_tube: OptsTube | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        opts_tube_defaults_override: Mapping[str, Any] | None = None,
        name: str | None = None,
        **kwargs,
    ):
        if not isinstance(line, DisclinationLineSmooth):
            raise TypeError("`line` must be a DisclinationLineSmooth instance.")
        own_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs) if k in OptsDefectLinePlot.__attrs__
        }
        super().__init__(
            opts_type=OptsDefectLinePlot,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            name=line.name if name is None else name,
            name_replace=line.name,
            **own_kwargs,
        )
        self.act_bind_relation_base("owner", line, is_weak=True)
        self.opts.act_finalize(defaults=self.opts_defaults)
        coords, line_index = self._helper_get_coords()
        tube = PlotTube(
            coords=coords,
            line_index=line_index,
            name=line.name,
            category="disclination line",
            figure=figure,
            opts=opts_tube,
            opts_defaults_override=opts_tube_defaults_override,
            **kwargs,
        )
        tube.act_bind_wrapper(self, protected_attrs=["coords", "line_index"])
        self.act_attach_enrich_kwargs_wrapped_task(
            "visual_coords", self._helper_enrich_kwargs_wrapped_visual
        )
        from .qt.interact_disclination_line import InteractDisclinationLine

        tube.act_set_interact_func(lambda: InteractDisclinationLine.show_once(tube))

    def _helper_get_coords(self):
        smooth = self.owner
        line = smooth.owner
        if not self.opts.is_wrap:
            coords = smooth.result if self.opts.is_smooth else line.calc_defect_coords
            if line.kind == "loop":
                coords = np.concatenate((coords, [coords[0]]))
            return coords, None
        coords_index = (
            smooth.calc_result_index if self.opts.is_smooth else line.raw_defect_indices
        )
        if line.kind == "loop":
            coords_index = np.concatenate((coords_index, [coords_index[0]]))
        periodic = np.isfinite(line.raw_box_size_periodic_index)
        coords_index = np.where(
            periodic,
            coords_index % line.raw_box_size_periodic_index,
            coords_index,
        )
        diff = np.diff(coords_index, axis=0)
        jump = (
            np.any(
                np.abs(diff[:, periodic])
                > line.raw_box_size_periodic_index[periodic] / 2,
                axis=-1,
            )
            if np.any(periodic)
            else np.zeros(len(diff), dtype=bool)
        )
        cuts = np.concatenate([[0], np.where(jump)[0] + 1, [len(coords_index)]])
        line_index = np.empty(len(coords_index), dtype=int)
        for i, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
            line_index[start:stop] = i
        coords = apply_linear_transform(
            coords_index,
            transform=line.raw_grid_transform,
            offset=line.raw_grid_offset,
        )
        return coords, line_index

    def _helper_enrich_kwargs_wrapped_visual(self, host=None, kwargs=None):
        del host, kwargs
        coords, line_index = self._helper_get_coords()
        return {"coords": coords, "line_index": line_index}

    def act_remove(self):
        owner, wrapped = self.owner, self.wrapped
        if wrapped is not None:
            wrapped.act_remove()
            if wrapped.wrapper is self:
                wrapped.act_unbind_wrapper()
        if owner is not None and getattr(owner, "visual", None) is self:
            owner.act_unbind_relation_base("visual")
        if owner is not None and self.owner is owner:
            self.act_unbind_relation_base("owner")

    def _helper_commit_apply_opts_main(self, is_reapply_opts=False, **kwargs):
        if not is_reapply_opts and not kwargs:
            return
        with self.opts.act_internal_update():
            cover_value(
                self.opts,
                is_allow_cover_target_set=True,
                is_allow_unset_source=False,
                **kwargs,
            )
        if self.wrapped is not None:
            with self.wrapped.act_wrapped_update():
                self.wrapped.act_commit(**self._helper_commit_enrich_kwargs_wrapped({}))

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.name!r}), "
            f"is_smooth={self.opts.is_smooth}, is_wrap={self.opts.is_wrap}"
        )


__all__ = ["DisclinationLineSmoothPlot", "OptsDefectLinePlot"]
