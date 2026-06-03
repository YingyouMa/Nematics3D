import numpy as np
from pyvistaqt import BackgroundPlotter

from copy import deepcopy
from typing import Any, Mapping

from .class_base import AttrDef
from .interpolate_surface import InterpolateSurface
from .opts import merge_opts_all
from .surface_sampling import SurfaceSampling, OptsSurfaceSampling
from .visual.plot_figure import OptsFigure, PlotFigure, as_plotfigure
from .visual.plot_rod import OptsRod, PlotRod
from nematics3d.field import Q_diagonalize, n_color_immerse
from .grid_field import GridInterpolator


class QSurface(InterpolateSurface):
    """
    QSurface samples a Q-tensor interpolator on a sampled surface and derives
    director and scalar-order quantities.

    This first version intentionally does not perform defect detection and only
    provides director visualization. S visualization can be added later.
    """

    __attr_defs__ = {
        "calc_n": AttrDef(
            doc="Director field arrays derived from Q-diagonalization.",
            kind="calc",
        ),
        "calc_S": AttrDef(
            doc="Scalar-order values derived from Q-diagonalization.",
            kind="calc",
        ),
        "state_is_interactable": AttrDef(
            doc="Whether to create a control window when the instance is double right-clicked.",
            kind="state",
        ),
        "default_visual_opts": AttrDef(
            doc="Default opts_defaults_override mappings used for derived visuals.",
            kind="default",
        ),
        "visual_n": AttrDef(
            doc="The PlotRod visual showing directors on this Q surface.",
            kind="relation",
            is_weak_by_default=False,
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in InterpolateSurface.__slots__
    )

    _origin_default_visual_opts = {
        "n": {"color": n_color_immerse, "opacity": 1},
    }

    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "Q-surface",
        sampling: SurfaceSampling | None = None,
        surface=None,
        opts: OptsSurfaceSampling | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        visual_default: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        default_visual_opts = deepcopy(self._origin_default_visual_opts)
        visual_default = {} if visual_default is None else dict(visual_default)

        unexpected_visual_keys = set(visual_default) - {"n"}
        if unexpected_visual_keys:
            raise ValueError(
                "`visual_default` must only contain the key 'n'. "
                f"Got unexpected keys: {sorted(unexpected_visual_keys)!r}."
            )

        override = visual_default.get("n", {})
        if not isinstance(override, Mapping):
            raise TypeError(
                "`visual_default['n']` must be a mapping of option overrides."
            )
        default_visual_opts["n"] = default_visual_opts["n"] | dict(override)

        object.__setattr__(self, "default_visual_opts", default_visual_opts)
        object.__setattr__(self, "state_is_interactable", True)

        super().__init__(
            interpolator=interpolator,
            name=name,
            sampling=sampling,
            surface=surface,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

    def act_refresh(self):
        coords = self.sampling.result
        Q = self.interpolator.interpolate(coords)
        S, n = Q_diagonalize(Q)

        object.__setattr__(self, "calc_result", Q)
        object.__setattr__(self, "calc_S", S)
        object.__setattr__(self, "calc_n", n)

        self._helper_update_visual()

    def _helper_update_visual(self):
        if getattr(self, "visual_n", None):
            self.visual_n.act_commit(
                coords=self.sampling.result,
                orient=self.calc_n,
            )

    def _helper_set_visual_interact_with_sampling(self, visual):
        default_func = getattr(visual, "impl_interact_func", None)

        def _interact():
            if callable(default_func):
                default_func()
            from .visual.qt.interact_surface_sampling import (
                InteractSurfaceSampling,
            )

            InteractSurfaceSampling.show_once(self.sampling, visual.fig)

        visual.act_set_interact_func(_interact)

    def act_visualize_n(
        self,
        figure: PlotFigure | BackgroundPlotter | None = None,
        opts_figure: OptsFigure | None = None,
        opts_n: OptsRod | None = None,
        **kwargs,
    ):
        """Create or refresh the director visual for this Q surface."""

        if opts_n is None:
            opts_n = OptsRod()
        if opts_figure is None:
            opts_figure = OptsFigure()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "n_": opts_n,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_n = merge["n_"]

        figure = as_plotfigure(figure, opts_figure)

        visual_n = PlotRod(
            coords=self.sampling.result,
            orient=self.calc_n,
            name=f"n of surface {self.name!r}",
            category="surface analysis",
            opts=opts_n,
            figure=figure,
            opts_defaults_override=self.default_visual_opts["n"],
        )

        visual_n.act_bind_relation_base("owner", self, is_weak=True)
        self.act_bind_relation_base("visual_n", visual_n, is_weak=False)
        self._helper_set_visual_interact_with_sampling(visual_n)

        return visual_n
