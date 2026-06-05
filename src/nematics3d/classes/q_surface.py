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
from .visual.plot_sphere import OptsSphere, PlotSphere
from nematics3d.field import Q_diagonalize, n_color_immerse
from nematics3d.geometry import triangulate_surface_points
from nematics3d.disclination import defect_detect_surface
from .grid_field import GridInterpolator


class QSurface(InterpolateSurface):
    """
    QSurface samples a Q-tensor interpolator on a sampled surface and derives
    director, scalar-order, and defect-related quantities.

    Defect positions are detected automatically on each refresh using
    ``defect_detect_surface``.  Directors are split into a bulk group
    (semi-transparent) and a near-defect group (opaque), matching the
    visual convention used by ``QPlane``.
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
        "calc_surface_mesh": AttrDef(
            doc="Triangulated PolyData mesh built from the current sample points.",
            kind="calc",
        ),
        "calc_defect_pos": AttrDef(
            doc="Detected defect positions on this Q surface.",
            kind="calc",
        ),
        "calc_is_near_defect": AttrDef(
            doc="Boolean mask indicating whether each sampled point is near a defect.",
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
        "visual_nb": AttrDef(
            doc="The PlotRod visual showing bulk directors on this Q surface.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "visual_nd": AttrDef(
            doc="The PlotRod visual showing directors near defects on this Q surface.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "visual_defect": AttrDef(
            doc="The PlotSphere visual showing detected defect positions on this Q surface.",
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
        "nb": {"color": n_color_immerse, "opacity": 0.2},
        "nd": {"color": n_color_immerse, "opacity": 1},
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

        expected_visual_keys = {"nb", "nd"}
        unexpected_visual_keys = set(visual_default) - expected_visual_keys
        if unexpected_visual_keys:
            raise ValueError(
                "`visual_default` must only contain the keys 'nb' and 'nd'. "
                f"Got unexpected keys: {sorted(unexpected_visual_keys)!r}."
            )

        for key in expected_visual_keys:
            override = visual_default.get(key, {})
            if not isinstance(override, Mapping):
                raise TypeError(
                    f"`visual_default[{key!r}]` must be a mapping of option overrides."
                )
            default_visual_opts[key] = default_visual_opts[key] | dict(override)

        object.__setattr__(self, "default_visual_opts", default_visual_opts)
        object.__setattr__(self, "state_is_interactable", True)
        object.__setattr__(self, "calc_surface_mesh", None)
        object.__setattr__(self, "calc_defect_pos", np.empty((0, 3), dtype=float))
        object.__setattr__(self, "calc_is_near_defect", np.empty(0, dtype=bool))

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

        # Build triangulated mesh from sample points for defect detection.
        mesh = triangulate_surface_points(coords)

        # Interpolate Q tensor and derive director and scalar order.
        Q = self.interpolator.interpolate(coords)
        S, n = Q_diagonalize(Q)

        # Detect defects and mark near-defect directors.
        # Pass the pre-computed director array to avoid a second interpolation.
        defect_pos, near_defect_mask = defect_detect_surface(
            mesh,
            n,
            is_return_mask=True,
        )

        object.__setattr__(self, "calc_surface_mesh", mesh)
        object.__setattr__(self, "calc_result", Q)
        object.__setattr__(self, "calc_S", S)
        object.__setattr__(self, "calc_n", n)
        object.__setattr__(self, "calc_defect_pos", defect_pos)
        object.__setattr__(self, "calc_is_near_defect", near_defect_mask)

        self._helper_update_visual()

    def _helper_update_visual(self):
        coords = self.sampling.result
        is_near = self.calc_is_near_defect

        if getattr(self, "visual_nb", None):
            bulk_mask = ~is_near
            if np.any(bulk_mask):
                self.visual_nb.act_commit(
                    coords=coords[bulk_mask],
                    orient=self.calc_n[bulk_mask],
                    is_visible=True,
                )
            else:
                self.visual_nb.opts.is_visible = False

        if getattr(self, "visual_nd", None):
            if np.any(is_near):
                self.visual_nd.act_commit(
                    coords=coords[is_near],
                    orient=self.calc_n[is_near],
                    is_visible=True,
                )
            else:
                self.visual_nd.opts.is_visible = False

        if getattr(self, "visual_defect", None):
            if self.calc_defect_pos is not None and len(self.calc_defect_pos) > 0:
                self.visual_defect.act_commit(
                    coords=self.calc_defect_pos,
                )
            else:
                self.visual_defect.opts.is_visible = False

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
        opts_nb: OptsRod | None = None,
        opts_nd: OptsRod | None = None,
        opts_defect: OptsSphere | None = None,
        is_defect: bool = False,
        **kwargs,
    ):
        """Create or refresh the director and defect visuals for this Q surface."""

        if opts_nb is None:
            opts_nb = OptsRod()
        if opts_nd is None:
            opts_nd = OptsRod()
        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_defect is None:
            opts_defect = OptsSphere()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "nb_": opts_nb,
                "nd_": opts_nd,
                "defect_": opts_defect,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_nb = merge["nb_"]
        opts_nd = merge["nd_"]
        opts_defect = merge["defect_"]

        figure = as_plotfigure(figure, opts_figure)

        coords = self.sampling.result
        is_near = self.calc_is_near_defect
        bulk_mask = ~is_near

        # --- visual_nb: bulk directors (semi-transparent) ---
        if np.any(bulk_mask):
            visual_nb = PlotRod(
                coords=coords[bulk_mask],
                orient=self.calc_n[bulk_mask],
                name=f"n bulk of surface {self.name!r}",
                category="surface analysis",
                opts=opts_nb,
                figure=figure,
                opts_defaults_override=self.default_visual_opts["nb"],
            )
        else:
            visual_nb = PlotRod(
                coords=coords[:2],
                orient=self.calc_n[:2],
                name=f"n bulk of surface {self.name!r}",
                category="surface analysis",
                opts=opts_nb,
                figure=figure,
                opts_defaults_override=self.default_visual_opts["nb"],
                is_visible=False,
            )

        visual_nb.act_bind_relation_base("owner", self, is_weak=True)
        self.act_bind_relation_base("visual_nb", visual_nb, is_weak=False)
        self._helper_set_visual_interact_with_sampling(visual_nb)

        # --- visual_nd: near-defect directors (opaque) ---
        if np.any(is_near):
            visual_nd = PlotRod(
                coords=coords[is_near],
                orient=self.calc_n[is_near],
                name=f"n near defect of surface {self.name!r}",
                category="surface analysis",
                opts=opts_nd,
                figure=figure,
                opts_defaults_override=self.default_visual_opts["nd"],
            )
        else:
            visual_nd = PlotRod(
                coords=coords[:2],
                orient=self.calc_n[:2],
                name=f"n near defect of surface {self.name!r}",
                category="surface analysis",
                opts=opts_nd,
                figure=figure,
                opts_defaults_override=self.default_visual_opts["nd"],
                is_visible=False,
            )

        visual_nd.act_bind_relation_base("owner", self, is_weak=True)
        self.act_bind_relation_base("visual_nd", visual_nd, is_weak=False)
        self._helper_set_visual_interact_with_sampling(visual_nd)

        # --- visual_defect: defect spheres ---
        defect_pos = self.calc_defect_pos
        has_defects = defect_pos is not None and len(defect_pos) > 0

        visual_defect = PlotSphere(
            coords=defect_pos if has_defects else coords[:2],
            name=f"defects of surface {self.name!r}",
            category="surface analysis",
            opts=opts_defect,
            figure=figure,
            is_visible=is_defect and has_defects,
        )

        visual_defect.act_bind_relation_base("owner", self, is_weak=True)
        self.act_bind_relation_base("visual_defect", visual_defect, is_weak=False)

        return visual_nb
