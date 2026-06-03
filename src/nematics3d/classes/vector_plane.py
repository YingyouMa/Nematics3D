"""Vector-field interpolation on physical-space plane grids."""

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from pyvistaqt import BackgroundPlotter

from .class_base import AttrDef
from .grid_field import GridInterpolator
from .interpolate_plane import InterpolatePlane
from .opts import merge_opts_all
from .plane_grid import OptsPlaneGrid, PlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar
from .visual.plot_figure import OptsFigure, PlotFigure, as_plotfigure
from .visual.plot_vector import OptsVector, PlotVector


# VectorPlane extends InterpolatePlane with vector-field-specific validation and
# vector-glyph visualization helpers.
#
# Subclasses should preserve the expectation that `calc_result` stores one
# 3-vector per selected plane point, keep `calc_magnitude` synchronized with
# `calc_result`, and refresh any live vector visual after resampling.
class VectorPlane(InterpolatePlane):
    """
    VectorPlane samples a vector-valued interpolator on a plane grid.

    The sampled vectors are available through `plane.result`, while
    `plane.calc_magnitude` stores the per-point vector norms. For Cartesian
    `PlaneGrid`, those vectors are sampled directly at the grid's physical-space
    plane coordinates. Use `act_visualize_vector()` to render the sampled plane
    field with arrow glyphs, and `plane.show_relations()` to inspect the bound
    grid.
    """

    __attr_defs__ = {
        "calc_magnitude": AttrDef(
            doc="Vector magnitudes derived from the sampled plane vectors.",
            kind="calc",
        ),
        "state_is_interactable": AttrDef(
            doc="Whether to create a control window when the instance is double right-clicked.",
            kind="state",
        ),
        "default_visual_opts": AttrDef(
            doc="Default opts_defaults_override mappings used for vector visuals.",
            kind="default",
        ),
        "visual": AttrDef(
            doc="The PlotVector visual showing sampled vectors on this plane.",
            kind="relation",
            is_weak_by_default=False,
        ),
    }

    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.kind not in ("relation", "property", "opts")
        and name not in InterpolatePlane.__slots__
    )

    _origin_default_visual_opts = {
        "vector": {
            "resolver_source": "orient",
            "length": lambda orient: np.linalg.norm(orient, axis=1),
        }
    }

    # ==================== OVERRIDE ====================
    # VectorPlane overrides InterpolatePlane.__init__ because it must initialize
    # vector-visual state before the first interpolation refresh.
    # ==================================================
    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "vector-plane",
        grid: PlaneGrid | PlaneGridPolar | None = None,
        opts: OptsPlaneGrid | OptsPlaneGridPolar | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        visual_default: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        default_visual_opts = deepcopy(self._origin_default_visual_opts)
        visual_default = {} if visual_default is None else dict(visual_default)
        unexpected_visual_keys = set(visual_default) - {"vector"}
        if unexpected_visual_keys:
            raise ValueError(
                "`visual_default` must only contain the key 'vector'. "
                f"Got unexpected keys: {sorted(unexpected_visual_keys)!r}."
            )

        override = visual_default.get("vector", {})
        if not isinstance(override, Mapping):
            raise TypeError(
                "`visual_default['vector']` must be a mapping of option overrides."
            )
        default_visual_opts["vector"] = default_visual_opts["vector"] | dict(override)

        object.__setattr__(self, "default_visual_opts", default_visual_opts)
        object.__setattr__(self, "state_is_interactable", True)

        super().__init__(
            interpolator=interpolator,
            name=name,
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

    # ==================== OVERRIDE ====================
    # VectorPlane overrides InterpolatePlane.act_refresh because it must verify
    # that the sampled field is vector-valued, cache magnitudes, and then
    # propagate updates into any live vector visual.
    # ==================================================
    def act_refresh(self):
        super().act_refresh()

        result = np.asarray(self.calc_result, dtype=float)
        if result.ndim != 2 or result.shape[1] != 3:
            raise ValueError(
                "VectorPlane requires interpolated values with shape (N, 3). "
                f"Got shape {result.shape} instead."
            )

        object.__setattr__(self, "calc_result", result)
        object.__setattr__(self, "calc_magnitude", np.linalg.norm(result, axis=1))

        self._helper_update_visual()

    def _helper_set_visual_interact_with_plane(self, visual):
        default_func = getattr(visual, "impl_interact_func", None)

        def _interact():
            if callable(default_func):
                default_func()
            from .visual.qt.interact_plane import InteractPlane

            InteractPlane.show_once(self, visual.fig)

        visual.act_set_interact_func(_interact)

    def _helper_update_visual(self):
        if getattr(self, "visual", None):
            self.visual.act_commit(
                coords=self.grid(),
                orient=self.result,
            )

    def act_visualize_vector(
        self,
        figure: PlotFigure | BackgroundPlotter | None = None,
        opts_figure: OptsFigure | None = None,
        opts_vector: OptsVector | None = None,
        **kwargs,
    ):
        """Create or refresh the vector-glyph visual for this sampled plane."""

        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_vector is None:
            opts_vector = OptsVector()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "": opts_vector,
            },
            kwargs,
            type(self).__name__,
        )
        opts_figure = merge["figure_"]
        opts_vector = merge[""]

        figure = as_plotfigure(figure, opts_figure)

        visual = PlotVector(
            coords=self.grid(),
            orient=self.result,
            name=f"vector field of plane {self.name!r}",
            category="plane analysis",
            opts=opts_vector,
            figure=figure,
            bounds=self.grid.bounds,
            is_subscribe_bounds=True,
            is_passive_bounds_sync=True,
            opts_defaults_override=self.default_visual_opts["vector"],
        )

        visual.act_bind_relation_base("owner", self, is_weak=True)
        self._helper_set_visual_interact_with_plane(visual)
        self.act_bind_relation_base("visual", visual, is_weak=False)

        return visual
