"""Q-tensor sampling and derived structure on Cartesian and polar planes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..analysis.disclination.plane import (
    detect_defects_on_cartesian_plane,
    detect_defects_on_polar_plane,
)
from ..analysis.disclination.section import OmegaResult, calc_polar_plane_omega
from ..grid.field import GridInterpolator
from ..core.class_base import AttrDef
from ..geometry import select_points_in_box
from ..logging_decorator import logging_and_warning_decorator
from ..q_field.diagonalization import q_diagonalize
from ..visual.color import n_color_immerse
from .interpolate_plane import InterpolatePlane
from .plane_grid import OptsPlaneGrid, PlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar


class QPlane(InterpolatePlane):
    """Sample a Q-tensor field on a rectangular plane and derive local structure."""

    __attr_defs__ = {
        "calc_n_all": AttrDef(
            doc="Directors on the complete plane before bounds filtering.", kind="calc"
        ),
        "calc_S_all": AttrDef(
            doc="Scalar order on the complete plane before bounds filtering.",
            kind="calc",
        ),
        "calc_n": AttrDef(doc="Directors on selected plane points.", kind="calc"),
        "calc_S": AttrDef(doc="Scalar order on selected plane points.", kind="calc"),
        "calc_is_near_defect_all": AttrDef(
            doc="Defect-vicinity mask on the complete plane.", kind="calc"
        ),
        "calc_is_near_defect": AttrDef(
            doc="Defect-vicinity mask on selected plane points.", kind="calc"
        ),
        "calc_defect_pos": AttrDef(
            doc="Detected defect positions selected by the current bounds state.",
            kind="calc",
        ),
        "calc_defect_pos_all": AttrDef(
            doc="Detected defect positions on the complete plane.", kind="calc"
        ),
        "state_is_interactable": AttrDef(
            doc="Whether visualization controls may be opened for this plane.",
            kind="state",
        ),
        "default_visual_opts": AttrDef(
            doc="Default option overrides used by optional Q-plane visuals.",
            kind="default",
        ),
        "visual_nb": AttrDef(
            doc="Optional bulk-director visual.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "visual_nd": AttrDef(
            doc="Optional near-defect director visual.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "visual_defect": AttrDef(
            doc="Optional defect-position visual.",
            kind="relation",
            is_weak_by_default=False,
        ),
        "visual_S": AttrDef(
            doc="Optional scalar-order visual.",
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
        "nb": {"color": n_color_immerse, "opacity": 0.2},
        "nd": {"color": n_color_immerse},
        "S": {"scalar_bar_title": "S"},
    }

    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "Q-plane",
        grid: PlaneGrid | None = None,
        opts: OptsPlaneGrid | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        visual_default: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        self._helper_init_visual_defaults(visual_default)
        super().__init__(
            interpolator=interpolator,
            name=name,
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

    def _helper_init_visual_defaults(self, visual_default):
        defaults = deepcopy(self._origin_default_visual_opts)
        visual_default = {} if visual_default is None else dict(visual_default)
        expected = set(defaults)
        unexpected = set(visual_default) - expected
        if unexpected:
            raise ValueError(
                f"`visual_default` only accepts {sorted(expected)!r}; got {sorted(unexpected)!r}."
            )
        for key in expected:
            override = visual_default.get(key, {})
            if not isinstance(override, Mapping):
                raise TypeError(f"`visual_default[{key!r}]` must be a mapping.")
            defaults[key] = defaults[key] | dict(override)
        object.__setattr__(self, "default_visual_opts", defaults)
        object.__setattr__(self, "state_is_interactable", True)

    def act_refresh(self):
        """Refresh full/selected Q, director, order, and plane-defect results."""
        super().act_refresh()
        diagonalization = q_diagonalize(self.calc_result_all)
        n_all = diagonalization.n
        S_all = diagonalization.S
        mask = self.grid.calc_box_mask

        object.__setattr__(self, "calc_n_all", n_all)
        object.__setattr__(self, "calc_S_all", S_all)
        object.__setattr__(self, "calc_n", n_all[mask])
        object.__setattr__(self, "calc_S", S_all[mask])

        defect_result = self._helper_detect_defects_all(n_all)
        object.__setattr__(
            self, "calc_is_near_defect_all", defect_result.adjacent_mask_all
        )
        object.__setattr__(
            self, "calc_is_near_defect", defect_result.adjacent_mask_all[mask]
        )
        object.__setattr__(self, "calc_defect_pos_all", defect_result.positions_all)
        object.__setattr__(
            self,
            "calc_defect_pos",
            self._helper_select_defect_positions(defect_result.positions_all),
        )
        self._helper_update_visuals_if_loaded()

    def _helper_detect_defects_all(self, directors_all):
        grid = self.grid
        spacing_extra = (
            grid.opts.spacing
            if grid.opts.spacing_extra is None
            else grid.opts.spacing_extra
        )
        return detect_defects_on_cartesian_plane(
            directors_all=directors_all,
            grid_shape=grid.entity_grid_all.shape[:2],
            grid_indices=grid.entity_grid_int,
            origin_grid0=grid.calc_origin_grid0,
            axis1=grid.opts.axis1,
            axis2=grid.calc_axis2,
            spacing=grid.opts.spacing,
            spacing_extra=spacing_extra,
        )

    def _helper_select_defect_positions(self, positions_all):
        if positions_all is None:
            return None
        grid = self.grid
        bounds = grid.bounds if grid.impl_is_bounds_enabled else None
        if bounds is None:
            return positions_all
        _, mask_inside = select_points_in_box(
            positions_all, bounds.corners, is_return_mask=True
        )
        mask = mask_inside if grid.opts.is_clip_inside else ~mask_inside
        return positions_all[mask]

    def _helper_update_visuals_if_loaded(self):
        if any(
            getattr(self, name, None) is not None
            for name in ("visual_nb", "visual_nd", "visual_defect", "visual_S")
        ):
            from ..visual.q_plane import update_q_plane_visuals

            update_q_plane_visuals(self)

    def act_visualize_n(self, *args, **kwargs):
        """Visualize sampled directors; implementation lives in the visual layer."""
        from ..visual.q_plane import visualize_q_plane_n

        visualize_q_plane_n(self, *args, **kwargs)

    def act_visualize_S(self, *args, **kwargs):
        """Visualize scalar order; implementation lives in the visual layer."""
        from ..visual.q_plane import visualize_q_plane_S

        visualize_q_plane_S(self, *args, **kwargs)


class QPlanePolar(QPlane):
    """Sample a Q-tensor field on a polar plane around a local section."""

    _origin_default_visual_opts = {
        "nb": {"color": n_color_immerse, "length": 0.6, "radius": 0.06},
        "nd": {"color": n_color_immerse, "length": 0.6, "radius": 0.06},
        "S": {"scalar_bar_title": "S"},
    }

    def __init__(
        self,
        interpolator: GridInterpolator,
        name: str = "Q-plane (polar)",
        grid: PlaneGridPolar | None = None,
        opts: OptsPlaneGridPolar | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        visual_default: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        if grid is not None and not isinstance(grid, PlaneGridPolar):
            raise TypeError("`grid` for QPlanePolar must be a PlaneGridPolar.")
        if opts is not None and not isinstance(opts, OptsPlaneGridPolar):
            raise TypeError("`opts` for QPlanePolar must be OptsPlaneGridPolar.")
        self._helper_init_visual_defaults(visual_default)
        InterpolatePlane.__init__(
            self,
            interpolator=interpolator,
            name=name,
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

    def _helper_detect_defects_all(self, directors_all):
        grid = self.grid
        return detect_defects_on_polar_plane(
            points_all=grid.entity_grid_all,
            polar_coords=grid.entity_polar,
            ring_offsets=grid.calc_ring_offsets,
            directors_all=directors_all,
        )

    @logging_and_warning_decorator()
    def act_calc_omega(self, layer, logger=None):
        """Estimate the average director rotation axis on one polar ring."""
        return calc_polar_plane_omega(
            grid=self.grid,
            interpolator=self.interpolator,
            defect_positions_all=self.calc_defect_pos_all,
            layer=layer,
            logger=logger,
        )


__all__ = ["OmegaResult", "QPlane", "QPlanePolar"]
