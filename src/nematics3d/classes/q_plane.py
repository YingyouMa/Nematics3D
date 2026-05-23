"""Q-tensor interpolation on plane grids with director and defect visualizations."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
from pyvistaqt import BackgroundPlotter

from nematics3d.datatypes import as_bool
from nematics3d.disclination import defect_detect, defect_vicinity_grid
from nematics3d.field import (
    Q_diagonalize,
    align_directors,
    n_color_immerse,
)
from nematics3d.grid import apply_linear_transform
from nematics3d.general import (
    find_rotation_axis,
    mark_points_membership,
    select_grid_in_box,
)
from nematics3d.geometry import wrap_to_pi
from nematics3d.logging_decorator import logging_and_warning_decorator

from .interpolate_plane import InterpolatePlane
from .grid_field import GridInterpolator
from .opts import merge_opts_all
from .plane_grid import OptsPlaneGrid, PlaneGrid
from .plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar
from .result_base import ResultBase
from .visual.plot_figure import OptsFigure, PlotFigure, as_plotfigure
from .visual.plot_rod import OptsRod, PlotRod
from .visual.plot_sphere import OptsSphere, PlotSphere
from .visual.plot_delaunay import OptsDelaunay, PlotDelaunay


@dataclass(slots=True, frozen=True, repr=False)
class OmegaResult(ResultBase):
    """Inspectable result returned by `QPlanePolar.act_calc_omega()`."""

    __result_name__: ClassVar[str] = "local omega"
    __field_docs__: ClassVar[dict[str, str]] = {
        "omega": "Estimated average in-plane rotation axis on the selected polar ring.",
        "metric": "Quality metrics and defect-domain flags returned by the omega fit.",
        "layer": "Polar ring layer index used for the omega evaluation.",
        "num_directors": "Number of sampled directors used on the selected ring.",
        "R": "Physical ring radius associated with the selected layer.",
        "opts": "Frozen copy of the polar-grid opts used to generate the section.",
    }

    omega: np.ndarray
    metric: dict[str, Any]
    layer: int
    num_directors: int
    R: float
    opts: OptsPlaneGridPolar


# QPlane extends InterpolatePlane with Q-tensor-specific post-processing and
# visualization helpers.
#
# Subclasses should preserve the coupling among interpolated Q values, the
# derived director/S caches, defect-detection outputs, and any live visual
# objects. If the recomputation path is overridden, keep `calc_result`,
# `calc_n`, `calc_S`, `calc_is_near_defect`, and `calc_defect_pos`
# synchronized before updating visualization objects.
class QPlane(InterpolatePlane):
    """
    QPlane samples a Q-tensor interpolator on a plane grid and derives
    director, scalar-order, and defect-related quantities on that plane.

    Normal users access the sampled Q values through `plane.result`, the
    derived director and order fields through the plane attributes, and can
    create visual summaries with `act_visualize_n()` and `act_visualize_S()`.
    Use `plane.show_relations()` to inspect the bound grid.
    """

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(InterpolatePlane.__attr_defs__),
        "raw_name": {
            **dict(InterpolatePlane.__attr_defs__["raw_name"]),
            "doc": "The name identifier of this Q-plane object.",
        },
        "calc_result": {
            **dict(InterpolatePlane.__attr_defs__["calc_result"]),
            "kind": "calc",
        },
        "calc_n": {
            "doc": "Director field arrays derived from Q-diagonalization.",
            "kind": "calc",
        },
        "calc_S": {
            "doc": "Scalar-order values derived from Q-diagonalization.",
            "kind": "calc",
        },
        "calc_is_near_defect": {
            "doc": "Boolean mask indicating whether each sampled point is near a defect.",
            "kind": "calc",
        },
        "calc_defect_pos": {
            "doc": "Detected defect positions on this Q plane.",
            "kind": "calc",
        },
        "calc_defect_pos_all": {
            "doc": (
                "Detected defect positions on this Q plane before optional "
                "bounds filtering."
            ),
            "kind": "calc",
        },
        "state_is_interactable": {
            "doc": "Whether to create a control window when the instance is double right-clicked.",
        },
        "default_visual_opts": {
            "doc": "Default opts_defaults_override mappings used for derived visuals.",
        },
        "visual_nb": {
            "doc": "The PlotRod visual showing directors in the bulk region of this Q plane.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "visual_nd": {
            "doc": "The PlotRod visual showing directors near detected defects on this Q plane.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "visual_defect": {
            "doc": "The PlotSphere visual showing detected defect positions on this Q plane.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
        "visual_S": {
            "doc": "The PlotDelaunay visual showing scalar order on this Q plane.",
            "kind": "relation",
            "is_weak_by_default": False,
            "is_weak": None,
            "relation_value": None,
            "doc_runtime": None,
        },
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in InterpolatePlane.__slots__
    )

    _origin_default_visual_opts = {
        "nb": {"color": n_color_immerse, "opacity": 0.2},
        "nd": {"color": n_color_immerse},
        "S": {"scalar_bar_title": "S"},
    }

    # ==================== OVERRIDE ====================
    # QPlane overrides InterpolatePlane.__init__ because it must initialize
    # Q-plane-specific visual state before delegating to the interpolation
    # pipeline and triggering the first defect-aware recomputation.
    # ==================================================
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

        default_visual_opts = deepcopy(self._origin_default_visual_opts)
        visual_default = {} if visual_default is None else dict(visual_default)
        expected_visual_keys = {"nb", "nd", "S"}
        unexpected_visual_keys = set(visual_default) - expected_visual_keys
        if unexpected_visual_keys:
            raise ValueError(
                "`visual_default` must only contain the keys 'nb', 'nd', and 'S'. "
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

        super().__init__(
            interpolator=interpolator,
            name=name,
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            **kwargs,
        )

    # ==================== OVERRIDE ====================
    # QPlane overrides InterpolatePlane.act_refresh because Q-plane updates
    # must diagonalize interpolated tensors, detect defects, refresh derived
    # caches, and then propagate the new data into any live visuals.
    # ==================================================
    def act_refresh(self):

        plane_grid = self.grid

        grid_all = plane_grid.entity_grid_all
        grid_all_flatten = np.reshape(grid_all, (-1, 3))

        Q_all = self.interpolator.interpolate(grid_all_flatten)
        S_all, n_all = Q_diagonalize(Q_all)
        object.__setattr__(self, "calc_n", n_all[plane_grid.calc_box_mask])
        object.__setattr__(self, "calc_S", S_all[plane_grid.calc_box_mask])
        object.__setattr__(self, "calc_result", Q_all[plane_grid.calc_box_mask])

        defect_centers, adjacent_mask = self._helper_detect_defect(n_all)

        object.__setattr__(
            self, "calc_is_near_defect", adjacent_mask[plane_grid.calc_box_mask]
        )
        object.__setattr__(self, "calc_defect_pos_all", defect_centers)

        if defect_centers is None:
            object.__setattr__(self, "calc_defect_pos", None)
        else:
            bounds = plane_grid.bounds
            if bounds is not None:
                _, defect_mask_inside = select_grid_in_box(
                    defect_centers, bounds.corners, is_return_mask=True
                )
                if not plane_grid.opts.is_clip_inside:
                    defect_mask_inside = ~defect_mask_inside
                defect_centers = defect_centers[defect_mask_inside]
            object.__setattr__(self, "calc_defect_pos", defect_centers)

        self._helper_update_visual()

    @logging_and_warning_decorator()
    def _helper_detect_defect(self, n_all, logger=None):

        plane_grid = self.grid
        grid_all = plane_grid.entity_grid_all

        shape_all = np.shape(grid_all)[:2]
        n_all = np.reshape(n_all, (*shape_all, 1, 3))

        defect_plane_index = defect_detect(n_all, planes=(False, False, True))  #!!! pbc
        defect_vicinity_index = defect_vicinity_grid(
            defect_plane_index, num_shell=1
        ).astype(int)
        defect_vicinity_index = defect_vicinity_index.reshape((-1, 3))[:, :-1]
        defect_plane_index = defect_plane_index[:, :-1]
        adjacent_mask = mark_points_membership(
            plane_grid.entity_grid_int.astype(int), defect_vicinity_index
        )

        if len(defect_plane_index) == 0:
            defect_centers = None
        else:
            space1 = plane_grid.opts.spacing
            space2 = (
                space1
                if plane_grid.opts.spacing_extra is None
                else plane_grid.opts.spacing_extra
            )
            step1 = plane_grid.opts.axis1 * space1
            step2 = plane_grid.calc_axis2 * space2
            step_both = np.array([step1, step2])

            defect_centers = (
                np.einsum("ai, ib -> ab", defect_plane_index, step_both)
                + plane_grid.calc_offset_real
            )
            defect_centers = apply_linear_transform(
                defect_centers,
                transform=plane_grid.opts.grid_transform,
                offset=plane_grid.opts.grid_offset,
            )

        return defect_centers, adjacent_mask

    def _helper_set_visual_interact_with_plane(self, visual):
        default_func = getattr(visual, "impl_interact_func", None)

        def _interact():
            if callable(default_func):
                default_func()
            from .visual.qt.interact_plane import InteractPlane

            InteractPlane.show_once(self, visual.fig)

        visual.act_set_interact_func(_interact)

    def _helper_set_visual_interact_with_defect_section(self, visual):
        default_func = getattr(visual, "impl_interact_func", None)

        def _interact():
            if callable(default_func):
                default_func()
            from .visual.qt.interact_defect_section import InteractDefectSection

            InteractDefectSection.show_once(self, visual.fig)

        visual.act_set_interact_func(_interact)

    def _helper_update_visual(self):

        if self.visual_nb or self.visual_nd:

            if np.sum(~self.calc_is_near_defect) > 0:
                self.visual_nb.act_commit(
                    coords=self.grid()[~self.calc_is_near_defect],
                    orient=self.calc_n[~self.calc_is_near_defect],
                    is_visible=True,
                )
            else:
                self.visual_nb.opts.is_visible = False

            if np.sum(self.calc_is_near_defect) > 0:
                self.visual_nd.act_commit(
                    coords=self.grid()[self.calc_is_near_defect],
                    orient=self.calc_n[self.calc_is_near_defect],
                    is_visible=True,
                )
            else:
                self.visual_nd.opts.is_visible = False

            if (
                getattr(self, "calc_defect_pos", None) is not None
                and len(self.calc_defect_pos) > 0
            ):
                self.visual_defect.act_commit(
                    coords=self.calc_defect_pos,
                    is_visible=self.visual_defect.is_show_defect,
                )
            else:
                self.visual_defect.opts.is_visible = False

        if getattr(self, "visual_S", None):
            self.visual_S.act_commit(
                coords=self.grid(),
                scalars=self.calc_S,
            )

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
        """Create or refresh the director and defect visuals for this Q plane."""

        is_defect = as_bool(is_defect, replace=True)

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
                "defect_": opts_defect,
                "nb_": opts_nb,
                "nd_": opts_nd,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_defect = merge["defect_"]
        opts_nb = merge["nb_"]
        opts_nd = merge["nd_"]

        figure = as_plotfigure(figure, opts_figure)

        if np.sum(~self.calc_is_near_defect) > 0:

            visual_nb = PlotRod(
                coords=self.grid()[~self.calc_is_near_defect],
                orient=self.calc_n[~self.calc_is_near_defect],
                name=f"n bulk of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nb,
                figure=figure,
                # Keep the same bounds clip on derived visuals and still register
                # them as bounds subscribers for shared tooling such as silhouette
                # suppression, but skip direct bounds-driven commits because
                # PlaneGrid -> QPlane already drives these updates.
                bounds=self.grid.bounds,
                is_subscribe_bounds=True,
                is_passive_bounds_sync=True,
                opts_defaults_override=self.default_visual_opts["nb"],
            )

        else:

            visual_nb = PlotRod(
                coords=self.grid()[self.calc_is_near_defect],
                orient=self.calc_n[self.calc_is_near_defect],
                name=f"n bulk of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nb,
                figure=figure,
                # Keep the same bounds clip on derived visuals and still register
                # them as bounds subscribers for shared tooling such as silhouette
                # suppression, but skip direct bounds-driven commits because
                # PlaneGrid -> QPlane already drives these updates.
                bounds=self.grid.bounds,
                is_subscribe_bounds=True,
                is_passive_bounds_sync=True,
                opts_defaults_override=self.default_visual_opts["nb"],
                is_visible=False,
            )

        visual_nb.act_bind_relation_base("owner", self, is_weak=True)
        self._helper_set_visual_interact_with_plane(visual_nb)
        self.act_bind_relation_base("visual_nb", visual_nb, is_weak=False)

        if np.sum(self.calc_is_near_defect) > 0:

            visual_nd = PlotRod(
                coords=self.grid()[self.calc_is_near_defect],
                orient=self.calc_n[self.calc_is_near_defect],
                name=f"n near defect of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nd,
                figure=figure,
                # Keep the same bounds clip on derived visuals and still register
                # them as bounds subscribers for shared tooling such as silhouette
                # suppression, but skip direct bounds-driven commits because
                # PlaneGrid -> QPlane already drives these updates.
                bounds=self.grid.bounds,
                is_subscribe_bounds=True,
                is_passive_bounds_sync=True,
                opts_defaults_override=self.default_visual_opts["nd"],
            )

            visual_defect = PlotSphere(
                coords=self.calc_defect_pos,
                name=f"defects of plane {self.name!r}",
                category="plane analysis",
                opts=opts_defect,
                figure=figure,
                # Keep the same bounds clip on derived visuals and still register
                # them as bounds subscribers for shared tooling such as silhouette
                # suppression, but skip direct bounds-driven commits because
                # PlaneGrid -> QPlane already drives these updates.
                bounds=self.grid.bounds,
                is_subscribe_bounds=True,
                is_passive_bounds_sync=True,
            )

        else:

            visual_nd = PlotRod(
                coords=self.grid()[~self.calc_is_near_defect][:2],
                orient=self.calc_n[~self.calc_is_near_defect][:2],
                name=f"n near defect of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nd,
                figure=figure,
                # Keep the same bounds clip on derived visuals and still register
                # them as bounds subscribers for shared tooling such as silhouette
                # suppression, but skip direct bounds-driven commits because
                # PlaneGrid -> QPlane already drives these updates.
                bounds=self.grid.bounds,
                is_subscribe_bounds=True,
                is_passive_bounds_sync=True,
                is_visible=False,
                opts_defaults_override=self.default_visual_opts["nd"],
            )

            visual_defect = PlotSphere(
                coords=self.grid()[~self.calc_is_near_defect][:2],
                name=f"defects of plane {self.name!r}",
                category="plane analysis",
                opts=opts_defect,
                figure=figure,
                # Keep the same bounds clip on derived visuals and still register
                # them as bounds subscribers for shared tooling such as silhouette
                # suppression, but skip direct bounds-driven commits because
                # PlaneGrid -> QPlane already drives these updates.
                bounds=self.grid.bounds,
                is_subscribe_bounds=True,
                is_passive_bounds_sync=True,
                is_visible=False,
            )

        visual_nd.act_bind_relation_base("owner", self, is_weak=True)
        self._helper_set_visual_interact_with_plane(visual_nd)
        self.act_bind_relation_base("visual_nd", visual_nd, is_weak=False)

        visual_defect.act_bind_relation_base("owner", self, is_weak=True)
        self.act_bind_relation_base("visual_defect", visual_defect, is_weak=False)

        visual_defect.act_add_attr(
            "is_show_defect",
            f"Whether to plot defect points during the visualization of directors on {self.name}.",
            default=is_defect,
        )

        visual_defect.opts.is_visible = visual_defect.is_show_defect

    def act_visualize_S(
        self,
        figure: PlotFigure | BackgroundPlotter | None = None,
        opts_figure: OptsFigure | None = None,
        opts_S: OptsDelaunay | None = None,
        **kwargs,
    ):
        """Create or refresh the scalar-order surface visual for this Q plane."""

        if opts_figure is None:
            opts_figure = OptsFigure()
        if opts_S is None:
            opts_S = OptsDelaunay()

        merge = merge_opts_all(
            {
                "figure_": opts_figure,
                "S_": opts_S,
            },
            kwargs,
            type(self).__name__,
        )

        opts_figure = merge["figure_"]
        opts_S = merge["S_"]

        figure = as_plotfigure(figure, opts_figure)

        visual_S = PlotDelaunay(
            coords=self.grid(),
            scalars=self.calc_S,
            figure=figure,
            name=f"S defect of plane {self.name!r}",
            category="plane analysis",
            opts=opts_S,
            # Keep the same bounds clip on derived visuals and still register
            # them as bounds subscribers for shared tooling such as silhouette
            # suppression, but skip direct bounds-driven commits because
            # PlaneGrid -> QPlane already drives these updates.
            bounds=self.grid.bounds,
            is_subscribe_bounds=True,
            is_passive_bounds_sync=True,
            opts_defaults_override=self.default_visual_opts["S"],
        )

        visual_S.act_bind_relation_base("owner", self, is_weak=True)
        self._helper_set_visual_interact_with_plane(visual_S)
        self.act_bind_relation_base("visual_S", visual_S, is_weak=False)


# QPlanePolar specializes QPlane for polar plane grids and ring-based
# defect detection.
#
# Subclasses should preserve the expectation that the bound grid is a
# `PlaneGridPolar`, and keep the ring-offset-based defect traversal aligned
# with the polar grid cache layout.
class QPlanePolar(QPlane):
    """
    QPlanePolar is the polar-grid variant of QPlane.

    It uses `PlaneGridPolar` sampling and a ring-aware defect-detection path
    that is better matched to polar lattice topology. Users interact with it
    through the same main interfaces as `QPlane`.
    """

    __attr_defs__: ClassVar[Mapping[str, dict[str, Any]]] = {
        **dict(QPlane.__attr_defs__),
        "raw_name": {
            **dict(QPlane.__attr_defs__["raw_name"]),
            "doc": "The name identifier of this polar Q-plane object.",
        },
    }
    __slots__ = tuple(
        name
        for name, spec in __attr_defs__.items()
        if spec.get("kind") not in ("relation", "property")
        and name not in QPlane.__slots__
    )

    _origin_default_visual_opts = {
        "nb": {"color": n_color_immerse, "length": 0.6, "radius": 0.06},
        "nd": {"color": n_color_immerse, "length": 0.6, "radius": 0.06},
        "S": {"scalar_bar_title": "S"},
    }

    # ==================== OVERRIDE ====================
    # QPlanePolar overrides QPlane.__init__ because it must ensure a polar
    # plane grid is constructed when the caller does not provide one.
    # ==================================================
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

        if grid is None:
            grid = PlaneGridPolar(
                opts=opts,
                opts_defaults_override=opts_defaults_override,
                name=name + "-grid",
                **kwargs,
            )
            kwargs = {}
            opts = None
            opts_defaults_override = None

        super().__init__(
            interpolator=interpolator,
            name=name,
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override,
            visual_default=visual_default,
            **kwargs,
        )

    # ==================== OVERRIDE ====================
    # QPlanePolar overrides QPlane._helper_set_visual_interact_with_plane because
    # polar-plane visuals should open the defect-section interaction UI by
    # default instead of the Cartesian plane interaction UI.
    # ==================================================
    def _helper_set_visual_interact_with_plane(self, visual):
        self._helper_set_visual_interact_with_defect_section(visual)

    # ==================== OVERRIDE ====================
    # QPlanePolar overrides QPlane._helper_detect_defect because polar grids
    # require ring-aware topology traversal instead of the Cartesian grid
    # defect-detection path used by QPlane.
    # ==================================================
    def _helper_detect_defect(self, directors, threshold: float = 0):

        plane_grid = self.grid
        points = plane_grid.entity_grid_all
        polar = plane_grid.entity_polar
        ring_offsets = plane_grid.calc_ring_offsets

        n_rings = ring_offsets.shape[0] - 1
        adjacent_mask = np.zeros((points.shape[0],), dtype=bool)
        defect_centers_chunks = []

        # If ring 0 is origin block (size 1 at r=0), skip it for quad loops
        start_ring = 0
        if n_rings >= 1:
            s0, e0 = ring_offsets[0], ring_offsets[1]
            if (e0 - s0) == 1 and np.isclose(polar[s0, 0], 0.0):
                start_ring = 1

        # ----------------------------
        # Helper: process one ring-pair (outer -> inner)
        # ----------------------------
        def _process_outer_to_inner(
            s_outer: int, e_outer: int, s_inner: int, e_inner: int
        ) -> None:
            n_outer = e_outer - s_outer
            n_inner = e_inner - s_inner
            if n_outer < 2 or n_inner < 2:
                return

            theta_outer = polar[s_outer:e_outer, 1]  # (n_outer,)
            theta_inner = polar[s_inner:e_inner, 1]  # (n_inner,)

            # base edges on OUTER ring
            j = np.arange(n_outer, dtype=np.int64)
            jn = (j + 1) % n_outer

            idx_a = s_outer + j
            idx_b = s_outer + jn

            theta_a = theta_outer[j]  # (n_outer,)
            theta_b = theta_outer[jn]  # (n_outer,)

            # c: nearest on INNER ring to theta_b (no sorted assumption)
            diff_b = wrap_to_pi(
                theta_inner[None, :] - theta_b[:, None]
            )  # (n_outer, n_inner)
            c_local = np.argmin(np.abs(diff_b), axis=1).astype(np.int64)  # (n_outer,)

            # inner-ring adjacency via sorting theta_inner into a circular order
            order = np.argsort(theta_inner)  # (n_inner,)
            rank_of = np.empty_like(order)
            rank_of[order] = np.arange(n_inner, dtype=np.int64)  # local_index -> rank

            c_rank = rank_of[c_local]  # (n_outer,)
            prev_rank = (c_rank - 1) % n_inner
            next_rank = (c_rank + 1) % n_inner

            prev_local = order[prev_rank]  # (n_outer,)
            next_local = order[next_rank]  # (n_outer,)

            # d: choose neighbor-of-c closer to theta_a
            d_prev = np.abs(wrap_to_pi(theta_inner[prev_local] - theta_a))  # (n_outer,)
            d_next = np.abs(wrap_to_pi(theta_inner[next_local] - theta_a))  # (n_outer,)
            d_local = np.where(d_prev <= d_next, prev_local, next_local).astype(
                np.int64
            )

            idx_c = s_inner + c_local
            idx_d = s_inner + d_local

            # loop vertex coords
            pa = points[idx_a]
            pb = points[idx_b]
            pc = points[idx_c]
            pd = points[idx_d]

            # loop directors
            a = directors[idx_a]
            b_raw = directors[idx_b]
            c_raw = directors[idx_c]
            d_raw = directors[idx_d]

            # align along loop: a -> b -> c -> d
            b = align_directors(a, b_raw)
            c = align_directors(b, c_raw)
            d = align_directors(c, d_raw)

            test = np.einsum("...i,...i->...", a, d)  # (n_outer,)
            hit = test < threshold
            if not np.any(hit):
                return

            centers = (pa + pb + pc + pd) * 0.25
            defect_centers_chunks.append(centers[hit])

            adjacent_mask[idx_a[hit]] = True
            adjacent_mask[idx_b[hit]] = True
            inner_idx = np.unique(np.concatenate([idx_c[hit], idx_d[hit]]))
            adjacent_mask[inner_idx] = True

        # ----------------------------
        # Outer -> inner traversal with stop rule (inner ring < 6)
        # ----------------------------
        outermost = n_rings - 1
        last_good_ring = (
            None  # the ring index (k) of the last ring with >= 6 points before stopping
        )

        # Walk inward: (outer=r, inner=r-1)
        for r in range(outermost, start_ring, -1):
            s_outer, e_outer = ring_offsets[r], ring_offsets[r + 1]
            s_inner, e_inner = ring_offsets[r - 1], ring_offsets[r]

            n_inner = e_inner - s_inner

            # stop condition: we are about to step into a ring with < 6 points
            if n_inner < 6:
                # Current outer ring is the last ring still >= 6 in practice.
                last_good_ring = r
                break

            # process this ring-pair (outer -> inner)
            _process_outer_to_inner(s_outer, e_outer, s_inner, e_inner)

        # If we never hit n_inner < 6, then the innermost ring we reached is start_ring
        if last_good_ring is None:
            last_good_ring = (
                start_ring
                if (ring_offsets[start_ring + 1] - ring_offsets[start_ring]) >= 6
                else None
            )

        # ----------------------------
        # Final-ring internal closure check (only if that ring has >= 6 points)
        # ----------------------------
        if last_good_ring is not None:
            s, e = ring_offsets[last_good_ring], ring_offsets[last_good_ring + 1]
            n_last = e - s

            if n_last >= 6:
                v = directors[s:e]  # (n_last, 3)

                # Vectorized "sequential alignment" using cumulative signs of neighbor dots.
                # NOTE: avoid sign==0 zeroing by using a strict <0 test here.
                dots = np.einsum("ij,ij->i", v[:-1], v[1:])  # (n_last-1,)
                step_sign = np.where(dots < 0.0, -1.0, 1.0).astype(
                    v.dtype
                )  # (n_last-1,)

                cum_sign = np.concatenate(
                    [np.ones((1,), dtype=v.dtype), np.cumprod(step_sign)]
                )  # (n_last,)

                v_aligned_last = v[-1] * cum_sign[-1]
                closure = float(np.dot(v[0], v_aligned_last))

                if closure < threshold:
                    # mark this whole ring as adjacent-to-defect
                    adjacent_mask[s:e] = True

                    # put one defect center for this ring (use mean position)
                    defect_centers_chunks.append(
                        points[s:e].mean(axis=0, keepdims=True)
                    )

        defect_centers = (
            np.concatenate(defect_centers_chunks, axis=0).astype(float)
            if defect_centers_chunks
            else None
        )

        return defect_centers, adjacent_mask

    def _helper_project_defect_radii(self, defect_centers):
        """Project defect centers onto this polar plane and return their radii."""
        if defect_centers is None or len(defect_centers) == 0:
            return np.array([], dtype=float)

        plane_grid = self.grid
        defects_local = apply_linear_transform(
            defect_centers,
            transform=plane_grid.opts.grid_transform,
            offset=plane_grid.opts.grid_offset,
            is_inv=True,
        )
        delta = defects_local - plane_grid.opts.origin
        axis1 = plane_grid.opts.theta0_axis
        axis2 = np.cross(plane_grid.opts.normal, axis1)

        defect_x = delta @ axis1
        defect_y = delta @ axis2
        return np.hypot(defect_x, defect_y)

    def _helper_get_omega_metric_flags(self, radius, out_points):
        """Return diagnostic flags for one omega calculation."""
        defect_radii = self._helper_project_defect_radii(self.calc_defect_pos_all)
        center_tol = max(1e-8, 1e-6 * max(1.0, radius))
        is_defect_center = defect_radii <= center_tol
        is_defect_inside_radius = bool(
            np.any((defect_radii <= radius) & ~is_defect_center)
        )
        is_defect_at_center = bool(np.any(is_defect_center))

        return {
            "is_out_of_domain": len(out_points) > 0,
            "is_defect_inside_R": is_defect_inside_radius,
            "is_defect_at_center": is_defect_at_center,
        }

    @logging_and_warning_decorator()
    def act_calc_omega(self, layer, logger=None):
        """Estimate one average in-plane rotation axis on a selected polar ring."""
        plane_grid = self.grid
        ring_offsets = plane_grid.calc_ring_offsets

        layer = int(layer)
        if layer < 0 or layer >= ring_offsets.shape[0] - 1:
            raise ValueError(
                f"`layer` must be between 0 and {ring_offsets.shape[0] - 2}, "
                f"got {layer}."
            )

        s, e = ring_offsets[layer], ring_offsets[layer + 1]
        if (e - s) < 2:
            raise ValueError(
                f"Layer {layer} contains fewer than 2 directors and cannot define a rotation axis."
            )

        radius = float(plane_grid.entity_polar[s, 0])
        q_layer, out_points = self.interpolator.interpolate(
            plane_grid.entity_grid_all[s:e],
            is_out_warning=True,
        )
        _, directors = Q_diagonalize(q_layer)
        directors = np.asarray(directors, dtype=float).copy()

        for i in range(1, len(directors)):
            directors[i] = align_directors(directors[i - 1], directors[i])

        omega, metric = find_rotation_axis(directors, is_return_metric=True)
        metric_flags = self._helper_get_omega_metric_flags(radius, out_points)

        if not metric_flags["is_defect_at_center"]:
            logger.warning("No defect is detected at the center of this polar plane.")
        if metric_flags["is_defect_inside_R"]:
            logger.warning(
                f"Defects are detected inside or on omega layer {layer} (R={radius})."
            )

        return OmegaResult(
            omega=omega,
            metric={
                **metric,
                **metric_flags,
            },
            layer=layer,
            num_directors=int(len(directors)),
            R=radius,
            opts=deepcopy(plane_grid.opts),
        )
