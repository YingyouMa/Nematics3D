from pyvistaqt import BackgroundPlotter
import numpy as np
import weakref
from typing import Mapping, Any

from .Interpolator import Interpolator
from Nematics3D.field import Q_diagonalize, n_color_immerse, apply_linear_transform
from Nematics3D.disclination import defect_detect, defect_vicinity_grid
from Nematics3D.general import select_grid_in_box, mark_points_membership
from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import as_bool, as_str
from .visual.plot_figure import PlotFigure, OptsFigure
from .visual.plot_sphere import PlotSphere, OptsSphere
from .visual.plot_rod import PlotRod, OptsRod
from .visual.plot_delaunay import OptsDelaunay, PlotDelaunay
from .opts import merge_opts_all
from .plane_grid import PlaneGrid, OptsPlaneGrid
from .interpolate_plane import InterpolatePlane


class QPlane(InterpolatePlane):

    __descriptions__ = {
        **(InterpolatePlane.__descriptions__),
        
        "raw_name": "The name identifier of this Q-plane object",
        "_entity_visual_nb": "The PlotRod objects of visualized directors in the bulk",
        "_entity_visual_nd": "The PlotRod objects of visualized directors near defects",
        "_entity_visual_defect": "The PlotSphere objects of visualized defects",
        "_entity_visual_S": "The PlotDealunay object of visualized S",
        "_calc_n": "List of director field arrays (from Q-diagonalization)",
        "_calc_S": "List of S field arrays (from Q-diagonalization)",
        "_calc_is_near_defect": "The flag indicating whether the local direcor surrounds a defect",
        "_calc_defect_pos": "The positions of defects on this n-plane",
        "_state_is_interactable": "Whether to create a control window when the instance is double right-clicked.",
    }

    __slots__ = tuple(
            k for k, v in __descriptions__.items() 
            if not v.startswith("Property:") and k not in InterpolatePlane.__slots__
        )

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        interpolator: Interpolator,
        name: str = "Q-plane",
        grid: PlaneGrid | None = None,
        opts: OptsPlaneGrid | None = None,
        opts_defaults_override: Mapping[str, Any] | None = None,
        logger=None,
        **kwargs,
    ):
        
        object.__setattr__(self, '_entity_visual_nb', None)
        object.__setattr__(self, '_entity_visual_nd', None)
        object.__setattr__(self, '_entity_visual_defect', None)
        object.__setattr__(self, '_entity_visual_S', None)
        object.__setattr__(self, '_state_is_interactable', True)
        
        super().__init__(
            interpolator=interpolator,
            name=name,
            grid=grid,
            opts=opts,
            opts_defaults_override=opts_defaults_override
            )

        self._helper_commit()
        

        

    @logging_and_warning_decorator()
    def _helper_commit(self, logger=None):

        plane_grid = self._entity_plane

        logger.detail("Retrieving the full grid in lattice index structure ...")
        grid_all = plane_grid._entity_grid_all
        shape_all = np.shape(grid_all)[:2]
        grid_all_flatten = np.reshape(grid_all, (-1, 3))

        logger.detail("Interpolating ...")
        Q_all = self._raw_interpolator.interpolate(grid_all_flatten)
        S_all, n_all = Q_diagonalize(Q_all)
        object.__setattr__(self, '_calc_n', n_all[plane_grid._calc_box_mask])
        object.__setattr__(self, '_calc_S', S_all[plane_grid._calc_box_mask])
        object.__setattr__(self, '_calc_result', Q_all[plane_grid._calc_box_mask])
        
        n_all = np.reshape(n_all, (*shape_all, 1, 3))

        logger.detail("Detecting the defects and surrounding directors ...")
        defect_plane_index = defect_detect(n_all, planes=(False, False, True))  #!!! pbc
        defect_vicinity_index = defect_vicinity_grid(
            defect_plane_index, num_shell=1
        ).astype(int)
        defect_vicinity_index = defect_vicinity_index.reshape((-1, 3))[:, :-1]
        defect_plane_index = defect_plane_index[:, :-1]
        mask_near_defect = mark_points_membership(
            plane_grid._entity_grid_int.astype(int), defect_vicinity_index
        )
        object.__setattr__(self, '_calc_is_near_defect', mask_near_defect[plane_grid._calc_box_mask])

        if len(defect_plane_index)==0:
            object.__setattr__(self, '_calc_defect_pos', None)
        else:
            logger.detail("Switching the lattice indices of defects into real space units ...")
            
            space1 = plane_grid.opts.spacing
            space2 = (
                space1
                if plane_grid.opts.spacing_extra is None
                else plane_grid.opts.spacing_extra
            )
            step1 = plane_grid.opts.axis1 * space1
            step2 = plane_grid._calc_axis2 * space2
            step_both = np.array([step1, step2])
            
            defect_pos = (
                np.einsum("ai, ib -> ab", defect_plane_index, step_both)
                + plane_grid._calc_offset_real
            )
            defect_pos = apply_linear_transform(
                defect_pos,
                transform=plane_grid.opts.grid_transform,
                offset=plane_grid.opts.grid_offset,
            )
            defect_pos = select_grid_in_box(defect_pos, plane_grid.opts.corners_limit)
            object.__setattr__(self, '_calc_defect_pos', defect_pos)
            
        if self._entity_visual_nb or self._entity_visual_nd:
            
            if np.sum(~self._calc_is_near_defect) > 0:
                self._entity_visual_nb.act_commit(       
                    coords=self._entity_plane()[~self._calc_is_near_defect],
                    orient=self._calc_n[~self._calc_is_near_defect],
                    is_silhouette=self._state_is_interactable,
                    is_visible=True
                    )
            else:
                self._entity_visual_nb.opts.is_visible = False
            
            if np.sum(self._calc_is_near_defect) > 0:
                self._entity_visual_nd.act_commit(       
                    coords=self._entity_plane()[self._calc_is_near_defect],
                    orient=self._calc_n[self._calc_is_near_defect],
                    is_silhouette=self._state_is_interactable,
                    is_visible=True
                    )
            else:
                self._entity_visual_nd.opts.is_visible = False
                
                
            if getattr(self, "_calc_defect_pos", None) is not None and len(self._calc_defect_pos)>0:   
                self._entity_visual_defect.act_commit( 
                    coords=self._calc_defect_pos,
                    is_silhouette=self._state_is_interactable,
                    is_visible=self._entity_visual_defect.is_show_defect
                    )
            else:
                self._entity_visual_defect.opts.is_visible = False
                
        if getattr(self, "_entity_visual_S", None):
            self._entity_visual_S.act_commit(
                coords=self.plane(),
                scalars=self._calc_S,
                is_silhouette=self._state_is_interactable,
                )
            
            
            

    @logging_and_warning_decorator()
    def act_visualize_n(
        self,
        figure: PlotFigure | BackgroundPlotter | None = None,
        opts_figure: OptsFigure | None = None,
        opts_nb: OptsRod | None = None,
        opts_nd: OptsRod | None = None,
        opts_defect: OptsSphere | None = None,
        is_defect: bool = False,
        logger=None,
        **kwargs,
    ):

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

        try:
            if figure is None:
                figure = PlotFigure(opts=opts_figure)
            elif isinstance(figure, PlotFigure):
                figure.act_commit(opts_figure)
            elif isinstance(figure, BackgroundPlotter):
                figure = PlotFigure(plotter=figure, opts=opts_figure)
            else:
                raise ValueError(
                    "`figure` input must be a valid PlotFigure object, "
                    "or a valid pyvista plotter object "
                    "or None (creating a new figure) "
                    "Got type {type(figure)!r} instead."
                )
        except:
            logger.exception("Invalid figure input")
            logger.recovery("Create a new figure instead.")
            figure = PlotFigure(opts=opts_figure)
            
        if np.sum(~self._calc_is_near_defect) > 0:
            
            visual_nb = PlotRod(
                coords=self._entity_plane()[~self._calc_is_near_defect],
                orient=self._calc_n[~self._calc_is_near_defect],
                name=f"n bulk of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nb,
                figure=figure,
                opts_defaults_override={"color": n_color_immerse, "opacity": 0.2}
            )
            
        else:
            
            visual_nb = PlotRod(
                coords=self._entity_plane()[self._calc_is_near_defect],
                orient=self._calc_n[self._calc_is_near_defect],
                name=f"n bulk of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nb,
                figure=figure,
                opts_defaults_override={"color": n_color_immerse, "opacity": 0.2},
                is_visible=False
            )
            
        object.__setattr__(visual_nb, "_impl_owner_ref", weakref.ref(self))
        object.__setattr__(self, '_entity_visual_nb', visual_nb)

        if np.sum(self._calc_is_near_defect) > 0:
            
            visual_nd = PlotRod(
                coords=self._entity_plane()[self._calc_is_near_defect],
                orient=self._calc_n[self._calc_is_near_defect],
                name=f"n near defect of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nd,
                figure=figure,
                opts_defaults_override={"color": n_color_immerse}
            )
            
            visual_defect = PlotSphere(
                coords=self._calc_defect_pos, 
                name=f"defects of plane {self.name!r}",
                category="plane analysis",
                opts=opts_defect, 
                figure=figure
            )
            
        else:
            
            visual_nd = PlotRod(
                coords=self._entity_plane()[~self._calc_is_near_defect][:2],
                orient=self._calc_n[~self._calc_is_near_defect][:2],
                name=f"n near defect of plane {self.name!r}",
                category="plane analysis",
                opts=opts_nd,
                figure=figure,
                is_visible=False,
                opts_defaults_override={"color": n_color_immerse}
            )
            
            visual_defect = PlotSphere(
                coords=self._entity_plane()[~self._calc_is_near_defect][:2], 
                name=f"defects of plane {self.name!r}",
                category="plane analysis",
                opts=opts_defect, 
                figure=figure,
                is_visible=False
            )
            
            
        object.__setattr__(visual_nd, "_impl_owner_ref", weakref.ref(self))
        object.__setattr__(self, '_entity_visual_nd', visual_nd)
        
        object.__setattr__(visual_defect, "_impl_owner_ref", weakref.ref(self))
        object.__setattr__(self, '_entity_visual_defect', visual_defect)
            
        visual_defect.act_add_attr(
            "is_show_defect", 
            f"Whether to plot defect points during the visualization of directors on {self.name}.",
            default=is_defect
            )
        
        visual_defect.opts.is_visible = visual_defect.is_show_defect


    def act_visualize_S(
        self,
        figure: PlotFigure | BackgroundPlotter | None = None,
        opts_figure: OptsFigure | None = None,
        opts_S: OptsDelaunay | None = None,
        logger=None,
        **kwargs,
    ):
    
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

        try:
            if figure is None:
                figure = PlotFigure(opts=opts_figure)
            elif isinstance(figure, PlotFigure):
                figure.act_commit(opts_figure)
            elif isinstance(figure, BackgroundPlotter):
                figure = PlotFigure(plotter=figure, opts=opts_figure)
            else:
                raise ValueError(
                    "`figure` input must be a valid PlotFigure object, "
                    "or a valid pyvista plotter object "
                    "or None (creating a new figure) "
                    "Got type {type(figure)!r} instead."
                )
        except:
            logger.exception("Invalid figure input")
            logger.recovery("Create a new figure instead.")
            figure = PlotFigure(opts=opts_figure)
            
        visual_S = PlotDelaunay(
            coords=self.plane(),
            scalars=self._calc_S,
            figure=figure,
            name=f"S defect of plane {self.name!r}",
            category="plane analysis",
            )
        
        object.__setattr__(visual_S, "_impl_owner_ref", weakref.ref(self))
        object.__setattr__(self, '_entity_visual_S', visual_S)