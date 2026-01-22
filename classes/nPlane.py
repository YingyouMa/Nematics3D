import pyvista as pv
import numpy as np
import weakref

from .Interpolator import Interpolator
from Nematics3D.field import Q_diagonalize, n_color_immerse, apply_linear_transform
from Nematics3D.disclination import defect_detect, defect_vicinity_grid
from Nematics3D.general import select_grid_in_box, mark_points_membership
from Nematics3D.logging_decorator import logging_and_warning_decorator
from Nematics3D.datatypes import as_bool, as_str
from .visual.plot_figure import PlotFigure, OptsFigure
from .visual.plot_sphere import PlotSphere, OptsSphere
from .visual.plot_rod import PlotRod, OptsRod
from .opts import merge_opts_all
from .plane_grid import PlaneGrid, OptsPlaneGrid

#!!! class name

class DirectorPlane:

    __descriptions__ = {
        "name": "The name identifier of this n-plane object",
        "_raw_QInterpolator": "Interpolator object for Q-tensor field (class Interpolator)",
        "_entity_plane": "The PlaneGrid entity (coordinates of 2D lattice)",
        "_entity_visual_nb": "The PlotRod objects of visualized directors in the bulk",
        "_entity_visual_nd": "The PlotRod objects of visualized directors near defects",
        "_entity_visual_defect": "The PlotSphere objects of visualized defects",
        "_calc_n": "List of director field arrays (from Q-diagonalization)",
        "_calc_is_near_defect": "The flag indicating whether the local direcor surrounds a defect",
        "_calc_defect_pos": "The positions of defects on this n-plane",
    }

    __slots__ = tuple(__descriptions__.keys()) + ("__weakref__",)

    @logging_and_warning_decorator(start_finish_level=5)
    def __init__(
        self,
        QInterpolator: Interpolator,
        name: str = "n plane",
        grid: PlaneGrid | None = None,
        opts_grid: OptsPlaneGrid | None = None,
        logger=None,
        **kwargs,
    ):
        
        name = as_str(name, name="The name identifier of this n-plane object", replace="n plane")
        self.name = name
        
        if opts_grid is None:
            opts_grid = OptsPlaneGrid()
        elif not isinstance(opts_grid, OptsPlaneGrid):
            try:
                raise TypeError(
                        f"opts must be an instance of {OptsPlaneGrid.__name__}, "
                        f"got {type(opts_grid).__name__}"
                    )
            except TypeError:
                logger.exception("Check input.")
                logger.recovery("Automatically ignore this input")
                opts_grid = OptsPlaneGrid()
        
        opts_grid = merge_opts_all({"": opts_grid}, kwargs, type(self).__name__)[""]

        if grid is None:
            self._entity_plane = PlaneGrid(opts=opts_grid)
        elif isinstance(grid, PlaneGrid):
            grid.act_commit(opts_grid)
            self._entity_plane = grid
        else:
            try:
                raise TypeError(
                    "`grid` must be PlaneGrid object or None (create a new PlaneGrid object)"
                    f"Got {type(grid)} instead."
                )
            except TypeError:
                logger.exception("Check input grid.")
                logger.recovery(
                    "Make a new blank PlaneGrid instead."
                    "Notice: a blank PlaneGrid could not work without specific kwargs"
                )
                self._entity_plane = PlaneGrid(opts=opts_grid)
                
        object.__setattr__(
            self._entity_plane, "_internal_owner_ref", weakref.ref(self)
        )

        if not isinstance(QInterpolator, Interpolator):
            raise TypeError(
                "Interpolator for PlotnPlane must be the class of Nematics3D.classes.Interpolator.Interpolator"
            )
        self._raw_QInterpolator = QInterpolator
        
        self._entity_visual_nb = None
        self._entity_visual_nd = None
        self._entity_visual_defect = None

        self._helper_commit()

    @logging_and_warning_decorator()
    def _helper_commit(self, logger=None):

        logger.debug("Start to identify the directors surrouding defects.")
        plane_grid = self._entity_plane

        logger.detail("Retrieving the full grid in lattice index structure ...")
        grid_all = plane_grid._entity_grid_all
        shape_all = np.shape(grid_all)[:2]
        grid_all_flatten = np.reshape(grid_all, (-1, 3))

        logger.detail("Interpolating ...")
        Q_all = self._raw_QInterpolator.interpolate(grid_all_flatten)
        _, n_all = Q_diagonalize(Q_all)
        n_all = np.reshape(n_all, (*shape_all, 1, 3))
        self._calc_n = (n_all.reshape((-1, 3)))[plane_grid._calc_box_mask]

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
        self._calc_is_near_defect = mask_near_defect[plane_grid._calc_box_mask]

        if len(defect_plane_index)==0:
            self._calc_defect_pos = None
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
            self._calc_defect_pos = defect_pos
            
        if self._entity_visual_nb:
            
            self._entity_visual_nb.act_commit(       
                coords=self._entity_plane()[~self._calc_is_near_defect],
                orient=self._calc_n[~self._calc_is_near_defect]
                )
            
            if np.sum(self._calc_is_near_defect) > 0:
                
                self._entity_visual_nd.act_commit(       
                    coords=self._entity_plane()[self._calc_is_near_defect],
                    orient=self._calc_n[self._calc_is_near_defect]
                    )
                
                self._entity_visual_defect.act_commit( 
                    coords=self._calc_defect_pos
                    )
            else:
                self._entity_visual_nd.act_remove()
                self._entity_visual_defect.act_remove()
                
            
            
            

    @logging_and_warning_decorator()
    def act_visualize(
        self,
        figure: PlotFigure | pv.Plotter | None = None,
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
            opts_nb = OptsRod(color=n_color_immerse, opacity=0.2)
        if opts_nd is None:
            opts_nd = OptsRod(color=n_color_immerse)
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
            elif isinstance(figure, pv.Plotter):
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
            
        visual_nb = PlotRod(
            coords=self._entity_plane()[~self._calc_is_near_defect],
            orient=self._calc_n[~self._calc_is_near_defect],
            name="n bulk",
            category=f"plane {self.name!r}",
            opts=opts_nb,
            figure=figure,
        )
        self._entity_visual_nb = visual_nb

        if np.sum(self._calc_is_near_defect) > 0:
            
            visual_nd = PlotRod(
                coords=self._entity_plane()[self._calc_is_near_defect],
                orient=self._calc_n[self._calc_is_near_defect],
                name="n near defect",
                category=f"plane {self.name!r}",
                opts=opts_nd,
                figure=figure,
            )
            self._entity_visual_nd = visual_nd

            visual_defect = PlotSphere(coords=self._calc_defect_pos, 
                                       name="defects",
                                       category=f"plane {self.name!r}",
                                       opts=opts_defect, 
                                       figure=figure)
            self._entity_visual_defect = visual_defect
            if not is_defect:
                visual_defect.opts.is_visible = False

    @property
    def plane(self):
        return self._entity_plane
    
