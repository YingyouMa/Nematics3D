import numpy as np
from typing import Optional, Literal, Callable, List, Union
from scipy.interpolate import RegularGridInterpolator

from .plot_plane_grid import PlotPlaneGrid
from ..opts import OptsPlaneGrid, OptsnPlane, merge_opts
from Nematics3D.datatypes import (
    Vect,
    as_Vect,
    nField,
    ColorRGB,
    as_ColorRGB,
    Tensor,
    as_Tensor,
)
from Nematics3D.field import Q_diagonalize, n_color_immerse, n_visualize
from Nematics3D.disclination import defect_detect, defect_vicinity_grid
from Nematics3D.general import select_grid_in_box, split_points
from Nematics3D.logging_decorator import logging_and_warning_decorator


class PlotnPlane:

    @logging_and_warning_decorator
    def __init__(
        self,
        QInterpolator: Optional[RegularGridInterpolator] = None,
        opts_grid=OptsPlaneGrid(),
        opts_nPlane=OptsnPlane(),
        logger=None,
        **kwargs,
    ):

        for name, value in {
            "normal": opts_grid.normal,
            "spacing1": opts_grid.spacing1,
            "spacing2": opts_grid.spacing2,
            "size": opts_grid.size,
        }.items():
            if value is None:
                raise ValueError(
                    f"Missing required variable {name} to generate plane_grid"
                )

        if QInterpolator is None:
            raise ValueError(
                "Missing required variable QInterpolator to generate nPlane"
            )

        opts_grid = merge_opts(opts_grid, kwargs, prefix="plane__")
        opts_nPlane = merge_opts(opts_nPlane, kwargs, prefix="n_")

        self._opts_all_nPlane = opts_nPlane
        self._raw_QInterpolator = QInterpolator

        self.make_figure(
            opts_grid=opts_grid,
            opts_nPlane=opts_nPlane,
            logger=logger,
        )

    @logging_and_warning_decorator
    def make_figure(
        self,
        opts_grid=OptsPlaneGrid(),
        opts_nPlane=OptsnPlane(),
        logger=None,
    ):

        self._opts_all = opts_nPlane

        self._entities_plane = [
            PlotPlaneGrid(
                opts=opts_grid,
                logger=logger,
            )
        ]

        plane_grid = self._entities_plane[0]

        QInterpolator = self._raw_QInterpolator
        is_n_defect = opts_nPlane.is_n_defect
        corners_limit = plane_grid.opts_corners_limit
        colors = opts_nPlane.colors
        opacity = opts_nPlane.opacity
        defect_opacity = opts_nPlane.defect_opacity
        length = opts_nPlane.length
        radius = opts_nPlane.radius

        if is_n_defect:

            axis_both = np.array(
                [
                    plane_grid.opts_axis1,
                    np.cross(plane_grid.opts_normal, plane_grid.opts_axis1),
                ]
            )

            grid_all = self._entities_plane[0]._entities_grid_all[0]
            shape_all = np.shape(grid_all)[:2]
            grid_all_flatten = np.reshape(grid_all, (-1, 3))

            Q_all = QInterpolator.interpolate(grid_all_flatten)
            _, n_all = Q_diagonalize(Q_all)
            n_all = np.reshape(n_all, (1, *shape_all, 3))
            defect_plane_index = defect_detect(n_all, planes=(True, False, False))

            defect_vicinity_index = defect_vicinity_grid(
                defect_plane_index, num_shell=1
            ).reshape((-1, 3))[:, 1:]
            bulk_index, defect_vicinity_index = split_points(
                self._entities_plane[0]._entities_grid_int[0], defect_vicinity_index
            )

            defect_vicinity = (
                np.einsum("ai, ib -> ab", defect_vicinity_index, axis_both)
                * plane_grid.opts_spacing1
                + plane_grid._calc_offset_real
            )
            defect_vicinity = select_grid_in_box(defect_vicinity, corners_limit)

            bulk = (
                np.einsum("ai, ib -> ab", bulk_index, axis_both)
                * plane_grid.opts_spacing1
                + plane_grid._calc_offset_real
            )
            bulk = select_grid_in_box(bulk, corners_limit)

        else:
            bulk = plane_grid._entities_grid[0]

        grid = plane_grid._entities_grid[0]
        self._calc_num_points = np.shape(grid)[0]

        self._calc_colors_func = self._helper_colors_check(colors)
        self._calc_opacity_func = self._helper_opacity_check(opacity)
        self._calc_defect_opacity_func = self._helper_opacity_check(defect_opacity)

        if hasattr(self, "items"):
            self._entities[0].remove()
            self._entities[1].remove()

        self._entities = []
        self._calc_n = []
        output = self._helper_n_visualize_each(
            bulk, self._calc_opacity_func, length, radius
        )
        self._entities.append(output[0])
        self._calc_n.append(output[1])

        if is_n_defect and len(defect_vicinity) > 0:
            output = self._helper_n_visualize_each(
                defect_vicinity, self._calc_defect_opacity_func, length, radius
            )
            self._entities.append(output[0])
            self._calc_n.append(output[1])

        self.radius = radius
        self.axis1 = plane_grid.opts_axis1
        self.normal = plane_grid.opts_normal
        self.origin = plane_grid.opts_origin
        self.shape = plane_grid.opts_shape
        self.size = plane_grid.opts_size
        self.corners_limit = corners_limit
        self.is_n_defect = is_n_defect
        self.defect_opacity = defect_opacity
        self.grid_offset = plane_grid.opts_grid_offset
        self.grid_transform = plane_grid.opts_grid_transform

    def _helper_n_visualize_each(self, data, opacity_func, length, radius):

        Q = self._raw_QInterpolator.interpolate(data, is_index=False)
        n = Q_diagonalize(Q)[1]

        colors_out = self._calc_colors_func(n)
        opacity_out = opacity_func(n)

        result = n_visualize(
            data,
            n,
            colors=colors_out,
            opacity=opacity_out,
            length=length,
            radius=radius,
        )

        return result, n

    @logging_and_warning_decorator
    def _helper_colors_check(self, data, logger=None):
        if isinstance(data, (tuple, list, np.ndarray)):
            data = as_ColorRGB(data)
            colors = lambda n: np.broadcast_to(data, (len(n), 3))
        elif not callable(data):
            msg = "Colors must be either callable function or a tuple of three elements.\n"
            msg = "Use default colormap in the following."
            logger.warning(msg)
            colors = n_color_immerse
        else:
            colors = data
        return colors

    @logging_and_warning_decorator
    def _helper_opacity_check(self, data, logger=None):
        if isinstance(data, (int, float)):
            opacity = lambda n: np.broadcast_to(data, len(n))
        elif not callable(input):
            msg = "Opacity must be either callable function or a float.\n"
            msg = "Use 1 in the following."
            logger.warning(msg)
            opacity = lambda n: np.broadcast_to(1, len(n))
        else:
            opacity = data
        return opacity

    @property
    def length(self):
        return self._entities[0].glyph.glyph_source.glyph_source.height

    @length.setter
    def length(self, value: float):
        self._entities[0].glyph.glyph_source.glyph_source.height = float(value)
        if len(self._entities) > 1:
            self._entities[1].glyph.glyph_source.glyph_source.height = float(value)

    @property
    def radius(self):
        return self._entities[0].glyph.glyph_source.glyph_source.radius

    @radius.setter
    def radius(self, value: float):
        self._entities[0].glyph.glyph_source.glyph_source.radius = float(value)
        if len(self._entities) > 1:
            self._entities[1].glyph.glyph_source.glyph_source.radius = float(value)

    @property
    def opacity_bulk(self):
        rgba = self._entities[0].parent.parent.data.point_data.scalars
        return np.array(rgba)[:, 3] / 255

    @opacity_bulk.setter
    def opacity_bulk(self, data):
        self._calc_opacity_func = self._helper_opacity_check(data)
        rgba = self._entities[0].parent.parent.data.point_data.scalars
        num_points = len(rgba)
        opacity_out = self._calc_opacity_func(self._calc_n[0]) * 255
        rgba = np.array(rgba)
        rgba[:, 3] = opacity_out
        for i in range(num_points):
            self._entities[0].parent.parent.data.point_data.scalars[i] = rgba[i]
        self._entities[0].parent.parent.data.point_data.scalars.modified()

    @property
    def opacity_defect(self):
        if self.is_n_defect:
            if len(self._entities) > 1:
                rgba = self._entities[1].parent.parent.data.point_data.scalars
                return np.array(rgba)[:, 3] / 255
            else:
                raise ValueError("There are no directors around defects")
        else:
            raise ValueError("Directors around defects are not plotted seperately")

    @opacity_defect.setter
    def opacity_defect(self, data):
        if self.is_n_defect:
            if len(self._entities) > 1:
                self._calc_opacity_func = self._helper_opacity_check(data)
                rgba = self._entities[1].parent.parent.data.point_data.scalars
                num_points = len(rgba)
                opacity_out = self._calc_defect_opacity_func(self._calc_n[1]) * 255
                rgba = np.array(rgba)
                rgba[:, 3] = opacity_out
                for i in range(num_points):
                    self._entities[1].parent.parent.data.point_data.scalars[i] = rgba[i]
                self._entities[1].parent.parent.data.point_data.scalars.modified()
            else:
                raise ValueError("There are no directors around defects")
        else:
            raise ValueError("There are no isolated directors around defects")

    @property
    def colors(self):
        rgba0 = self._entities[0].parent.parent.data.point_data.scalars
        result = []
        result.append(np.array(rgba0)[:, :3] / 255)
        if len(self._entities) > 1:
            rgba1 = self._entities[1].parent.parent.data.point_data.scalars
            result.append(np.array(rgba1)[:, :3] / 255)
        return result

    @colors.setter
    def colors(self, data):
        self._calc_colors_func = self._helper_colors_check(data)

        def set_color(index):
            rgba = self._entities[index].parent.parent.data.point_data.scalars
            num_points = len(rgba)
            colors_out = self._calc_colors_func(self._calc_n[index]) * 255
            rgba = np.array(rgba)
            rgba[:, :3] = colors_out
            for i in range(num_points):
                self._entities[index].parent.parent.data.point_data.scalars[i] = rgba[i]
            self._entities[index].parent.parent.data.point_data.scalars.modified()

        set_color(0)

        if self.is_n_defect and len(self._entities) > 0:
            set_color(1)

    @logging_and_warning_decorator
    def act_commit(self, logger=None, **changes):

        if not changes:
            return

        for k, v in changes.items():
            setattr(self._opts_all, k, v)

        keys_rebuild = ["axis1", "normal", "origin", "shape", "spacing", "size"]

        for k in keys_rebuild:
            if k in changes:
                self.make_figure(
                    opts_grid=self._entities_plane[0]._opts_all,
                    opts_plane=self._opts_all,
                    logger=logger,
                )
                return
