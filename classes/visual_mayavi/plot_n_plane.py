import numpy as np
from typing import Optional, Literal, Callable, List, Union

from .plot_plane_grid import PlotPlaneGrid
from ..opts import OptsPlaneGrid, OptsnPlane ,merge_opts
from Nematics3D.datatypes import Vect, as_Vect, nField, ColorRGB, as_ColorRGB, Tensor, as_Tensor
from Nematics3D.field import Q_diagonalize, n_color_immerse, n_visualize
from Nematics3D.disclination import defect_detect, defect_vicinity_grid
from Nematics3D.general import select_grid_in_box, split_points
from Nematics3D.logging_decorator import logging_and_warning_decorator


class PlotnPlane:

    @logging_and_warning_decorator
    def __init__(
        self,
        opts_grid = OptsPlaneGrid(),
        opts_nPlane = OptsnPlane(),
        logger=None,
        **kwargs
    ):
        
        for name, value in {"normal": opts_grid.normal, "spacing1": opts_grid.spacing1, "spacing2": opts_grid.spacing2, "size": opts_grid.size}.items():
            if value is None:
                raise ValueError(f"Missing required variable {name} to generate plane_grid")
                
        if opts_nPlane.QInterpolator is None:
            raise ValueError("Missing required variable QInterpolator to generate nPlane")

        opts_grid = merge_opts(opts_grid, kwargs, prefix="plane__")
        opts_nPlane = merge_opts(opts_nPlane, kwargs, prefix="n_")
        
        self._internal_opts_nPlane = opts_nPlane
        self._QInterpolator = opts_nPlane.QInterpolator

        self.make_figure(
            opts_grid = opts_grid,
            opts_nPlane = opts_nPlane,
            logger=logger,
        )

    @logging_and_warning_decorator
    def make_figure(
        self,
        opts_grid = OptsPlaneGrid(),
        opts_nPlane = OptsnPlane(),
        logger=None,
    ):
        
        self._items_plane = PlotPlaneGrid(
            opts = opts_grid,
            logger=logger,
        )
        
        is_n_defect = opts_nPlane.is_n_defect
        QInterpolator = opts_nPlane.QInterpolator
        corners_limit = self._items_plane._corners_limit
        colors = opts_nPlane.colors
        opacity = opts_nPlane.opacity
        defect_opacity = opts_nPlane.defect_opacity
        length = opts_nPlane.length
        radius = opts_nPlane.radius

        if is_n_defect:

            axis_both = np.array(
                [self._items_plane._axis1, np.cross(self._items_plane._normal, self._items_plane._axis1)]
            )
            shape_all = np.shape(self._items_plane._grid_all)[:2]
            grid_all_flatten = np.reshape(self._items_plane._grid_all, (-1, 3))

            Q_all = QInterpolator.interpolate(grid_all_flatten)
            _, n_all = Q_diagonalize(Q_all)
            n_all = np.reshape(n_all, (1, *shape_all, 3))
            defect_plane_index = defect_detect(n_all, planes=(True, False, False))

            defect_vicinity_index = defect_vicinity_grid(
                defect_plane_index, num_shell=1
            ).reshape((-1, 3))[:, 1:]
            bulk_index, defect_vicinity_index = split_points(
                self._items_plane._grid_int, defect_vicinity_index
            )

            defect_vicinity = (
                np.einsum("ai, ib -> ab", defect_vicinity_index, axis_both)
                * self._items_plane._spacing1
                + self._items_plane._offset
            )
            defect_vicinity = select_grid_in_box(defect_vicinity, corners_limit)

            bulk = (
                np.einsum("ai, ib -> ab", bulk_index, axis_both) * self._items_plane._spacing1
                + self._items_plane._offset
            )
            bulk = select_grid_in_box(bulk, corners_limit)

        else:
            bulk = self._items_plane._grid

        grid = self._items_plane._grid
        self.num_points = np.shape(grid)[0]

        self.colors_func = self.colors_check(colors)
        self.opacity_func = self.opacity_check(opacity)
        self.defect_opacity_func = self.opacity_check(defect_opacity)

        if hasattr(self, "items"):
            self.items[0].remove()
            self.items[1].remove()

        self.items = []
        self.n = []
        output = self.n_visualize_each(bulk, self.opacity_func, length, radius)
        self.items.append(output[0])
        self.n.append(output[1])

        if is_n_defect and len(defect_vicinity) > 0:
            output = self.n_visualize_each(
                defect_vicinity, self.defect_opacity_func, length, radius
            )
            self.items.append(output[0])
            self.n.append(output[1])

        self.radius = radius
        self.axis1 = self._items_plane._axis1
        self.normal = self._items_plane._normal
        self.origin = self._items_plane._origin
        self.shape = self._items_plane._shape
        self.size = self._items_plane._size
        self.corners_limit = corners_limit
        self.is_n_defect = is_n_defect
        self.defect_opacity = defect_opacity
        self.grid_offset = self._items_plane._grid_offset
        self.grid_transform = self._items_plane._grid_transform

    def n_visualize_each(self, data, opacity_func, length, radius):

        Q = self._QInterpolator.interpolate(data, is_index=False)
        n = Q_diagonalize(Q)[1]

        colors_out = self.colors_func(n)
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
    def colors_check(self, data, logger=None):
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
    def opacity_check(self, data, logger=None):
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
        return self.items[0].glyph.glyph_source.glyph_source.height

    @length.setter
    def length(self, value: float):
        self.items[0].glyph.glyph_source.glyph_source.height = float(value)
        if len(self.items) > 1:
            self.items[1].glyph.glyph_source.glyph_source.height = float(value)

    @property
    def radius(self):
        return self.items[0].glyph.glyph_source.glyph_source.radius

    @radius.setter
    def radius(self, value: float):
        self.items[0].glyph.glyph_source.glyph_source.radius = float(value)
        if len(self.items) > 1:
            self.items[1].glyph.glyph_source.glyph_source.radius = float(value)

    @property
    def opacity_bulk(self):
        rgba = self.items[0].parent.parent.data.point_data.scalars
        return np.array(rgba)[:, 3] / 255

    @opacity_bulk.setter
    def opacity_bulk(self, data):
        self.opacity_func = self.opacity_check(data)
        rgba = self.items[0].parent.parent.data.point_data.scalars
        num_points = len(rgba)
        opacity_out = self.opacity_func(self.n[0]) * 255
        rgba = np.array(rgba)
        rgba[:, 3] = opacity_out
        for i in range(num_points):
            self.items[0].parent.parent.data.point_data.scalars[i] = rgba[i]
        self.items[0].parent.parent.data.point_data.scalars.modified()

    @property
    def opacity_defect(self):
        if self.is_n_defect:
            if len(self.items) > 1:
                rgba = self.items[1].parent.parent.data.point_data.scalars
                return np.array(rgba)[:, 3] / 255
            else:
                raise ValueError("There are no directors around defects")
        else:
            raise ValueError("Directors around defects are not plotted seperately")

    @opacity_defect.setter
    def opacity_defect(self, data):
        if self.is_n_defect:
            if len(self.items) > 1:
                self.opacity_func = self.opacity_check(data)
                rgba = self.items[1].parent.parent.data.point_data.scalars
                num_points = len(rgba)
                opacity_out = self.defect_opacity_func(self.n[1]) * 255
                rgba = np.array(rgba)
                rgba[:, 3] = opacity_out
                for i in range(num_points):
                    self.items[1].parent.parent.data.point_data.scalars[i] = rgba[i]
                self.items[1].parent.parent.data.point_data.scalars.modified()
            else:
                raise ValueError("There are no directors around defects")
        else:
            raise ValueError("There are no isolated directors around defects")

    @property
    def colors(self):
        rgba0 = self.items[0].parent.parent.data.point_data.scalars
        result = []
        result.append(np.array(rgba0)[:, :3] / 255)
        if len(self.items) > 1:
            rgba1 = self.items[1].parent.parent.data.point_data.scalars
            result.append(np.array(rgba1)[:, :3] / 255)
        return result

    @colors.setter
    def colors(self, data):
        self.colors_func = self.colors_check(data)

        def set_color(index):
            rgba = self.items[index].parent.parent.data.point_data.scalars
            num_points = len(rgba)
            colors_out = self.colors_func(self.n[index]) * 255
            rgba = np.array(rgba)
            rgba[:, :3] = colors_out
            for i in range(num_points):
                self.items[index].parent.parent.data.point_data.scalars[i] = rgba[i]
            self.items[index].parent.parent.data.point_data.scalars.modified()

        set_color(0)

        if self.is_n_defect and len(self.items) > 0:
            set_color(1)

    @logging_and_warning_decorator
    def update(self, logger=None, **changes):

        if not changes:
            return

        for k, v in changes.items():
            setattr(self, k, v)

        keys_rebuild = ["axis1", "normal", "origin", "shape", "spacing", "size"]

        for k in keys_rebuild:
            if k in changes:
                self.make_figure(
                    self.normal,
                    self.spacing,
                    self.size,
                    self._QInterpolator,
                    self.shape,
                    self.origin,
                    self.axis1,
                    self.corners_limit,
                    self.colors_func,
                    self.opacity_func,
                    self.length,
                    self.radius,
                    self.is_n_defect,
                    self.defect_opacity,
                    self.grid_offset,
                    self.grid_transform,
                    logger=logger,
                )
                return
