import numpy as np
from typing import Optional, Callable, Union
from dataclasses import dataclass

from ..plane_grid import PlaneGrid, OptsPlaneGrid
from ..opts import merge_opts_all
from Nematics3D.datatypes import (
    nField,
    ColorRGB,
    as_ColorRGB,
    Number,
    as_Number
)
from Nematics3D.field import Q_diagonalize, n_color_immerse, n_visualize
from Nematics3D.disclination import defect_detect, defect_vicinity_grid
from Nematics3D.general import select_grid_in_box, split_points
from Nematics3D.logging_decorator import logging_and_warning_decorator
from ..Interpolator import Interpolator


# --- nPlane Options ---
@dataclass(slots=True)
class OptsnPlane:
    colors: Union[Callable[nField, ColorRGB], ColorRGB] = n_color_immerse
    opacity: Union[Callable[nField, np.ndarray], float] = 1
    length: Number = 3.5
    radius: Number = 0.5
    is_n_defect: bool = True
    defect_opacity: Union[Callable[nField, np.ndarray], float] = 1

    __descriptions__ = {
        "colors": "RGB color or callable mapping n-field → RGB",
        "opacity": "opacity value or callable mapping n-field → array",
        "length": "length of directors in plane visualization",
        "radius": "radius of directors in plane visualization",
        "is_n_defect": "flag whether to highlight n around defects",
        "defect_opacity": "opacity value or callable mapping n-field → array for defects",
    }

    _validators = {
        "length": lambda self, v: as_Number(v, name=self.__descriptions__["length"]),
        "radius": lambda self, v: as_Number(v, name=self.__descriptions__["radius"]),
        "is_n_defect": lambda self, v: (
            v
            if isinstance(v, bool)
            else (_ for _ in ()).throw(
                TypeError(
                    f"{self.__descriptions__['is_n_defect']} must be a boolean, got {v}"
                )
            )
        ),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


class PlotnPlane:

    @logging_and_warning_decorator
    def __init__(
        self,
        QInterpolator: Optional[Interpolator] = None,
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
            
        if not isinstance(QInterpolator, Interpolator):
            raise TypeError(
                "Interpolator for PlotnPlane must be the class of Nematics3D.classes.Interpolator.Interpolator"
                )

        merge = merge_opts_all(
            {
             "plane_": opts_grid,
             "n_": opts_nPlane
             },
            kwargs, type(self).__name__
            )

        opts_grid = merge["plane_"]
        opts_nPlane = merge["n_"]

        self._opts_all_nPlane = opts_nPlane
        self._raw_QInterpolator = QInterpolator

        self._helper_make_figure(
            opts_grid=opts_grid,
            opts_nPlane=opts_nPlane,
            logger=logger,
        )

    @logging_and_warning_decorator
    def _helper_make_figure(
        self,
        opts_grid=OptsPlaneGrid(),
        opts_nPlane=OptsnPlane(),
        logger=None,
    ):

        self._opts_all_nplane = opts_nPlane

        self._entities_plane = [
            PlaneGrid(
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
            n_all = np.reshape(n_all, (*shape_all, 1, 3))

            defect_plane_index = defect_detect(n_all, planes=(False, False, True))
            defect_vicinity_index = defect_vicinity_grid(
                defect_plane_index, num_shell=1
            )
            defect_vicinity_index = defect_vicinity_index.reshape((-1, 3))[:, :-1]
                
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
        
        if hasattr(self, "_entities"):
            for item in self._entities:
                item.remove()

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

        self.opts_radius = radius
        self.opts_axis1 = plane_grid.opts_axis1
        self.opts_normal = plane_grid.opts_normal
        self.opts_origin = plane_grid.opts_origin
        self.opts_shape = plane_grid.opts_shape
        self.opts_size = plane_grid.opts_size
        self.opts_corners_limit = corners_limit
        self.opts_is_n_defect = is_n_defect
        self.opts_defect_opacity = defect_opacity
        self.opts_length = length
        self._raw_grid_offset = plane_grid.opts_grid_offset
        self._raw_grid_transform = plane_grid.opts_grid_transform

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
    def opts_length(self):
        return self._entities[0].glyph.glyph_source.glyph_source.height

    @opts_length.setter
    def opts_length(self, value: float):
        self._entities[0].glyph.glyph_source.glyph_source.height = float(value)
        if len(self._entities) > 1:
            self._entities[1].glyph.glyph_source.glyph_source.height = float(value)

    @property
    def opts_radius(self):
        return self._entities[0].glyph.glyph_source.glyph_source.radius

    @opts_radius.setter
    def opts_radius(self, value: float):
        self._entities[0].glyph.glyph_source.glyph_source.radius = float(value)
        if len(self._entities) > 1:
            self._entities[1].glyph.glyph_source.glyph_source.radius = float(value)

    @property
    def opts_opacity_bulk(self):
        rgba = self._entities[0].parent.parent.data.point_data.scalars
        return np.array(rgba)[:, 3] / 255

    @opts_opacity_bulk.setter
    def opts_opacity_bulk(self, data):
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
    def opts_opacity_defect(self):
        if self.opts_is_n_defect:
            if len(self._entities) > 1:
                rgba = self._entities[1].parent.parent.data.point_data.scalars
                return np.array(rgba)[:, 3] / 255
            else:
                raise ValueError("There are no directors around defects")
        else:
            raise ValueError("Directors around defects are not plotted seperately")

    @opts_opacity_defect.setter
    def opts_opacity_defect(self, data):
        if self.opts_is_n_defect:
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
    def opts_colors(self):
        rgba0 = self._entities[0].parent.parent.data.point_data.scalars
        result = []
        result.append(np.array(rgba0)[:, :3] / 255)
        if len(self._entities) > 1:
            rgba1 = self._entities[1].parent.parent.data.point_data.scalars
            result.append(np.array(rgba1)[:, :3] / 255)
        return result

    @opts_colors.setter
    def opts_colors(self, data):
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

        if self.opts_is_n_defect and len(self._entities) > 0:
            set_color(1)

    @logging_and_warning_decorator
    def act_commit(self, logger=None, **changes):

        if not changes:
            return
        
        keys_modify = ["opts_radius", "opts_length", "opts_opacity_bulk", "opts_opacity_defect", "opts_colors"]
        keys_rebuild = ["opts_axis1", "opts_normal", "opts_origin", "opts_shape", "opts_spacing", "opts_size"]
        
        for k, v in changes.items():
            if k in keys_modify:
                setattr(self._opts_all_nplane, k[5:], v)
                setattr(self, k, v)
            elif k in keys_rebuild:
                if k == "opts_spacing":
                    setattr(self._entities_plane[0]._opts_all, "spacing1", v)
                    setattr(self._entities_plane[0]._opts_all, "spacing2", v)
                else:
                    setattr(self._entities_plane[0]._opts_all, k[5:], v)
                setattr(self, k, v)
            else:
                try:
                    raise ValueError(f"{k} is not attribute in PlotnPlane")
                except:
                    logger.recovery(f"Ignore {k}")
                
        for k in keys_rebuild:
            if k in changes:
                self._helper_make_figure(
                    opts_grid=self._entities_plane[0]._opts_all,
                    opts_nPlane=self._opts_all_nplane,
                    logger=logger,
                )
                return
            
