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
    as_Number,
    as_str,
    as_Tensor
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
    opacity: Union[Callable[nField, np.ndarray], float] = 0.2
    length: Number = 3.5
    radius: Number = 0.5
    is_n_defect: bool = True
    opacity_defect: Union[Callable[nField, np.ndarray], float] = 1

    __descriptions__ = {
        "colors": "RGB color or callable mapping n-field → RGB",
        "opacity": "opacity value or callable mapping n-field → array",
        "length": "length of directors in plane visualization",
        "radius": "radius of directors in plane visualization",
        "is_n_defect": "flag whether to highlight n around defects",
        "opacity_defect": "opacity value or callable mapping n-field → array for defects",
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

SLOT = "slot"
PROP = "property"

class PlotnPlane:
    
    __descriptions__ = {
        "name": ("Name identifier of this n-plane object", SLOT),

        # --- internal states ---
        "_raw_QInterpolator": ("Interpolator object for Q-tensor field (class Interpolator)", SLOT),
        "_entities_plane": ("List containing the PlaneGrid entity (geometry of the plane)", SLOT),
        "_entities": ("List of Mayavi visualized objects for n-field directors", SLOT),
        "_calc_n": ("List of director field arrays (from Q-diagonalization)", SLOT),
        "_calc_num_points": ("Total number of lattice points in the plane", SLOT),
        "_calc_colors_func": ("Callable that maps n-field to RGB colors (bulk region)", SLOT),
        "_calc_opacity_func": ("Callable that maps n-field to opacity values (bulk region)", SLOT),
        "_calc_opacity_defect_func": ("Callable that maps n-field to opacity values (defect region)", SLOT),
        "_raw_grid_offset": ("Grid offset of the plane in real-space coordinates", SLOT),
        "_raw_grid_transform": ("Grid transformation matrix of the plane (3×3)", SLOT),
        "_opts_all_nPlane": ("Dataclass OptsnPlane storing all options for n-plane visualization", SLOT),

        # --- visualization options mirrored onto instance ---
        "opts_axis1": ("First in-plane axis (3-vector)", SLOT),
        "opts_normal": ("Normal vector of the plane (3-vector)", SLOT),
        "opts_origin": ("Origin of the plane in real-space coordinates (3-vector)", SLOT),
        "opts_shape": ("Shape of the plane grid (tuple of integers)", SLOT),
        "opts_size": ("Size of the plane in real-space coordinates", SLOT),
        "opts_corners_limit": ("Bounding box corners of the plane (8×3 array)", SLOT),
        "opts_spacing": ("Spacing between neighboring directors", SLOT),
        "opts_is_n_defect": ("Flag whether directors near defects are visualized separately", SLOT),

        # --- properties (not in __slots__) ---
        "opts_length": ("Length of directors in Mayavi visualization", PROP),
        "opts_radius": ("Radius (thickness) of directors in Mayavi visualization", PROP),
        "opts_opacity_bulk": ("Opacity values (array) of directors in bulk region", PROP),
        "opts_opacity_defect": ("Opacity values (array) of directors in defect region", PROP),
        "opts_colors": ("RGB colors (array) of directors in bulk and defect regions", PROP),
    }
    
    __slots__ = tuple(
        k for k, (_, flag) in __descriptions__.items() if flag == SLOT
    )

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
            "spacing": opts_grid.spacing,
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

    @logging_and_warning_decorator()
    def _helper_make_figure(
        self,
        opts_grid=OptsPlaneGrid(),
        opts_nPlane=OptsnPlane(),
        logger=None,
    ):

        self._opts_all_nPlane = opts_nPlane

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
        opacity_defect = opts_nPlane.opacity_defect
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
                * plane_grid.opts_spacing
                + plane_grid._calc_offset_real
            )
            defect_vicinity = select_grid_in_box(defect_vicinity, corners_limit)

            bulk = (
                np.einsum("ai, ib -> ab", bulk_index, axis_both)
                * plane_grid.opts_spacing
                + plane_grid._calc_offset_real
            )
            bulk = select_grid_in_box(bulk, corners_limit)

        else:
            bulk = plane_grid._entities_grid[0]

        grid = plane_grid._entities_grid[0]
        self._calc_num_points = np.shape(grid)[0]

        self._calc_colors_func = self._helper_colors_check(colors)
        self._calc_opacity_func = self._helper_opacity_check(opacity)
        self._calc_opacity_defect_func = self._helper_opacity_check(opacity_defect)
        
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
                defect_vicinity, self._calc_opacity_defect_func, length, radius
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
        self.opts_opacity_defect = opacity_defect
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
            data = as_ColorRGB(data, name="Color for directors on PlotnPlane")
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
            data = as_Number(data, name="Opacity for directors on PlotnPlane", value_range=(0,1), bounded=True)
            opacity = lambda n: np.broadcast_to(data, len(n))
        elif not callable(data):
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
    @logging_and_warning_decorator
    def opts_opacity_defect(self, logger=None):
        if self.opts_is_n_defect:
            if len(self._entities) > 1:
                rgba = self._entities[1].parent.parent.data.point_data.scalars
                return np.array(rgba)[:, 3] / 255
            else:
                logger.info("There are no directors around defects")
        else:
            logger.info("There are no directors around defects")
    
    @opts_opacity_defect.setter
    @logging_and_warning_decorator
    def opts_opacity_defect(self, data, logger=None):
        if self.opts_is_n_defect:
            if len(self._entities) > 1:
                self._calc_opacity_func = self._helper_opacity_check(data)
                rgba = self._entities[1].parent.parent.data.point_data.scalars
                num_points = len(rgba)
                opacity_out = self._calc_opacity_defect_func(self._calc_n[1]) * 255
                rgba = np.array(rgba)
                rgba[:, 3] = opacity_out
                for i in range(num_points):
                    self._entities[1].parent.parent.data.point_data.scalars[i] = rgba[i]
                self._entities[1].parent.parent.data.point_data.scalars.modified()
            else:
                logger.info("There are no directors around defects")
        else:
            logger.info("There are no directors around defects")

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
    def opts_colors(self, data, logger=None):
        self._calc_colors_func = self._helper_colors_check(data)

        def set_color(index):
            rgba = self._entities[index].parent.parent.data.point_data.scalars
            colors_out = self._calc_colors_func(self._calc_n[index]) * 255
            rgba = np.array(rgba)
            rgba[:, :3] = colors_out
            num_points = len(rgba)
            for i in range(num_points):
                self._entities[index].parent.parent.data.point_data.scalars[i] = rgba[i]
            self._entities[index].parent.parent.data.point_data.scalars.modified()

        set_color(0)

        if self.opts_is_n_defect and len(self._entities) > 0:
            set_color(1)
            
    def __setattr__(self, key, value):
        
        if key in ["length", "radius", "colors", "opacity_bulk", "opacity_defect"]:
            key = "opts_" + key
        object.__setattr__(self, key, value)

    @logging_and_warning_decorator
    def act_commit(self, logger=None, **changes):

        if not changes:
            return
        
        keys_modify = ["opts_radius", "opts_length", "opts_opacity_bulk", "opts_opacity_defect", "opts_colors"]
        keys_rebuild = ["opts_axis1", "opts_normal", "opts_origin", "opts_shape", "opts_spacing", "opts_size"]
        
        for k, v in changes.items():
            if k == "name":
                setattr(self, "name", v)
            elif not k.startswith("opts_"):
                k = "opts_" + k
            if k in keys_modify:
                setattr(self._opts_all_nPlane, k[5:], v)
                setattr(self, k, v)
            elif k in keys_rebuild:
                if k == "opts_spacing":
                    setattr(self._entities_plane[0]._opts_all, "spacing", v)
                    setattr(self._entities_plane[0]._opts_all, "spacing_extra", v)
                else:
                    setattr(self._entities_plane[0]._opts_all, k[5:], v)
                setattr(self, k, v)
            else:
                try:
                    raise NameError(f"{k} is not attribute in PlotnPlane")
                except:
                    logger.exception("Error is caught")
                    logger.recovery(f"Ignore {k}")
                
        for k in keys_rebuild:
            if k in changes:
                self._helper_make_figure(
                    opts_grid=self._entities_plane[0]._opts_all,
                    opts_nPlane=self._opts_all_nPlane,
                    logger=logger,
                )
                return
            
    @logging_and_warning_decorator()
    def act_log_parameters(self, is_return: bool = False, logger=None) -> None:
        """
        Log parameters for inspection.

        This is the standard logging interface used in this library, which
        can be redirected to console or to a file depending on the logger
        configuration and the behavior of ``logging_and_warning_decorator``.

        All attributes listed in ``__descriptions__`` are included,
        formatted in a single log entry with a clear separator.
        
        Includes both slots and properties (differentiated by flag).
        """
        lines = []
        lines.append("-------------- PlotnPlane Parameters --------------")
        lines.append(f"[{self.name}] parameters:")

        for attr, (desc, flag) in self.__descriptions__.items():
            value = getattr(self, attr, None)
            if attr in ("opts_axis1", "opts_spacing"):
                lines.append(f"  {attr}: {value!r}  # {desc} (derived final value)")
            else:    
                lines.append(f"  {attr}: {value!r}  # {desc}")

        lines.append("-----------------------------------------------------")
        msg = "\n".join(lines)

        if is_return:
            return msg
        else:
            logger.info(msg)
            
    @logging_and_warning_decorator
    def act_copy(self, is_deep_interpolator: bool = True, logger=None) -> "PlotnPlane":
        
        import copy

        if is_deep_interpolator:
            QInterp_new = copy.deepcopy(self._raw_QInterpolator)
        else:
            QInterp_new = self._raw_QInterpolator

        opts_grid_new = copy.deepcopy(self._entities_plane[0]._opts_all)
        opts_nPlane_new = copy.deepcopy(self._opts_all_nPlane)

        new_obj = self.__class__(
            QInterpolator=QInterp_new,
            opts_grid=opts_grid_new,
            opts_nPlane=opts_nPlane_new,
            logger=logger,
        )

        if hasattr(self, "name"):
            new_obj.name = getattr(self, "name")

        return new_obj
    
    def act_set_rgba(self, rgba, index=0):
        
        num_points = len(self._calc_n[index])
        rgba = as_Tensor((num_points,4), name="The RGBA value to reset PlotnPlane")
        for i in range(num_points):
            self._entities[index].parent.parent.data.point_data.scalars[i] = rgba[i]
        self._entities[index].parent.parent.data.point_data.scalars.modified()
    
    # def act_save(self, dirpath: Optional[str] = None, logger=None) -> None:
        
    #     import os
    #     import json
        
    #     if dirpath is None:
    #          dirpath = os.path.join("save", "PlotnPlane", str(self.name))
             
    #     dirpath = as_str(
    #         dirpath,
    #         name=f"the folder to store PlotnPlane ``{getattr(self, 'name', None)}``"
    #     )
        
    #     logger.debug(f"Start to save plotnPlane ``{self.name}`` into {dirpath}")
    #     logger.debug("Start with parameters")
    #     os.makedirs(dirpath, exist_ok=True)

    #     grid_path = os.path.join(dirpath, "PlaneGrid.json")
    #     self._entities_plane[0].act_save(grid_path, logger=logger)

    #     params = {
    #         "name": getattr(self, "name", None),
    #         "opts_nPlane": {
    #             "length": self.opts_length,
    #             "radius": self.opts_radius,
    #             "is_n_defect": self.opts_is_n_defect,
    #         },
    #     }
        
    #     with open(os.path.join(dirpath, "opts.json"), "w") as f:
    #         json.dump(params, f, indent=2)
        
    #     logger.debug("Now it's time to save data")
        
    #     data_dict = {}
    #     data_dict["n_bulk"] = self._calc_n[0]
    #     data_dict["rgba_bulk"] = np.array(
    #         self._entities[0].parent.parent.data.point_data.scalars
    #     )

    #     if self.opts_is_n_defect and len(self._entities) > 1:
    #         data_dict["n_defect"] = self._calc_n[1]
    #         data_dict["rgba_defect"] = np.array(
    #             self._entities[1].parent.parent.data.point_data.scalars
    #         )
            
    #     np.savez(os.path.join(dirpath, "nplane_data.npz"), **data_dict)
        
            
    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.act_log_parameters(is_return=True)
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}(name={self.name!r}), normal={self.opts_normal}, axis1={self.opts_axis1}, origin={self.opts_origin}"
        return msg
            
