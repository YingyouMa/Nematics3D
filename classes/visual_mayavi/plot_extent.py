from mayavi import mlab
import numpy as np
from typing import Optional
from dataclasses import dataclass

from Nematics3D.logging_decorator import logging_and_warning_decorator
from ..opts import auto_opts_tubes
from Nematics3D.datatypes import Number, as_Number, ColorRGB, as_ColorRGB, Tensor, as_Tensor, as_str, as_bool
from .plot_tube_each import OptsTubeEach, PlotTubeEach


# --- Extent Options ---
@dataclass(slots=True)
class OptsExtent:
    corners: Optional[Tensor((8,3))] = None
    radius: Number = 1.0
    sides: Number = 6
    opacity: Number = 1.0
    color: ColorRGB = (0, 0, 0)
    name: str = "None"
    is_visible: bool = True

    __descriptions__ = {
        "corners": "bounding box corners (8×3 array)",
        "radius": "radius of extent tubes",
        "sides": "sides number of extent tubes",
        "opacity": "opacity of extent tubes",
        "color": "RGB color of extent tubes",
        "name": "name of extent",
        "is_visible": "whether represent this line",
    }

    _validators = {
        "corners": lambda self, v: (
            None
            if v is None
            else as_Tensor(v, (8, 3), name=self.__descriptions__["corners"])
        ),
        "radius": lambda self, v: as_Number(v, name=self.__descriptions__["radius"]),
        "sides": lambda self, v: as_Number(v, name=self.__descriptions__["sides"]),
        "opacity": lambda self, v: as_Number(v, name=self.__descriptions__["opacity"]),
        "color": lambda self, v: as_ColorRGB(v, name=self.__descriptions__["color"]),
        "name": lambda self, v: as_str(v, name=self.__descriptions__["name"]),
        "is_visible": lambda self, v: as_bool(v, name=self.__descriptions__["is_visible"]),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


@auto_opts_tubes(
    {
        "opts_color": "actor.property.diffuse_color",
        "opts_opacity": "actor.property.opacity",
        "opts_radius": "parent.parent.filter.radius",
        "opts_sides": "parent.parent.filter.number_of_sides",
        "opts_is_visible": "actor.visible",
    }
)
class PlotExtent:

    __descriptions__ = {
    "name": "Name identifier of this extent object",

    # --- internal states ---
    "_entities": "List of Mayavi tube objects (mlab.plot3d items), as 12 edges of the box",
    "_raw_corners": "bounding box corners (8×3 array)",

    # --- mirrored options ---
    "opts_color": "Diffuse RGB color of tube surface (ignored if scalars are provided)",
    "opts_opacity": "Opacity of tube surface",
    "opts_radius": "Tube radius (applied in mlab.plot3d)",
    "opts_sides": "Number of polygonal sides used to approximate tube cross-section",
    "opts_is_visible": "Boolean flag indicating whether tubes are visible in the scene",
    "_opts_all": "The dataclass OptsTube storing all option values",
}
    
    __slots__ = tuple(__descriptions__.keys())

    def __init__(self, opts=OptsExtent()):

        if opts.corners is None:
            raise ValueError(
                "The array 'corners' as bounding box corners, which stores the positions of the 8 points, are not inputted (the value is None)"
            )
        object.__setattr__(self, "_raw_corners", opts.corners)
        object.__setattr__(self, "_opts_all", opts)
        object.__setattr__(self, "_entities", [])
        self.name = opts.name
        
        corners = self._raw_corners
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 6),
            (4, 7),
            (3, 5),
            (3, 6),
            (5, 7),
            (6, 7),
        ]
        
        index = 0
        for i, j in edges:
            p1, p2 = corners[i], corners[j]
            coords = np.array([p1, (p1 + p2) / 2, p2])
            opts_each = OptsTubeEach(radius=opts.radius, opacity=opts.opacity, color=opts.color, name=self.name+f"_{index}")
            subline = PlotTubeEach(coords, opts=opts_each)
            self._entities.append(subline)


    def act_hide(self):
        self.opts_is_visible = False

    def act_show(self):
        self.opts_is_visible = True

    def act_remove(self):
        for item in self._entities:
            item.remove()
            
    def __setattr__(self, key, value):
        if key.startswith("_"):
            raise AttributeError(f"Internal attribute {key} cannot be modified.")
            
        if key == "name":
            object.__setattr__(self, key, value)
            return
        
        if not key.startswith("opts"):
            key_new = "opts_" + key
        else:
            key_new = key
        if key_new not in self.__slots__:
            raise NameError(f"Either {key} or {key_new} is not a valid attribute of {type(self).__name__}")
            
        descriptor = type(self).__dict__.get(key_new, None)
        descriptor.__set__(self, value)
        
    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.act_log_parameters(is_return=True)
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}(name={self.name!r}). with color={self.opts_color!r} and box corners: \n"
        msg += f"{self._raw_corners}"
        return msg
        
    def __iter__(self):
        return iter(self._entities)
    
    def __getitem__(self, idx):
        return self._entities[idx]

    @logging_and_warning_decorator()
    def act_log_parameters(self, is_return: bool = False, logger=None) -> None:
        """
        Log parameters for inspection.

        This is the standard logging interface used in this library, which
        can be redirected to console or to a file depending on the logger
        configuration and the behavior of ``logging_and_warning_decorator``.

        All attributes listed in ``__descriptions__`` are included,
        formatted in a single log entry with a clear separator.
        """
        lines = []
        lines.append("-------------- PlotExtent Parameters --------------")
        
        lines.append(f"[{self.name}] plotting parameters:")
        for attr in self.__slots__:
            desc = self.__descriptions__.get(attr, "(no description)")
            value = getattr(self, attr, None)
            lines.append(f"  {attr}: {value!r}  # {desc}")
        lines.append("-----------------------------------------------------")

        msg = "\n".join(lines)

        if is_return:
            return msg
        else:
            logger.info(msg)