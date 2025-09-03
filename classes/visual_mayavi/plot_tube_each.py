from mayavi import mlab
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, asdict

from Nematics3D.logging_decorator import logging_and_warning_decorator
from ..opts import auto_opts_tubes
from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    Number,
    as_Number,
    as_str,
    as_bool
)


# --- Tube Options ---
@dataclass(slots=True)
class OptsTubeEach:
    radius: Number = 0.5
    opacity: Number = 1
    color: ColorRGB = (1.0, 1.0, 1.0)
    sides: Number = 6
    specular: Number = 1
    specular_color: ColorRGB = (1.0, 1.0, 1.0)
    specular_power: Number = 11
    name: str = "None"
    is_visible: bool = True

    __descriptions__ = {
        "radius": "radius of tube",
        "opacity": "opacity of tube",
        "color": "RGB color of tube surface",
        "sides": "number of sides of tube",
        "specular": "strength of specular highlight",
        "specular_color": "RGB color of specular highlight",
        "specular_power": "shininess of specular highlight",
        "name": "name identifier of tube",
        "is_visible": "whether represent this line"
    }

    _validators = {
        "radius": lambda self, v: as_Number(v, name=self.__descriptions__["radius"]),
        "opacity": lambda self, v: as_Number(v, name=self.__descriptions__["opacity"]),
        "color": lambda self, v: (
            None if v is None else as_ColorRGB(v, name=self.__descriptions__["color"])
        ),
        "sides": lambda self, v: as_Number(v, name=self.__descriptions__["sides"]),
        "specular": lambda self, v: as_Number(
            v, name=self.__descriptions__["specular"]
        ),
        "specular_color": lambda self, v: as_ColorRGB(
            v, name=self.__descriptions__["specular_color"]
        ),
        "specular_power": lambda self, v: as_Number(
            v, name=self.__descriptions__["specular_power"]
        ),
        "name": lambda self, v: as_str(v, name=self.__descriptions__["name"]),
        "is_visible": lambda self, v: as_bool(v, name=self.__descriptions__["is_visible"])
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
        "opts_specular": "actor.property.specular",
        "opts_specular_color": "actor.property.specular_color",
        "opts_specular_power": "actor.property.specular_power",
        "opts_is_visible": "actor.visible",
    }
)
class PlotTube:

    
    __descriptions__ = {
        "name": "Name identifier of this tube object",
    
        # --- internal states ---
        "_entities": "List of Mayavi tube objects (mlab.plot3d items)",
        "_raw_coords": "Raw input coordinates (shape: N×3)",
        "_raw_scalars": "Optional scalar values for coloring (shape N, or None)",
    
        # --- mirrored options ---
        "opts_color": "Diffuse RGB color of tube surface (ignored if scalars are provided)",
        "opts_opacity": "Opacity of tube surface",
        "opts_radius": "Tube radius (applied in mlab.plot3d)",
        "opts_sides": "Number of polygonal sides used to approximate tube cross-section",
        "opts_specular": "Strength of specular highlight on tube surface",
        "opts_specular_color": "RGB color of the specular highlight",
        "opts_specular_power": "Shininess exponent controlling specular highlight size",
        "opts_is_visible": "Boolean flag indicating whether tubes are visible in the scene",
        "_opts_all": "The dataclass OptsTubeEach storing all option values",
    }
    
    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator()
    def __init__(
        self,
        coords: List,
        scalars: Optional[List, np.ndarray] = None,
        opts=OptsTubeEach(),
        logger=None,
    ) -> None:

        # We deliberately use object.__setattr__ here to bypass the custom __setattr__.
        # This ensures that internal state variables (e.g., _initializing, _entities,
        # _state_is_smoothed, etc.) can be assigned without triggering the validation
        # or auto-commit logic of __setattr__. (same below)
        object.__setattr__(self, "_entities", [])
        object.__setattr__(self, "_raw_coords", coords)
        object.__setattr__(self, "_raw_scalars", scalars)
        object.__setattr__(self, "_opts_all", opts)
        self.name = opts.name
            
        logger.debug("Plotting ...")

        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        if scalars is not None and len(scalars) != len(x):
            msg = f">>> Line {self.nema} has {len(x)} points, while scalars has {len(scalars)} values. \n"
            msg += ">>> Ignore scalars in the following"
            logger.warning(msg)
            scalars = None
            
        item = _helper_make_figure(x, y, z, scalars, 
                                   opts.color, opts.radius, opts.sides, opts.opacity,
                                   opts.specular, opts.specular_color, opts.specular_power)
        
        self._entities.append(item)
            
            
    def _helper_make_figure(x, y, z, scalars, color, radius, sides, opacity, specular, specular_color, specular_power):
    
        if scalars is not None:

            item = mlab.plot3d(
                x,
                y,
                z,
                scalars,
                tube_radius=radius,
                tube_sides=sides,
                opacity=opacity,
            )
        else:
            item = mlab.plot3d(
                x,
                y,
                z,
                color=color,
                tube_radius=radius,
                tube_sides=sides,
                opacity=opacity,
            )

        prop = item.actor.property
        prop.specular = specular
        prop.specular_color = specular_color
        prop.specular_power = specular_power
            
        return item

    def act_hide(self):
        self.opts_is_visible = False

    def act_show(self):
        self.opts_is_visible = True

    def act_remove(self):
        self._entities[0].remove()
            
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
        lines.append("-------------- PlotTube Parameters --------------")
        
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

    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} object>"
        return header + "\n" + self.act_log_parameters(is_return=True)
    
    def __len__(self) -> int:
        return len(self._raw_coords)
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}(name={self.name!r}), with color={self.opts_color!r} and radius={self.opts_radius!r}"
        return msg
    
    def __iter__(self):
        return iter(self._raw_coords)
    
    def __getitem__(self, idx):
        return self._raw_coords[idx]

      


    


