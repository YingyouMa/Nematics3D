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
from .plot_tube_each import OptsTubeEach, PlotTubeEach


# --- Tube Options ---
@dataclass(slots=True)
class OptsTube:
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
    """
    Visualize 3D polylines as tubular surfaces using Mayavi's ``mlab.plot3d``.

    Workflow
    --------
    1. For each subline of input coordinates, call ``mlab.plot3d`` to create a tube
       mesh with either uniform color or per-point scalars.
    2. Apply visual options such as radius, sides, opacity, and specular highlights
       from the associated :class:`OptsTube` dataclass.
    3. Expose these options back to the user as ``opts_*`` attributes, which are
       automatically synchronized with both the internal state and the underlying
       Mayavi objects.

    Parameters
    ----------
    coords_all : list of (N, 3) arrays
        List of 3D coordinate arrays, one for each subline to be drawn as a tube.

    scalars_all : list of arrays or None, optional
        Optional scalar values to color each subline. If provided, diffuse RGB
        color (``opts_color``) will be ignored.

    opts : OptsTube, optional
        Options controlling tube rendering, such as radius, color, opacity,
        and specular highlights. See :attr:`OptsTube.__descriptions__` for details.

    logger : logging.Logger, optional
        Logger instance for warnings and information messages.
        If None, falls back to the global logging configuration.

    Attributes
    ----------
    See :attr:`PlotTube.__descriptions__` for a full list and explanation of
    attributes (including both internal state such as ``_entities`` and
    mirrored options such as ``opts_radius``).

    Methods
    -------
    act_hide()
        Hide all tubes (set ``opts_is_visible=False``).

    act_show()
        Show all tubes (set ``opts_is_visible=True``).

    act_remove()
        Remove all Mayavi objects associated with this tube.

    act_log_parameters(is_return=False, logger=None)
        Log or return a formatted summary of parameters and results.

    Python Special Methods
    ----------------------
    - ``str(tube)`` → formatted summary of parameters (e.g., ``print(tube)``).
    - ``len(tube)`` → # sublines.
    - ``iter(tube)`` → iterate over sublines
    - ``line[tube]`` → get the i-th subline
    - ``repr(line)`` → short identifier for debugging. (e.g., just type ``line`` in an interactive shell)

    Notes
    -----
    - If scalar values are provided for coloring, ``opts_color`` will be ignored.
    - Internal attributes (prefixed with ``_``) are protected and cannot be
      modified directly.
    - Option attributes (prefixed with ``opts_``) are implemented as properties
      and automatically update the underlying Mayavi objects when changed.
    - For convenience, during user assignment the ``opts_`` prefix is optional:
      e.g. ``tube.color = (1,0,0)`` is automatically redirected to
      ``tube.opts_color = (1,0,0)``.
    """
    
    __descriptions__ = {
        "name": "Name identifier of this tube object",
    
        # --- internal states ---
        "_entities": "List of Mayavi tube objects (mlab.plot3d items)",
        "_raw_coords_all": "Raw input coordinates for all sublines (list of arrays, each shape: N×3)",
        "_raw_scalars_all": "Optional scalar values for coloring each subline (list of arrays or None)",
    
        # --- mirrored options ---
        "opts_color": "Diffuse RGB color of tube surface (ignored if scalars are provided)",
        "opts_opacity": "Opacity of tube surface",
        "opts_radius": "Tube radius (applied in mlab.plot3d)",
        "opts_sides": "Number of polygonal sides used to approximate tube cross-section",
        "opts_specular": "Strength of specular highlight on tube surface",
        "opts_specular_color": "RGB color of the specular highlight",
        "opts_specular_power": "Shininess exponent controlling specular highlight size",
        "opts_is_visible": "Boolean flag indicating whether tubes are visible in the scene",
        "_opts_all": "The dataclass OptsTube storing all option values",
    }
    
    __slots__ = tuple(__descriptions__.keys())

    @logging_and_warning_decorator()
    def __init__(
        self,
        coords_all: List,
        scalars_all: Optional[List] = None,
        opts=OptsTube(),
        logger=None,
    ) -> None:

        # We deliberately use object.__setattr__ here to bypass the custom __setattr__.
        # This ensures that internal state variables (e.g., _initializing, _entities,
        # _state_is_smoothed, etc.) can be assigned without triggering the validation
        # or auto-commit logic of __setattr__. (same below)
        object.__setattr__(self, "_entities", [])
        object.__setattr__(self, "_raw_coords_all", coords_all)
        object.__setattr__(self, "_raw_scalars_all", scalars_all)
        object.__setattr__(self, "_opts_all", opts)
        self.name = opts.name

        if opts.color is None:
            logger.warning("The color input of tube is None. Changed it into (1,1,1).")
            opts.color = (1, 1, 1)

        num_sublines = len(self._raw_coords_all)
        if self._raw_scalars_all is not None:
            logger.debug(">>> The scalars of tube is input")
            logger.debug(">>> The color of tube will be ignored")
        else:
            object.__setattr__(self, "_raw_scalars_all", [None for i in range(num_sublines)])
            
        logger.debug("Plotting ...")
        for coords, scalars, index in zip(self._raw_coords_all, self._raw_scalars_all, np.arange(num_sublines)):

            opts_each = OptsTubeEach(
                radius=opts.radius,
                sides=opts.sides,
                opacity=opts.opacity,
                color=opts.color,
                specular = opts.specular,
                specular_color = opts.specular_color,
                specular_power = opts.specular_power,
                name = self.name + f"_{index}"
                )

            subline = PlotTubeEach(coords, scalars, opts=opts_each)
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
        return len(self._raw_coords_all)
    
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        msg = f"{cls_name}(name={self.name!r}), with color={self.opts_color!r} and radius={self.opts_radius!r}"
        return msg
    
    def __iter__(self):
        return iter(self._entities)
    
    def __getitem__(self, idx):
        return self._entities[idx]

      


    


