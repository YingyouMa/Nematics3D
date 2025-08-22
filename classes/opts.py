from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Tuple, Optional, Union, Literal, Callable
import numpy as np

from Nematics3D.datatypes import (
    ColorRGB,
    as_ColorRGB,
    Vect,
    as_Vect,
    Tensor,
    as_Tensor,
    Number,
    as_Number,
    nField,
    as_str,
)
from Nematics3D.field import n_color_immerse


@dataclass()
class OptsSmoothen:
    window_ratio: Number = 3
    window_length: Optional[Number] = 41
    order: Number = 3
    N_out_ratio: Number = 3.0
    mode: Literal["interp", "wrap"] = "interp"
    min_line_length: int = 50
    name: str = "None"

    __descriptions__ = {
        "window_ratio": "window ratio for smoothening",
        "window_length": "explicit window length for smoothening",
        "order": "smoothing polynomial order",
        "N_out_ratio": "ratio between output and input #points in smoothening",
        "mode": "smoothing mode (interp or wrap)",
        "min_line_length": "minimum line length to be smoothened",
        "name": "name identifier of smoothen options",
    }

    _validators = {
        "window_ratio": lambda self, v: as_Number(
            v, name=self.__descriptions__["window_ratio"]
        ),
        "window_length": lambda self, v: (
            None
            if v is None
            else as_Number(v, name=self.__descriptions__["window_length"])
        ),
        "order": lambda self, v: as_Number(v, name=self.__descriptions__["order"]),
        "N_out_ratio": lambda self, v: as_Number(
            v, name=self.__descriptions__["N_out_ratio"]
        ),
        "mode": lambda self, v: (
            v
            if v in ("interp", "wrap")
            else (_ for _ in ()).throw(
                ValueError(
                    f"{self.__descriptions__['mode']} must be 'interp' or 'wrap', got {v!r}"
                )
            )
        ),
        "min_line_length": lambda self, v: (
            v
            if isinstance(v, int)
            else (_ for _ in ()).throw(
                TypeError(
                    f"{self.__descriptions__['min_line_length']} must be int, got {type(v)}"
                )
            )
        ),
        "name": lambda self, v: as_str(v, name=self.__descriptions__["name"]),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


# --- Scene Options ---
@dataclass(slots=True)
class OptsScene:
    is_new: bool = True
    fig_size: Tuple[float, float] = (1920, 1360)
    bgcolor: ColorRGB = (1.0, 1.0, 1.0)
    fgcolor: ColorRGB = (0.0, 0.0, 0.0)
    name: str = "None"
    azimuth: Number = 45
    elevation: Number = 54.735610317245346
    roll: Number = 0
    distance: Optional[Number] = None
    focal_point: Optional[Vect(3)] = None
    name: Optional[str] = "None"

    __descriptions__ = {
        "is_new": "whether to create a new scene",
        "fig_size": "size of figure window (width, height)",
        "bgcolor": "background color (RGB)",
        "fgcolor": "foreground color (RGB)",
        "name": "name identifier of scene",
        "azimuth": "azimuth angle of camera",
        "elevation": "elevation angle of camera",
        "roll": "roll angle of camera",
        "distance": "distance of camera from focal point",
        "focal_point": "3D focal point of camera",
    }

    _validators = {
        "fig_size": lambda self, v: tuple(
            as_Vect(v, dim=2, name=self.__descriptions__["fig_size"])
        ),
        "bgcolor": lambda self, v: as_ColorRGB(
            v, name=self.__descriptions__["bgcolor"]
        ),
        "fgcolor": lambda self, v: as_ColorRGB(
            v, name=self.__descriptions__["fgcolor"]
        ),
        "name": lambda self, v: as_str(v, name=self.__descriptions__["name"]),
        "azimuth": lambda self, v: as_Number(v, name=self.__descriptions__["azimuth"]),
        "elevation": lambda self, v: as_Number(
            v, name=self.__descriptions__["elevation"]
        ),
        "roll": lambda self, v: as_Number(v, name=self.__descriptions__["roll"]),
        "distance": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["distance"])
        ),
        "focal_point": lambda self, v: (
            None if v is None else as_Vect(v, name=self.__descriptions__["focal_point"])
        ),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


# --- Plane Options ---
@dataclass(slots=True)
class OptsPlaneGrid:
    normal: Optional[Vect(3)] = None
    spacing1: Optional[Number] = None
    spacing2: Optional[Number] = None
    size: Optional[Number] = None
    shape: Literal["circle", "rectangle"] = "rectangle"
    origin: Vect(3) = (0, 0, 0)
    axis1: Optional[Vect(3)] = None
    corners_limit: Optional[np.ndarray] = None
    grid_offset: Vect(3) = (0, 0, 0)
    grid_transform: Tensor((3, 3)) = field(default_factory=lambda: np.eye(3))

    __descriptions__ = {
        "normal": "normal of plane",
        "spacing1": "grid spacing along axis1",
        "spacing2": "grid spacing along axis2",
        "size": "size of plane",
        "origin": "origin of plane",
        "axis1": "first in-plane axis",
        "corners_limit": "bounding box corners (8×3 array)",
        "grid_offset": "translation offset of grid",
        "grid_transform": "grid transform matrix (3×3)",
        "shape": "plane shape (circle or rectangle)",
    }

    _validators = {
        "normal": lambda self, v: (
            None
            if v is None
            else as_Vect(v, name=self.__descriptions__["normal"], is_norm=True)
        ),
        "origin": lambda self, v: as_Vect(v, name=self.__descriptions__["origin"]),
        "grid_offset": lambda self, v: as_Vect(
            v, name=self.__descriptions__["grid_offset"]
        ),
        "axis1": lambda self, v: (
            None
            if v is None
            else as_Vect(v, name=self.__descriptions__["axis1"], is_norm=True)
        ),
        "spacing1": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["spacing1"])
        ),
        "spacing2": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["spacing2"])
        ),
        "size": lambda self, v: (
            None if v is None else as_Number(v, name=self.__descriptions__["size"])
        ),
        "grid_transform": lambda self, v: as_Tensor(
            v, (3, 3), name=self.__descriptions__["grid_transform"]
        ),
        "corners_limit": lambda self, v: (
            None
            if v is None
            else as_Tensor(v, (8, 3), name=self.__descriptions__["corners_limit"])
        ),
        "shape": lambda self, v: (
            v
            if v in ("circle", "rectangle")
            else (_ for _ in ()).throw(
                ValueError(
                    f"Invalid {self.__descriptions__['shape']}: {v!r}. "
                    f"Allowed values: 'circle', 'rectangle'"
                )
            )
        ),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


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


# --- Extent Options ---
@dataclass(slots=True)
class OptsExtent:
    corners: Optional[np.ndarray] = None
    radius: Number = 1.0
    sides: Number = 6
    opacity: Number = 1.0
    color: ColorRGB = (0, 0, 0)
    name: str = "None"

    __descriptions__ = {
        "corners": "bounding box corners (8×3 array)",
        "radius": "radius of extent tubes",
        "sides": "sides number of extent tubes",
        "opacity": "opacity of extent tubes",
        "color": "RGB color of extent tubes",
        "name": "name of extent",
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
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


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

    __descriptions__ = {
        "radius": "radius of tube",
        "opacity": "opacity of tube",
        "color": "RGB color of tube surface",
        "sides": "number of sides of tube",
        "specular": "strength of specular highlight",
        "specular_color": "RGB color of specular highlight",
        "specular_power": "shininess of specular highlight",
        "name": "name identifier of tube",
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
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)
        object.__setattr__(self, key, value)


def merge_opts(opts, kwargs, prefix=""):
    """
    Update a dataclass instance `opts` with values from `kwargs` whose
    keys start with a given prefix. The prefix is removed before matching
    the remaining part of the key to a field name in the dataclass.

    Parameters
    ----------
    opts : dataclass instance
        The target dataclass object to be updated.
    kwargs : dict
        A dictionary of keyword arguments that may contain keys with the
        specified prefix. Matching keys will be consumed (removed) from
        this dictionary.
    prefix : str, optional
        The prefix used to identify relevant keys in `kwargs`. Defaults to "".

    Returns
    -------
    dataclass instance
        A new dataclass object with updated field values.

    Raises
    ------
    TypeError
        If `opts` is not a dataclass instance.

    Example
    -------
    >>> @dataclass
    ... class LineOpts:
    ...     color: str = "blue"
    ...     width: int = 1
    ...
    >>> opts = LineOpts()
    >>> kwargs = {"line_color": "red", "line_width": 2, "alpha": 0.5}
    >>> merge_opts(opts, kwargs, prefix="line_")
    LineOpts(color='red', width=2)
    >>> kwargs
    {'alpha': 0.5}
    """
    if not is_dataclass(opts):
        raise TypeError("opts must be a dataclass instance")

    # Collect dataclass field names
    field_names = {f.name for f in fields(opts)}

    updates = {}
    for key, val in list(kwargs.items()):
        if key.startswith(prefix):
            name = key[len(prefix) :]  # strip prefix
            if name in field_names:
                updates[name] = val
                kwargs.pop(key)  # consume the key

    return replace(opts, **updates)


def auto_opts_tubes(bindings: dict):

    def decorator(cls):
        for name, path in bindings.items():
            attrs = path.split(".")  # e.g. ["actor", "property", "diffuse_color"]
            key = name[len("opts_") :]  # 去掉 "opts_" 前缀，映射到 _internal.xxx

            def getter(self, _key=key):
                return getattr(self._internal_opts, _key)

            def setter(self, value, _attrs=attrs, _key=key):
                # 1. 存到 _internal，会触发校验
                setattr(self._internal_opts, _key, value)

                processed = getattr(self._internal_opts, _key)

                for item in self._items:
                    target = item
                    for attr in _attrs[:-1]:
                        target = getattr(target, attr)
                    setattr(target, _attrs[-1], processed)

            setattr(cls, name, property(getter, setter))

        return cls

    return decorator
