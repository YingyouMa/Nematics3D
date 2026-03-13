from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping

import numpy as np

from Nematics3D.datatypes import Number, UNSET, Unset, Vect, as_Number, as_Vect, as_str
from .host_base import OptsBase


@dataclass(slots=True, repr=False)
class OptsBounds(OptsBase):
    origin: Vect(3) | Unset = UNSET
    axis1: Vect(3) | Unset = UNSET
    axis2: Vect(3) | None | Unset = UNSET

    length1: Number | Unset = UNSET
    length2: Number | None | Unset = UNSET
    length3: Number | None | Unset = UNSET

    alignment: Literal["min_corner", "center"] | Unset = UNSET

    __descriptions__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__descriptions__,
        "origin": (
            "The anchor point of the bounds box. "
            "Its geometric meaning is determined by ``alignment``."
        ),
        "axis1": (
            "The first direction of the bounds box. "
            "It is typically normalized by the host before use."
        ),
        "axis2": (
            "The second direction of the bounds box. "
            "The third direction is derived from the cross product of ``axis1`` and ``axis2``. "
            "If None, the host may infer a default orthogonal direction."
        ),
        "length1": "The box length along ``axis1``.",
        "length2": "The box length along ``axis2``.",
        "length3": (
            "The box length along the third direction derived from "
            "``axis1 x axis2``."
        ),
        "alignment": (
            "How ``origin`` is interpreted relative to the box. "
            'Typical values include ``"min_corner"`` and ``"center"``.'
        ),
    }

    _validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase._validators,
        "origin": lambda v, d: as_Vect(v, name=d, dim=3),
        "axis1": lambda v, d: as_Vect(v, name=d, dim=3, is_norm=True),
        "axis2": lambda v, d: None
        if v is None
        else as_Vect(v, name=d, dim=3, is_norm=True),
        "length1": lambda v, d: as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "length2": lambda v, d: None
        if v is None
        else as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "length3": lambda v, d: None
        if v is None
        else as_Number(v, name=d, value_range=(1e-12, np.inf)),
        "alignment": lambda v, d: as_str(v, name=d, pool=("min_corner", "center")),
    }

    _DEFAULTS_FROZEN: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(OptsBase._DEFAULTS_FROZEN),
            "tag": "bounds options",
            "origin": (0.0, 0.0, 0.0),
            "axis1": (1.0, 0.0, 0.0),
            "axis2": None,
            "length2": None,
            "length3": None,
            "alignment": "min_corner",
        }
    )
