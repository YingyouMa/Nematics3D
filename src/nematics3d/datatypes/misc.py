"""
Miscellaneous semantic data aliases and runtime conversion helpers.

This module contains helpers that do not yet have a dedicated datatype module.
Dedicated validators are re-exported here temporarily for compatibility.
"""

import numpy as np

from .axes import as_axes
from .color_rgb import ColorRGB, as_ColorRGB, as_ColorRGB_array
from .lattice_field import GeneralField, MaskField, as_lattice_mask, as_real_lattice_field
from .list import as_list
from .number import Number, as_number, as_value_range
from .string import as_str
from .tensor import Tensor, as_tensor
from .unset import UNSET, Unset
from .vector import Vect, as_vector


def as_readonly_array(input_data, *, dtype=float, copy: bool = True) -> np.ndarray:
    """Return one NumPy array view/copy with write access disabled."""
    values = np.asarray(input_data, dtype=dtype)
    if copy:
        values = values.copy()
    values.setflags(write=False)
    return values
