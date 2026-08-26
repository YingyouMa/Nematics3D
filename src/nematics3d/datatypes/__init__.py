"""Semantic data aliases and runtime conversion helpers."""

from .axes import as_axes
from .bool import as_bool
from .box_size_periodic import BoxSizePeriodic, as_box_size_periodic
from .color_rgb import ColorRGB, as_ColorRGB, as_ColorRGB_array
from .defect_index import DefectIndex, as_defect_index
from .dimension_info import DimensionInfo, as_dimension_info
from .director_field import as_director_field, nField
from .list import as_list
from .misc import *
from .number import Number, as_number, as_value_range
from .q_field import QField5, QField9, as_qfield5, as_qfield9
from .scalar_field import SField, ScalarField, as_scalar_field
from .string import as_str
from .tensor import Tensor, as_tensor
from .vector import Vect, as_vector
