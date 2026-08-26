"""Semantic data aliases and runtime conversion helpers."""

from .bool import as_bool
from .defect_index import DefectIndex, as_defect_index
from .dimension_info import DimensionInfo, as_dimension_info
from .director_field import as_director_field, nField
from .misc import *
from .number import Number, as_number, as_value_range
from .q_field import QField5, QField9, as_qfield5, as_qfield9
from .scalar_field import SField, ScalarField, as_scalar_field
from .tensor import Tensor, as_tensor
from .vector import Vect, as_vector
