"""Physical fields defined on a shared structured lattice."""

from nematics3d.datatypes import as_grid_shape

from .dataset import FieldData, GridFieldDataset
from .derivatives import SpatialDerivativeInfo
from .input import InputGridField
from .interpolation import GridInterpolator
from .smoothing import GaussianSmoothInfo

__all__ = [
    "FieldData",
    "GaussianSmoothInfo",
    "GridFieldDataset",
    "GridInterpolator",
    "InputGridField",
    "SpatialDerivativeInfo",
    "as_grid_shape",
]
