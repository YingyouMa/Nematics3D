"""Surface-smoothing object model and Taubin smoothing configuration."""

from dataclasses import dataclass
from math import sqrt
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from ...core.host_base import OptsBase
from ...datatypes import Number, UNSET, Unset, as_number


@dataclass(slots=True, repr=False)
class OptsSmoothedSurface(OptsBase):
    """Options controlling wavelength-based smoothing of triangulated surfaces.

    The public smoothing strength is expressed in geometric/spectral terms
    rather than directly exposing Taubin's ``lambda`` and ``mu`` coefficients.
    ``cutoff_wavelength`` defines the physical wavelength at which the filter
    response is calibrated, while ``attenuation`` is the remaining amplitude
    fraction at that wavelength after all smoothing iterations.

    ``iterations=None`` requests automatic iteration selection. The future
    ``SmoothedSurface`` implementation will choose the smallest iteration count
    satisfying its discrete-Laplacian stability criterion, then derive the
    internal Taubin coefficients from ``cutoff_wavelength``, ``attenuation``,
    and that resolved iteration count.

    Important readable attributes
    -----------------------------
    cutoff_wavelength
        Positive physical wavelength separating geometry that should largely be
        preserved from shorter-wavelength roughness that should be suppressed.
        ``None`` means no smoothing scale has been configured yet.
    attenuation
        Remaining amplitude fraction at ``cutoff_wavelength`` after the complete
        smoothing pass. It must lie in ``(0, 1]``. The default ``1/sqrt(2)`` is
        the conventional -3 dB amplitude gain.
    iterations
        Positive integer number of Taubin lambda/mu pairs. ``None`` requests
        automatic selection from the mesh spectrum and stability requirement.
    """

    cutoff_wavelength: Number | None | Unset = UNSET
    attenuation: Number | Unset = UNSET
    iterations: int | None | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "cutoff_wavelength": (
            "physical cutoff wavelength for surface smoothing; positive when "
            "specified, or None when the smoothing scale is not yet configured"
        ),
        "attenuation": (
            "remaining amplitude fraction at cutoff_wavelength after the full "
            "smoothing pass"
        ),
        "iterations": (
            "number of Taubin lambda/mu iteration pairs; None selects the "
            "smallest stable iteration count automatically"
        ),
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase.impl_validators,
        "cutoff_wavelength": lambda v, d: (
            None
            if v is None
            else as_number(v, name=d, value_range=(np.nextafter(0.0, 1.0), np.inf))
        ),
        "attenuation": lambda v, d: as_number(
            v,
            name=d,
            value_range=(np.nextafter(0.0, 1.0), 1.0),
        ),
        "iterations": lambda v, d: (
            None
            if v is None
            else as_number(v, name=d, is_integer=True, value_range=(1, np.inf))
        ),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(getattr(OptsBase, "impl_defaults_frozen", {})),
            "tag": "smoothed surface options",
            "cutoff_wavelength": None,
            "attenuation": 1.0 / sqrt(2.0),
            "iterations": None,
        }
    )
