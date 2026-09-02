"""Surface-smoothing object model and Taubin smoothing configuration."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from ...core.host_base import OptsBase
from ...datatypes import Number, UNSET, Unset, as_number


@dataclass(slots=True, repr=False)
class OptsSmoothedSurface(OptsBase):
    """Options controlling wavelength-based Taubin smoothing of a surface.

    The public smoothing scale is specified by ``cutoff_wavelength`` rather
    than by Taubin's raw ``lambda`` and ``mu`` coefficients.

    ``cutoff_wavelength`` is a physical wavelength measured in the same length
    units as the surface coordinates. For the discrete Laplace--Beltrami mode
    convention ``L phi = -kappa phi``, wavelength and eigenvalue are related by

        kappa_c = (2*pi / cutoff_wavelength)**2.

    Nematics3D defines this wavelength as the -3 dB cutoff of the *complete*
    smoothing pass: a mode at ``kappa_c`` retains an amplitude gain of
    ``1/sqrt(2)`` after all Taubin iteration pairs. Longer-wavelength modes are
    preserved more strongly; shorter-wavelength modes are suppressed more
    strongly. Thus the parameter describes the geometric roughness scale to be
    filtered, not a mesh spacing and not a displacement distance.

    The implementation uses the symmetric Taubin family ``mu = -lambda``. If
    ``N`` is the resolved number of iteration pairs, one pair has spectral gain

        g(kappa) = 1 - lambda**2 * kappa**2,

    and the complete pass has

        G_N(kappa) = (1 - lambda**2 * kappa**2)**N.

    Requiring ``G_N(kappa_c) = 1/sqrt(2)`` gives

        lambda = sqrt(1 - (1/sqrt(2))**(1/N)) / kappa_c,
        mu = -lambda.

    Therefore the original Taubin parameters ``(lambda, mu, N)`` are not three
    independent public controls here. The constraint ``mu = -lambda`` reduces
    them to two degrees of freedom, represented by
    ``(cutoff_wavelength, iterations)``. Once those two are known, ``lambda``
    and ``mu`` are determined uniquely by the equations above.

    Important readable attributes
    -----------------------------
    cutoff_wavelength
        Required positive physical wavelength defining the -3 dB cutoff of the
        complete smoothing pass. No universal default is provided because its
        appropriate value depends on the physical length scale of the surface.
    iterations
        Number of Taubin lambda/mu iteration pairs. A positive integer uses that
        exact count. ``None`` explicitly requests automatic selection of the
        smallest count satisfying the implementation's stability criterion.
        If the user leaves this option ``UNSET``, the normal option-default
        mechanism supplies the library default iteration count.
    """

    cutoff_wavelength: Number | Unset = UNSET
    iterations: int | None | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "cutoff_wavelength": (
            "positive physical wavelength defining the -3 dB amplitude cutoff "
            "of the complete surface-smoothing pass"
        ),
        "iterations": (
            "number of Taubin lambda/mu iteration pairs; None explicitly "
            "requests automatic stable iteration selection"
        ),
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase.impl_validators,
        "cutoff_wavelength": lambda v, d: as_number(
            v,
            name=d,
            value_range=(np.nextafter(0.0, 1.0), np.inf),
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
            "iterations": 10,
        }
    )
