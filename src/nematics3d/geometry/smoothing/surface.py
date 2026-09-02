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

    ``cutoff_wavelength`` is the main geometric smoothing parameter. It is a
    physical wavelength in the same length units as the surface coordinates,
    not a mesh spacing or a displacement distance. With the discrete
    Laplace--Beltrami convention ``L phi = -kappa phi``, define

        kappa_c = (2*pi / cutoff_wavelength)**2.

    Nematics3D defines ``cutoff_wavelength`` as the -3 dB amplitude cutoff of
    the *complete* smoothing pass:

        G_N(kappa_c) = 1/sqrt(2).

    Thus a surface mode whose wavelength equals ``cutoff_wavelength`` retains
    1/sqrt(2) of its original amplitude after smoothing. Longer-wavelength
    modes are intended to be preserved more strongly, while shorter-wavelength
    modes are intended to be suppressed more strongly.

    A Taubin iteration pair applies coefficients ``lambda > 0`` and ``mu < 0``.
    This interface parameterizes their relative magnitude by

        taubin_ratio = -mu / lambda,

    so ``mu = -taubin_ratio * lambda``. The default ratio is 1.0674, close to
    the ratio of the widely used Taubin coefficients 0.6307 and -0.6732. The
    ratio is dimensionless; unlike those raw coefficients, it remains meaningful
    when a dimensional Laplace--Beltrami operator is used.

    The iteration count is intentionally not a public smoothing option. For a
    fixed ``cutoff_wavelength`` and ``taubin_ratio``, the implementation chooses
    the smallest positive integer N for which the resolved Taubin filter is
    stable over the discrete surface spectrum. This also minimizes the
    pass-band amplification among stable choices for the same ratio.

    For any candidate N, let

        q = 2**(-1 / (2*N)),
        x = lambda * kappa_c,
        r = taubin_ratio.

    The cutoff condition requires one iteration pair to satisfy

        (1 - x) * (1 + r*x) = q,

    giving the positive solution

        x = ((r - 1) + sqrt((r - 1)**2 + 4*r*(1 - q))) / (2*r),
        lambda = x / kappa_c,
        mu = -r * lambda.

    If ``kappa_max`` is the largest resolved Laplace--Beltrami eigenvalue, the
    candidate is accepted only when the high-frequency edge is not amplified:

        abs((1 - lambda*kappa_max) * (1 - mu*kappa_max)) <= 1.

    Candidates are tested in increasing N, and the first accepted N is used.
    The resolved N, lambda, mu, and kappa_max are calculation results rather
    than user-controlled options and should be exposed by ``SmoothedSurface``
    as diagnostics.

    Important readable attributes
    -----------------------------
    cutoff_wavelength
        Required positive physical wavelength defining the -3 dB amplitude
        cutoff of the complete smoothing pass. No universal default exists
        because the appropriate value depends on the surface length scale.
    taubin_ratio
        Positive dimensionless ratio ``-mu/lambda``. Values greater than 1
        correspond to the usual Taubin choice in which the negative step has a
        slightly larger magnitude than the positive step. The library default
        is 1.0674.
    """

    cutoff_wavelength: Number | Unset = UNSET
    taubin_ratio: Number | Unset = UNSET

    __attrs__: ClassVar[Mapping[str, str]] = {
        **OptsBase.__attrs__,
        "cutoff_wavelength": (
            "positive physical wavelength defining the -3 dB amplitude cutoff "
            "of the complete surface-smoothing pass"
        ),
        "taubin_ratio": (
            "dimensionless Taubin coefficient ratio -mu/lambda; values greater "
            "than 1 give the usual slightly stronger negative step"
        ),
    }

    impl_validators: ClassVar[Mapping[str, Any]] = {
        **OptsBase.impl_validators,
        "cutoff_wavelength": lambda v, d: as_number(
            v,
            name=d,
            value_range=(np.nextafter(0.0, 1.0), np.inf),
        ),
        "taubin_ratio": lambda v, d: as_number(
            v,
            name=d,
            value_range=(np.nextafter(1.0, np.inf), np.inf),
        ),
    }

    impl_defaults_frozen: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            **dict(getattr(OptsBase, "impl_defaults_frozen", {})),
            "tag": "smoothed surface options",
            "taubin_ratio": 1.0674,
        }
    )
