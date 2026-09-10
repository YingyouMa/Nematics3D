"""Analysis helpers for lattice fields."""

from .bounds import (
    Bounds,
    BoundsData,
    OptsBounds,
    as_bounds,
    bounds_expanded,
    bounds_minimal_wrapping_points,
    bounds_sample_points,
    obb_bounds_from_fit,
)

from .fourier import (
    CorrelationResult,
    DistanceCorrelationResult,
    FourierResult,
    RadialSpectrumResult,
    act_correlation,
    act_correlation_values,
    act_distance,
    act_filter,
    act_fourier,
    act_inverse,
    act_mean_subtracted_values,
    act_radial_spectrum,
)
from .relaxation import (
    FitRelaxationResult,
    RelaxationLengthResult,
    ThresholdRelaxationResult,
    act_relaxation_length,
)
from .principal_plane import (
    NMLIterationResult,
    NMLPrincipalPlaneResult,
    nml_principal_plane_analysis,
)
from .sampling import sample_van_der_corput
from .surface_director import (
    SurfaceDirectorInterpolationResult,
    SurfaceDirectorProjectionResult,
    interpolate_surface_directors,
    project_surface_directors,
)
from .surface_streamline import SurfaceStreamlineResult, integrate_surface_streamline

__all__ = [
    "Bounds",
    "BoundsData",
    "CorrelationResult",
    "DistanceCorrelationResult",
    "FitRelaxationResult",
    "FourierResult",
    "OptsBounds",
    "NMLIterationResult",
    "NMLPrincipalPlaneResult",
    "RadialSpectrumResult",
    "QDiagonalizationResult",
    "RelaxationLengthResult",
    "ThresholdRelaxationResult",
    "SurfaceDirectorProjectionResult",
    "SurfaceDirectorInterpolationResult",
    "SurfaceStreamlineResult",
    "as_bounds",
    "act_correlation",
    "act_correlation_values",
    "act_distance",
    "act_filter",
    "act_fourier",
    "act_inverse",
    "act_mean_subtracted_values",
    "act_radial_spectrum",
    "act_relaxation_length",
    "bounds_expanded",
    "bounds_minimal_wrapping_points",
    "bounds_sample_points",
    "obb_bounds_from_fit",
    "nml_principal_plane_analysis",
    "q_diagonalize",
    "sample_van_der_corput",
    "project_surface_directors",
    "interpolate_surface_directors",
    "integrate_surface_streamline",
]


def __getattr__(name: str):
    if name in {"QDiagonalizationResult", "q_diagonalize"}:
        from ..q_field.diagonalization import QDiagonalizationResult, q_diagonalize

        return {
            "QDiagonalizationResult": QDiagonalizationResult,
            "q_diagonalize": q_diagonalize,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
