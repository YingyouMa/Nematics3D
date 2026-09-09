import numpy as np

from nematics3d.analysis import (
    NMLPrincipalPlaneResult,
    nml_principal_plane_analysis,
)
from nematics3d.analysis.principal_plane import NMLIterationResult
from nematics3d.analysis.principal_plane import (
    NMLPrincipalPlaneResult as LegacyNMLPrincipalPlaneResult,
)
from nematics3d.q_field import get_q


class _UniformQField:
    def __init__(self, director):
        self.q = get_q(np.asarray(director, dtype=float), S=1)

    def act_interpolate(self, points, *, is_index=True):
        del is_index
        return np.broadcast_to(self.q, (len(points), 3, 3)).copy()


def test_principal_plane_public_exports_share_canonical_types():
    assert LegacyNMLPrincipalPlaneResult is NMLPrincipalPlaneResult
    assert issubclass(NMLIterationResult, object)


def test_uniform_texture_converges_and_returns_consistent_plane():
    q_obj = _UniformQField([1.0, 0.0, 0.0])
    required_points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    result = nml_principal_plane_analysis(
        q_obj,
        required_points,
        expand_factors=1.0,
        min_lengths=2.0,
        spacing=1.0,
        angle_tol_deg=90.0,
        max_iterations=2,
    )

    assert result.converged
    assert len(result.iterations) == 1
    assert result.axes.shape == (3, 3)
    assert result.plane_axes.shape == (3, 2)
    assert result.plane_normal.shape == (3,)
    np.testing.assert_allclose(result.plane_axes, result.axes[:, :2])
    np.testing.assert_allclose(result.plane_normal, result.axes[:, 2])
    np.testing.assert_allclose(result.plane_center, result.minimal_bounds.opts.origin)


def test_principal_plane_requires_geometry_source():
    q_obj = _UniformQField([1.0, 0.0, 0.0])

    try:
        nml_principal_plane_analysis(q_obj)
    except ValueError as error:
        assert "required_points" in str(error)
        assert "seed_bounds" in str(error)
    else:
        raise AssertionError("Expected missing geometry source to raise ValueError.")
