import sys
from pathlib import Path
import types

import numpy as np
import pyvista as pv
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.geometry.smoothing.surface import (
    OptsSmoothedSurface,
    SmoothedSurface,
    SurfaceSmoothingConfigError,
)


def _tetra_surface(scale=1.0):
    points = scale * np.array(
        [
            [1.0, 1.0, 1.2],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [3, 0, 1, 2],
            [3, 0, 3, 1],
            [3, 0, 2, 3],
            [3, 1, 3, 2],
        ],
        dtype=np.int64,
    ).ravel()
    return pv.PolyData(points, faces)


def _direct_smooth(mesh, opts):
    return mesh.smooth_taubin(
        n_iter=opts.n_iter,
        pass_band=opts.pass_band,
        edge_angle=opts.edge_angle,
        feature_angle=opts.feature_angle,
        boundary_smoothing=opts.boundary_smoothing,
        feature_smoothing=opts.feature_smoothing,
        non_manifold_smoothing=False,
        normalize_coordinates=opts.normalize_coordinates,
        inplace=False,
        progress_bar=False,
    )


def test_smoothed_surface_matches_pyvista_and_result_contract():
    mesh = _tetra_surface()
    mesh.point_data["sample"] = np.arange(mesh.n_points)
    points_before = np.asarray(mesh.points).copy()

    surface = SmoothedSurface(mesh)
    expected = _direct_smooth(mesh, surface.opts)

    assert isinstance(surface.result, pv.PolyData)
    assert surface.result.n_points == mesh.n_points
    assert surface.result.n_cells == mesh.n_cells
    assert len(surface.result.point_data) == 0
    assert len(surface.result.cell_data) == 0
    assert len(surface.result.field_data) == 0

    np.testing.assert_array_equal(mesh.points, points_before)
    np.testing.assert_array_equal(surface.raw_surface.points, points_before)
    np.testing.assert_allclose(surface.vertices, expected.points)
    assert surface.vertices.flags.writeable is False

    expected_vectors = surface.vertices - points_before
    np.testing.assert_allclose(surface.error_vectors, expected_vectors)
    np.testing.assert_allclose(
        surface.error_scalars,
        np.linalg.norm(expected_vectors, axis=1),
    )
    assert surface.error_vectors.shape == (mesh.n_points, 3)
    assert surface.error_scalars.shape == (mesh.n_points,)
    assert surface.error_vectors.flags.writeable is False
    assert surface.error_scalars.flags.writeable is False


def test_smoothed_surface_defaults_and_override():
    opts = OptsSmoothedSurface()
    surface = SmoothedSurface(
        _tetra_surface(),
        opts=opts,
        opts_defaults_override={"pass_band": 0.2},
    )

    assert surface.opts is not opts
    assert surface.opts.n_iter == 20
    assert surface.opts.pass_band == 0.2
    assert surface.opts.boundary_smoothing is True
    assert surface.opts.feature_smoothing is False
    assert surface.opts.feature_angle == 45.0
    assert surface.opts.edge_angle == 15.0
    assert surface.opts.normalize_coordinates is False


def test_smoothed_surface_opts_commit_recomputes_from_raw_surface():
    mesh = _tetra_surface()
    surface = SmoothedSurface(mesh, pass_band=0.2)
    vertices_before = surface.vertices.copy()

    surface.act_commit(pass_band=0.02)

    expected = _direct_smooth(mesh, surface.opts)
    np.testing.assert_allclose(surface.vertices, expected.points)
    assert not np.array_equal(surface.vertices, vertices_before)


def test_smoothed_surface_raw_surface_commit_recomputes():
    surface = SmoothedSurface(_tetra_surface(), pass_band=0.05)
    replacement = _tetra_surface(scale=2.0)

    surface.act_commit(surface=replacement)

    expected = _direct_smooth(replacement, surface.opts)
    np.testing.assert_allclose(surface.vertices, expected.points)
    np.testing.assert_allclose(
        surface.error_vectors,
        expected.points - replacement.points,
    )


def test_smoothed_surface_zero_iterations_is_identity():
    mesh = _tetra_surface()

    surface = SmoothedSurface(mesh, n_iter=0)

    np.testing.assert_array_equal(surface.vertices, mesh.points)
    np.testing.assert_array_equal(surface.error_vectors, np.zeros((mesh.n_points, 3)))
    np.testing.assert_array_equal(surface.error_scalars, np.zeros(mesh.n_points))


@pytest.mark.parametrize(
    ("kwargs", "attr_name", "default"),
    [
        ({"n_iter": -1}, "n_iter", 20),
        ({"n_iter": 1.5}, "n_iter", 20),
        ({"pass_band": -0.1}, "pass_band", 0.1),
        ({"pass_band": 2.1}, "pass_band", 0.1),
        ({"feature_angle": 181}, "feature_angle", 45.0),
        ({"edge_angle": -1}, "edge_angle", 15.0),
        ({"boundary_smoothing": 1}, "boundary_smoothing", True),
    ],
)
def test_smoothed_surface_invalid_opts_fall_back_to_defaults(
    kwargs, attr_name, default
):
    surface = SmoothedSurface(_tetra_surface(), **kwargs)

    assert getattr(surface.opts, attr_name) == default


def test_smoothed_surface_rejects_empty_or_nonfinite_surface():
    with pytest.raises(SurfaceSmoothingConfigError, match="non-empty"):
        SmoothedSurface(pv.PolyData())

    mesh = _tetra_surface()
    mesh.points[0, 0] = np.nan
    with pytest.raises(SurfaceSmoothingConfigError, match="finite"):
        SmoothedSurface(mesh)
