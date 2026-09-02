import sys
from pathlib import Path
import types

import numpy as np
import pyvista as pv

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
            [1.0, 1.0, 1.0],
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


def test_smoothed_surface_initialization_and_result_contract():
    mesh = _tetra_surface()
    points_before = np.asarray(mesh.points).copy()

    surface = SmoothedSurface(mesh, cutoff_wavelength=7.0)

    assert surface.calc_is_smoothed is True
    assert surface.calc_status == "Success"
    assert isinstance(surface.result, pv.PolyData)
    assert surface.result.is_all_triangles
    assert surface.result.n_points == mesh.n_points
    assert surface.result.n_cells == mesh.n_cells
    assert surface.vertices.shape == points_before.shape
    assert np.all(np.isfinite(surface.vertices))
    assert surface.vertices.flags.writeable is False

    np.testing.assert_array_equal(mesh.points, points_before)
    np.testing.assert_array_equal(surface.raw_surface.points, points_before)
    np.testing.assert_allclose(surface.result.points, surface.vertices)

    assert surface.calc_iterations >= 1
    assert surface.calc_lambda > 0.0
    assert surface.calc_mu < 0.0
    assert surface.calc_kappa_max > 0.0
    assert surface.calc_kappa_cutoff > 0.0

    # Large mesh/operator caches are implementation details, not public calc attrs.
    for removed_name in (
        "calc_surface_initial",
        "calc_vertices_initial",
        "calc_faces",
        "calc_mass_lumped",
        "calc_stiffness_matrix",
        "calc_vertices_result",
        "calc_surface_result",
    ):
        assert removed_name not in type(surface).__attr_defs__


def test_smoothed_surface_complete_cutoff_gain_is_minus_3db():
    surface = SmoothedSurface(_tetra_surface(), cutoff_wavelength=7.0)

    pair_gain = (
        (1.0 - surface.calc_lambda * surface.calc_kappa_cutoff)
        * (1.0 - surface.calc_mu * surface.calc_kappa_cutoff)
    )
    total_gain = pair_gain ** surface.calc_iterations

    np.testing.assert_allclose(total_gain, 1.0 / np.sqrt(2.0), rtol=1e-12, atol=1e-12)


def test_smoothed_surface_iteration_count_is_minimum_stable():
    surface = SmoothedSurface(_tetra_surface(), cutoff_wavelength=7.0)

    def edge_gain(iterations):
        lambda_, mu = surface._helper_coefficients_for_iterations(
            surface.calc_kappa_cutoff,
            surface.opts.taubin_ratio,
            iterations,
        )
        return (
            (1.0 - lambda_ * surface.calc_kappa_max)
            * (1.0 - mu * surface.calc_kappa_max)
        )

    assert abs(edge_gain(surface.calc_iterations)) <= 1.0 + 1e-12
    if surface.calc_iterations > 1:
        assert abs(edge_gain(surface.calc_iterations - 1)) > 1.0


def test_smoothed_surface_opts_object_and_default_override():
    opts = OptsSmoothedSurface(cutoff_wavelength=7.0)
    surface = SmoothedSurface(
        _tetra_surface(),
        opts=opts,
        opts_defaults_override={"taubin_ratio": 1.05},
    )

    assert surface.opts is opts
    np.testing.assert_allclose(surface.opts.taubin_ratio, 1.05)


def test_smoothed_surface_opts_update_reuses_fixed_operator():
    surface = SmoothedSurface(_tetra_surface(), cutoff_wavelength=7.0)
    stiffness_before = surface.impl_stiffness_matrix
    initial_vertices_before = surface.impl_vertices_initial
    vertices_before = surface.vertices.copy()

    surface.act_commit(cutoff_wavelength=6.5)

    assert surface.impl_stiffness_matrix is stiffness_before
    assert surface.impl_vertices_initial is initial_vertices_before
    assert surface.calc_is_smoothed is True
    assert not np.array_equal(surface.vertices, vertices_before)


def test_smoothed_surface_raw_surface_update_rebuilds_operator():
    surface = SmoothedSurface(_tetra_surface(), cutoff_wavelength=7.0)
    stiffness_before = surface.impl_stiffness_matrix
    initial_vertices_before = surface.impl_vertices_initial
    kappa_before = surface.calc_kappa_max

    surface.act_commit(surface=_tetra_surface(scale=2.0))

    assert surface.impl_stiffness_matrix is not stiffness_before
    assert surface.impl_vertices_initial is not initial_vertices_before
    assert surface.calc_is_smoothed is True
    np.testing.assert_allclose(surface.calc_kappa_max, kappa_before / 4.0, rtol=1e-12, atol=1e-12)


def test_smoothed_surface_requires_cutoff_wavelength():
    try:
        SmoothedSurface(_tetra_surface())
    except SurfaceSmoothingConfigError as error:
        assert "cutoff_wavelength" in str(error)
    else:
        raise AssertionError("missing cutoff_wavelength should raise")
