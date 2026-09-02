import numpy as np
import pytest
import pyvista as pv
import importlib

from nematics3d.datatypes import UNSET
from nematics3d.classes.surface_sampling import (
    OptsSurfaceSampling as LegacyOptsSurfaceSampling,
)
from nematics3d.classes.surface_sampling import SurfaceSampling as LegacySurfaceSampling
from nematics3d.sample import OptsSurfaceSampling, SurfaceSampling


def test_surface_sampling_uses_canonical_sample_package():
    assert LegacySurfaceSampling is SurfaceSampling
    assert LegacyOptsSurfaceSampling is OptsSurfaceSampling
    assert SurfaceSampling.__module__ == "nematics3d.sample.surface_sampling"
    assert OptsSurfaceSampling.__module__ == "nematics3d.sample.surface_sampling"


def test_surface_sampling_prepares_valid_surface_during_initial_commit():
    sampling = SurfaceSampling(
        pv.Plane().triangulate(),
        opts_defaults_override={"default_sample_count_target": 8},
    )

    assert sampling.calc_surface_clean.n_cells > 0
    assert sampling.calc_surface_area > 0.0
    assert sampling.calc_spacing_effective > 0.0
    assert sampling.result.shape == (8, 3)
    assert not sampling.result.flags.writeable
    assert not sampling.calc_surface_points.flags.writeable
    assert not sampling.calc_surface_normals.flags.writeable
    assert sampling.calc_sample_normals is UNSET
    assert sampling.calc_nearest_distance_mean is UNSET


def test_surface_sampling_rejects_zero_area_geometry_during_initial_commit():
    point_cloud = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="surface cells"):
        SurfaceSampling(point_cloud)


def test_surface_sampling_optional_outputs_follow_opts_switches():
    sampling = SurfaceSampling(
        pv.Sphere(theta_resolution=12, phi_resolution=12),
        opts_defaults_override={"default_sample_count_target": 12},
    )

    sampling.opts.is_calc_sample_normals = True
    assert sampling.calc_sample_cell_ids.shape == (12,)
    assert sampling.calc_sample_barycentric.shape == (12, 3)
    assert sampling.calc_sample_normals.shape == (12, 3)
    assert np.allclose(np.sum(sampling.calc_sample_barycentric, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(sampling.calc_sample_normals, axis=1), 1.0)
    assert not sampling.calc_sample_cell_ids.flags.writeable
    assert not sampling.calc_sample_barycentric.flags.writeable
    assert not sampling.calc_sample_normals.flags.writeable

    sampling.opts.is_calc_spacing_statistics = True
    assert sampling.calc_nearest_distance_mean > 0.0
    assert sampling.calc_nearest_distance_min > 0.0
    assert sampling.calc_nearest_distance_std >= 0.0

    sampling.opts.is_calc_sample_normals = False
    sampling.opts.is_calc_spacing_statistics = False
    assert sampling.calc_sample_cell_ids is UNSET
    assert sampling.calc_sample_barycentric is UNSET
    assert sampling.calc_sample_normals is UNSET
    assert sampling.calc_nearest_distance_mean is UNSET
    assert sampling.calc_nearest_distance_min is UNSET
    assert sampling.calc_nearest_distance_std is UNSET


def test_surface_sampling_rejects_target_above_safety_limit():
    with pytest.raises(ValueError, match="exceeding max_sample_count=10"):
        SurfaceSampling(
            pv.Plane(),
            opts_defaults_override={
                "spacing": 0.01,
                "max_sample_count": 10,
            },
        )


def test_surface_sampling_optional_layers_do_not_resample(monkeypatch):
    sampling_module = importlib.import_module("nematics3d.sample.surface_sampling")
    sampling = SurfaceSampling(
        pv.Sphere(theta_resolution=12, phi_resolution=12),
        opts_defaults_override={"default_sample_count_target": 12},
    )
    initial_points = sampling.result
    call_counts = {
        "prepare": 0,
        "candidates": 0,
        "normals": 0,
        "statistics": 0,
    }

    def wrap_helper(name):
        original = getattr(sampling_module, name)
        count_key = {
            "_helper_prepare_surface": "prepare",
            "_helper_sample_triangle_candidates": "candidates",
            "_helper_calc_sample_surface_geometry": "normals",
            "_helper_calc_spacing_statistics": "statistics",
        }[name]

        def wrapped(*args, **kwargs):
            call_counts[count_key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(sampling_module, name, wrapped)

    for helper_name in (
        "_helper_prepare_surface",
        "_helper_sample_triangle_candidates",
        "_helper_calc_sample_surface_geometry",
        "_helper_calc_spacing_statistics",
    ):
        wrap_helper(helper_name)

    sampling.opts.is_calc_spacing_statistics = True
    assert call_counts == {
        "prepare": 0,
        "candidates": 0,
        "normals": 0,
        "statistics": 1,
    }
    assert sampling.result is initial_points

    sampling.opts.is_calc_sample_normals = True
    assert call_counts == {
        "prepare": 0,
        "candidates": 0,
        "normals": 1,
        "statistics": 1,
    }
    assert sampling.result is initial_points

    sampling.opts.seed = 1
    assert call_counts == {
        "prepare": 0,
        "candidates": 1,
        "normals": 2,
        "statistics": 2,
    }
    assert sampling.result is not initial_points

    sampling.opts.seed = 1
    assert call_counts == {
        "prepare": 0,
        "candidates": 1,
        "normals": 2,
        "statistics": 2,
    }

    sampling.act_commit(surface=pv.Plane())
    assert call_counts == {
        "prepare": 1,
        "candidates": 2,
        "normals": 3,
        "statistics": 3,
    }
