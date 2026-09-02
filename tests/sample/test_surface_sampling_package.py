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
