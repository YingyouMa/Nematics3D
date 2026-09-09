import sys
from pathlib import Path
import types
import unittest

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.grid_field import GridFieldDataset, InputGridField  # noqa: E402
from nematics3d.sample.plane_grid import OptsPlaneGrid  # noqa: E402
from nematics3d.sample.plane_grid_base import PlaneGridBase  # noqa: E402
from nematics3d.sample.plane_grid_polar import (  # noqa: E402
    OptsPlaneGridPolar,
    PlaneGridPolar,
)
from nematics3d.sample.vector_plane import VectorPlane  # noqa: E402


class TestVectorPlane(unittest.TestCase):
    def test_plane_grid_polar_uses_shared_plane_grid_base(self):
        self.assertTrue(issubclass(PlaneGridPolar, PlaneGridBase))

    def test_vector_plane_builds_polar_grid_from_polar_opts(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(5, 5, 5)))
        values = dataset.act_generate_grid()
        field = dataset.act_add_field("v", values)

        plane = VectorPlane(
            interpolator=field.act_add_interpolator(),
            opts=OptsPlaneGridPolar(
                origin=(2.0, 2.0, 2.0),
                normal=(0.0, 0.0, 1.0),
                theta0_axis=(1.0, 0.0, 0.0),
                r_min=0.0,
                layers=2,
                dr=1.0,
                arc_dist=1.0,
            ),
        )

        self.assertIsInstance(plane.grid, PlaneGridPolar)
        expected = field.act_interpolate(plane.grid.entity_grid)
        self.assertTrue(np.allclose(plane.result, expected))

    def test_vector_plane_samples_vector_field_and_caches_magnitude(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        values = dataset.act_generate_grid()
        field = dataset.act_add_field("v", values)
        interpolator = field.act_add_interpolator()

        plane = VectorPlane(
            interpolator=interpolator,
            name="vector-plane",
            opts=OptsPlaneGrid(
                normal=(0.0, 0.0, 1.0),
                axis1=(1.0, 0.0, 0.0),
                origin=(1.0, 1.0, 1.0),
                spacing=1.0,
                size=2.0,
            ),
        )

        expected = field.act_interpolate(plane.grid.entity_grid)

        self.assertEqual(plane.result.shape[1], 3)
        self.assertTrue(np.allclose(plane.result, expected))
        self.assertTrue(
            np.allclose(plane.calc_magnitude, np.linalg.norm(plane.result, axis=1))
        )
        self.assertTrue(
            np.allclose(
                plane.calc_magnitude_all,
                np.linalg.norm(plane.calc_result_all, axis=1),
            )
        )

    def test_vector_plane_refresh_tracks_grid_changes(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(4, 4, 4)))
        values = dataset.act_generate_grid()
        field = dataset.act_add_field("v", values)

        plane = VectorPlane(
            interpolator=field.act_add_interpolator(),
            opts=OptsPlaneGrid(
                normal=(0.0, 0.0, 1.0),
                axis1=(1.0, 0.0, 0.0),
                origin=(0.0, 0.0, 0.0),
                spacing=1.0,
                size=2.0,
            ),
        )

        result_initial = plane.result.copy()
        plane.grid.act_commit(origin=(2.0, 2.0, 2.0))

        self.assertFalse(np.allclose(plane.result, result_initial))
        self.assertTrue(
            np.allclose(plane.calc_magnitude, np.linalg.norm(plane.result, axis=1))
        )

    def test_vector_plane_rejects_non_vector_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        field = dataset.act_add_field(
            "scalar",
            np.arange(8, dtype=float).reshape(2, 2, 2),
        )

        with self.assertRaisesRegex(ValueError, r"shape \(N, 3\)"):
            VectorPlane(
                interpolator=field.act_add_interpolator(),
                opts=OptsPlaneGrid(
                    normal=(0.0, 0.0, 1.0),
                    axis1=(1.0, 0.0, 0.0),
                    origin=(0.0, 0.0, 0.0),
                    spacing=1.0,
                    size=1.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
