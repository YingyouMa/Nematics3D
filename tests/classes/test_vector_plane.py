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

from nematics3d.classes.grid_field import GridFieldDataset, InputGridField
from nematics3d.classes.plane_grid import OptsPlaneGrid
from nematics3d.classes.vector_plane import VectorPlane


class TestVectorPlane(unittest.TestCase):
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
