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

from nematics3d.classes.plane_grid_polar import OptsPlaneGridPolar, PlaneGridPolar
from nematics3d.classes.q_plane import QPlanePolar


class TestPlaneGridPolar(unittest.TestCase):
    def test_plane_grid_polar_builds_physical_points_directly_from_basis(self):
        grid = PlaneGridPolar(
            opts=OptsPlaneGridPolar(
                origin=(10.0, 20.0, 30.0),
                normal=(0.0, 0.0, 1.0),
                theta0_axis=(1.0, 0.0, 0.0),
                r_min=1.0,
                layers=1,
                dr=1.0,
                arc_dist=10.0,
            )
        )

        expected = np.array([[11.0, 20.0, 30.0]])

        self.assertTrue(np.allclose(grid.entity_grid_all, expected))
        self.assertTrue(np.allclose(grid.entity_polar[:, 0], np.array([1.0])))

    def test_opts_plane_grid_polar_no_longer_accepts_legacy_transform_kwargs(self):
        with self.assertRaisesRegex(TypeError, r"grid_offset"):
            OptsPlaneGridPolar(
                origin=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                dr=1.0,
                grid_offset=(1.0, 2.0, 3.0),
            )

        with self.assertRaisesRegex(TypeError, r"grid_transform"):
            OptsPlaneGridPolar(
                origin=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
                dr=1.0,
                grid_transform=np.diag((2.0, 1.0, 1.0)),
            )

    def test_q_plane_polar_projects_defect_radii_in_physical_space(self):
        grid = PlaneGridPolar(
            opts=OptsPlaneGridPolar(
                origin=(10.0, 20.0, 30.0),
                normal=(0.0, 0.0, 1.0),
                theta0_axis=(1.0, 0.0, 0.0),
                r_min=1.0,
                layers=1,
                dr=1.0,
                arc_dist=10.0,
            )
        )
        dummy_plane = types.SimpleNamespace(grid=grid)

        radii = QPlanePolar._helper_project_defect_radii(
            dummy_plane,
            np.array(
                [
                    [10.0, 20.0, 30.0],
                    [13.0, 24.0, 30.0],
                ]
            ),
        )

        self.assertTrue(np.allclose(radii, np.array([0.0, 5.0])))


if __name__ == "__main__":
    unittest.main()
