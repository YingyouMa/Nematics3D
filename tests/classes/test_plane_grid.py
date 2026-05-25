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

from nematics3d.classes.plane_grid import OptsPlaneGrid, PlaneGrid


class TestPlaneGrid(unittest.TestCase):
    def test_plane_grid_builds_physical_points_directly_from_basis(self):
        grid = PlaneGrid(
            opts=OptsPlaneGrid(
                normal=(0.0, 0.0, 1.0),
                axis1=(1.0, 0.0, 0.0),
                origin=(10.0, 20.0, 30.0),
                spacing=2.0,
                spacing_extra=3.0,
                size=4.0,
                size_extra=6.0,
                alignment="bottom-left",
            )
        )

        expected = np.array(
            [
                [[10.0, 20.0, 30.0], [10.0, 23.0, 30.0], [10.0, 26.0, 30.0]],
                [[12.0, 20.0, 30.0], [12.0, 23.0, 30.0], [12.0, 26.0, 30.0]],
                [[14.0, 20.0, 30.0], [14.0, 23.0, 30.0], [14.0, 26.0, 30.0]],
            ]
        )

        self.assertTrue(np.allclose(grid.entity_grid_all, expected))
        self.assertTrue(np.allclose(grid.calc_origin_grid0, (10.0, 20.0, 30.0)))

    def test_plane_grid_exposes_lattice_zero_point_for_center_alignment(self):
        grid = PlaneGrid(
            opts=OptsPlaneGrid(
                normal=(0.0, 0.0, 1.0),
                axis1=(1.0, 0.0, 0.0),
                origin=(0.0, 0.0, 0.0),
                spacing=2.0,
                spacing_extra=4.0,
                size=4.0,
                size_extra=8.0,
                alignment="center",
            )
        )

        self.assertTrue(np.allclose(grid.calc_origin_grid0, (-2.0, -4.0, 0.0)))
        self.assertTrue(np.allclose(grid.entity_grid_all[0, 0], grid.calc_origin_grid0))

    def test_opts_plane_grid_no_longer_accepts_legacy_transform_kwargs(self):
        with self.assertRaisesRegex(TypeError, r"grid_offset"):
            OptsPlaneGrid(
                normal=(0.0, 0.0, 1.0),
                spacing=1.0,
                size=1.0,
                grid_offset=(1.0, 2.0, 3.0),
            )

        with self.assertRaisesRegex(TypeError, r"grid_transform"):
            OptsPlaneGrid(
                normal=(0.0, 0.0, 1.0),
                spacing=1.0,
                size=1.0,
                grid_transform=np.diag((2.0, 1.0, 1.0)),
            )


if __name__ == "__main__":
    unittest.main()
