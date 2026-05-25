import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

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
from nematics3d.classes.q_plane import QPlane
from nematics3d.field import getQ


class TestQPlane(unittest.TestCase):
    def test_q_plane_maps_detected_defect_centers_into_physical_plane_coords(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 2)))
        director = np.zeros((3, 3, 2, 3), dtype=float)
        director[..., 0] = 1.0
        q_values = getQ(director, S=np.ones((3, 3, 2), dtype=float))
        field = dataset.act_add_field("Q", q_values)
        interpolator = field.act_add_interpolator()

        opts = OptsPlaneGrid(
            normal=(0.0, 0.0, 1.0),
            axis1=(1.0, 0.0, 0.0),
            origin=(10.0, 20.0, 30.0),
            spacing=2.0,
            size=4.0,
            alignment="bottom-left",
        )

        with (
            patch(
                "nematics3d.classes.q_plane.defect_detect",
                return_value=np.array([[1, 1, 0]], dtype=int),
            ),
            patch(
                "nematics3d.classes.q_plane.defect_vicinity_grid",
                return_value=np.array([[[1, 1, 0]]], dtype=int),
            ),
        ):
            plane = QPlane(interpolator=interpolator, opts=opts)

        self.assertIsNotNone(plane.calc_defect_pos_all)
        self.assertTrue(
            np.allclose(plane.calc_defect_pos_all, np.array([[12.0, 22.0, 30.0]]))
        )
        self.assertEqual(int(np.sum(plane.calc_is_near_defect)), 1)


if __name__ == "__main__":
    unittest.main()
