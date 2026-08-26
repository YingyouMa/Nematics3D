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

from nematics3d.classes.disclination_line import DisclinationLine
from nematics3d.classes.grid_field import InputGridField
from nematics3d.classes.q_field_object import InputQ
from nematics3d.analysis.disclination import defect_classify_into_lines


class TestGridOffsetNone(unittest.TestCase):
    def test_input_validators_accept_none_grid_offset(self):
        self.assertIsNone(InputGridField(grid_offset=None).grid_offset)
        self.assertIsNone(InputQ(grid_offset=None).grid_offset)

    def test_disclination_line_accepts_none_grid_offset(self):
        defect_indices = np.array(
            [
                [0.5, 0.5, 0.0],
                [1.5, 0.5, 0.0],
                [2.5, 0.5, 0.0],
            ]
        )

        line = DisclinationLine(
            defect_indices=defect_indices,
            grid_offset=None,
            is_sorted=True,
        )

        self.assertIsNone(line.raw_grid_offset)
        self.assertTrue(np.allclose(line.calc_defect_coords, defect_indices))

    def test_defect_classify_default_grid_offset_remains_none(self):
        defect_indices = np.array(
            [
                [0.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [2.0, 0.5, 0.5],
            ]
        )

        lines = defect_classify_into_lines(defect_indices)

        self.assertEqual(len(lines), 1)
        self.assertIsNone(lines[0].raw_grid_offset)
        self.assertTrue(np.allclose(lines[0].calc_defect_coords, defect_indices))


if __name__ == "__main__":
    unittest.main()
