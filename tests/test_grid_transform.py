import sys
from pathlib import Path
import types
import unittest

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.grid import GRID_TRANSFORM_IDENTITY, as_grid_transform


class TestGridTransform(unittest.TestCase):
    def test_identity_sentinel_is_preserved(self):
        self.assertIs(
            as_grid_transform(GRID_TRANSFORM_IDENTITY), GRID_TRANSFORM_IDENTITY
        )
        self.assertIs(as_grid_transform(None), None)

    def test_scaled_rotation_is_accepted(self):
        theta = np.pi / 4
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = rotation @ np.diag([2.0, 3.0, 4.0])

        result = as_grid_transform(transform)

        self.assertTrue(np.allclose(result, transform))

    def test_shear_is_rejected(self):
        transform = np.array(
            [
                [1.0, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        with self.assertRaisesRegex(ValueError, "orthogonal grid basis"):
            as_grid_transform(transform)

    def test_reflection_is_rejected(self):
        transform = np.diag([-1.0, 1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "right-handed grid basis"):
            as_grid_transform(transform)

    def test_degenerate_axis_is_rejected(self):
        transform = np.diag([1.0, 0.0, 1.0])

        with self.assertRaisesRegex(ValueError, "nonzero column vectors"):
            as_grid_transform(transform)


if __name__ == "__main__":
    unittest.main()
