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

from nematics3d.grid import (
    GRID_TRANSFORM_IDENTITY,
    GridTransform,
    apply_linear_transform,
    as_grid_offset,
    as_grid_transform,
)


class TestGridTransform(unittest.TestCase):
    def test_identity_sentinel_is_preserved(self):
        self.assertIs(
            as_grid_transform(GRID_TRANSFORM_IDENTITY), GRID_TRANSFORM_IDENTITY
        )
        self.assertIs(as_grid_transform(None), GRID_TRANSFORM_IDENTITY)

    def test_semantic_alias_is_available(self):
        self.assertIsNotNone(GridTransform)

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

    def test_transform_can_be_returned_readonly(self):
        source = np.diag([2.0, 3.0, 4.0])

        result = as_grid_transform(source, is_readonly=True)

        self.assertFalse(result.flags.writeable)
        source[0, 0] = 10.0
        self.assertEqual(result[0, 0], 2.0)

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

    def test_grid_offset_validation_and_readonly_storage(self):
        source = np.array([1.0, 2.0, 3.0])

        result = as_grid_offset(source, is_readonly=True)

        self.assertFalse(result.flags.writeable)
        source[0] = 10.0
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
        self.assertIsNone(as_grid_offset(None))

    def test_grid_offset_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            as_grid_offset([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            as_grid_offset([1.0, np.nan, 3.0])

    def test_apply_linear_transform_round_trip(self):
        points = np.arange(24.0).reshape(2, 4, 3)
        transform = np.array(
            [
                [0.0, -2.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        offset = [10.0, 20.0, 30.0]

        physical = apply_linear_transform(points, transform, offset)
        restored = apply_linear_transform(
            physical,
            transform,
            offset,
            is_inv=True,
        )

        np.testing.assert_allclose(restored, points)
        self.assertEqual(physical.shape, points.shape)

    def test_apply_linear_transform_preserves_single_point_shape(self):
        result = apply_linear_transform([1.0, 2.0, 3.0], offset=[3.0, 2.0, 1.0])

        np.testing.assert_array_equal(result, [4.0, 4.0, 4.0])
        self.assertEqual(result.shape, (3,))

    def test_apply_linear_transform_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "trailing coordinate axis"):
            apply_linear_transform(np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_linear_transform([[1.0, np.nan, 3.0]])
        with self.assertRaisesRegex(TypeError, "boolean"):
            apply_linear_transform([[1.0, 2.0, 3.0]], is_inv="yes")


if __name__ == "__main__":
    unittest.main()
