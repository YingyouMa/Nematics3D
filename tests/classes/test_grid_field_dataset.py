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

from nematics3d.classes.grid_field import (
    GridFieldDataset,
    GridInterpolator,
    InputGridField,
)
from nematics3d.datatypes import UNSET
from nematics3d.grid import apply_linear_transform, generate_coordinate_grid
from nematics3d.general import get_box_corners


class TestGridFieldDataset(unittest.TestCase):
    def test_dataset_builds_shared_grid_cache_from_explicit_shape(self):
        input_value = InputGridField(
            shape=(2, 3, 4),
            box_periodic_flag=(True, False, True),
            grid_offset=(10, 20, 30),
            grid_transform=np.diag((2.0, 3.0, 4.0)),
        )

        dataset = GridFieldDataset(inputValue=input_value, name="dataset")

        expected_grid_index = generate_coordinate_grid((2, 3, 4), (2, 3, 4))[0]
        expected_grid = apply_linear_transform(
            expected_grid_index,
            transform=input_value.grid_transform,
            offset=input_value.grid_offset,
        )
        expected_corners_index = get_box_corners(1, 2, 3)
        expected_corners = apply_linear_transform(
            expected_corners_index,
            transform=input_value.grid_transform,
            offset=input_value.grid_offset,
        )

        self.assertEqual(tuple(dataset.raw_shape), (2, 3, 4))
        self.assertTrue(
            np.allclose(
                dataset.calc_box_size_periodic_index, np.array([2.0, np.inf, 4.0])
            )
        )
        self.assertTrue(np.allclose(dataset.calc_grid_index, expected_grid_index))
        self.assertTrue(np.allclose(dataset.calc_grid, expected_grid))
        self.assertTrue(np.allclose(dataset.calc_corners_index, expected_corners_index))
        self.assertTrue(np.allclose(dataset.calc_corners, expected_corners))
        self.assertTrue(np.allclose(dataset.calc_bounds.corners, expected_corners))

    def test_first_field_can_infer_dataset_shape_and_refresh_caches(self):
        dataset = GridFieldDataset()
        values = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)

        self.assertIs(dataset.raw_shape, UNSET)
        self.assertIsNone(dataset.raw_grid_offset)
        self.assertIs(dataset.calc_grid, UNSET)

        dataset.act_add_field("Q", values)

        self.assertEqual(tuple(dataset.raw_shape), (2, 3, 4))
        self.assertEqual(dataset.calc_grid.shape, (2, 3, 4, 3))
        self.assertTrue(np.allclose(dataset.calc_grid, dataset.calc_grid_index))
        self.assertEqual(dataset.calc_corners_index.shape, (8, 3))
        self.assertEqual(dataset.calc_corners.shape, (8, 3))

    def test_field_values_are_real_floating_lattice_fields(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        field = dataset.act_add_field("scalar", np.ones((2, 2, 2), dtype=int))

        self.assertTrue(np.issubdtype(field.raw_values.dtype, np.floating))

        with self.assertRaises(TypeError):
            dataset.act_add_field("complex", np.ones((2, 2, 2), dtype=complex))

    def test_dataset_core_grid_metadata_is_fixed_after_initialization(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))

        with self.assertRaises(AttributeError):
            dataset.shape = (3, 3, 3)
        with self.assertRaises(AttributeError):
            dataset.grid_offset = (1.0, 2.0, 3.0)

        self.assertEqual(tuple(dataset.raw_shape), (2, 2, 2))
        self.assertEqual(dataset.calc_grid.shape, (2, 2, 2, 3))

    def test_field_can_create_generic_interpolator_and_sample_world_points(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(2, 2, 2),
                box_periodic_flag=(True, False, False),
                grid_offset=(10.0, 20.0, 30.0),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        values = np.arange(8, dtype=float).reshape(2, 2, 2)
        field = dataset.act_add_field("scalar", values)

        interpolator = field.act_add_interpolator()

        self.assertIsInstance(interpolator, GridInterpolator)
        self.assertIs(field.interpolator, interpolator)
        self.assertIs(interpolator.owner, field)

        sampled = field.act_interpolate(np.array([[12.0, 23.0, 34.0]]))
        self.assertTrue(np.allclose(sampled, np.array([7.0])))

        single_point_sampled = field.act_interpolate(np.array([12.0, 23.0, 34.0]))
        self.assertTrue(np.allclose(single_point_sampled, np.array([7.0])))

        periodic_sampled = field.act_interpolate(
            np.array([[14.0, 23.0, 34.0]]),
            is_out_warning=True,
        )
        self.assertTrue(np.allclose(periodic_sampled[0], np.array([3.0])))
        self.assertEqual(len(periodic_sampled[1]), 0)


if __name__ == "__main__":
    unittest.main()
