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
        self.assertTrue(np.allclose(dataset.calc_corners.corners, expected_corners))

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

    def test_gradient_returns_index_derivatives_with_nonperiodic_boundaries(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(5, 4, 3)))
        i, j, k = np.indices((5, 4, 3), dtype=float)
        values = i**2 + 2.0 * j + 3.0 * k
        dataset.act_add_field("scalar", values)

        grad = dataset.act_gradient("scalar", coord="index")

        expected_di = np.zeros((5, 4, 3), dtype=float)
        expected_di[0] = 1.0
        expected_di[1] = 2.0
        expected_di[2] = 4.0
        expected_di[3] = 6.0
        expected_di[4] = 7.0

        self.assertEqual(grad.shape, (5, 4, 3, 3))
        self.assertTrue(np.allclose(grad[..., 0], expected_di))
        self.assertTrue(np.allclose(grad[..., 1], 2.0))
        self.assertTrue(np.allclose(grad[..., 2], 3.0))

    def test_gradient_uses_periodic_stencil_on_periodic_axes(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(4, 2, 2), box_periodic_flag=(True, False, False)
            )
        )
        values = np.broadcast_to(
            np.arange(4, dtype=float).reshape(4, 1, 1),
            (4, 2, 2),
        )
        dataset.act_add_field("scalar", values)

        grad = dataset.act_gradient("scalar", coord="index")

        self.assertTrue(np.allclose(grad[:, 0, 0, 0], np.array([-1.0, 1.0, 1.0, -1.0])))
        self.assertTrue(np.allclose(grad[..., 1], 0.0))
        self.assertTrue(np.allclose(grad[..., 2], 0.0))

    def test_gradient_converts_derivative_axis_to_physical_coordinates(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(3, 4, 5),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = 4.0 * i + 6.0 * j + 8.0 * k
        dataset.act_add_field("scalar", values)

        grad = dataset.act_gradient("scalar")

        self.assertTrue(np.allclose(grad[..., 0], 2.0))
        self.assertTrue(np.allclose(grad[..., 1], 2.0))
        self.assertTrue(np.allclose(grad[..., 2], 2.0))

    def test_gradient_accepts_temporary_arrays_and_preserves_component_axes(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        values = np.zeros((3, 4, 5, 2), dtype=float)
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values[..., 0] = i
        values[..., 1] = j + k

        grad = dataset.act_gradient(values, coord="index")

        self.assertEqual(grad.shape, (3, 4, 5, 2, 3))
        self.assertTrue(np.allclose(grad[..., 0, 0], 1.0))
        self.assertTrue(np.allclose(grad[..., 0, 1], 0.0))
        self.assertTrue(np.allclose(grad[..., 0, 2], 0.0))
        self.assertTrue(np.allclose(grad[..., 1, 0], 0.0))
        self.assertTrue(np.allclose(grad[..., 1, 1], 1.0))
        self.assertTrue(np.allclose(grad[..., 1, 2], 1.0))

    def test_derivative_selects_one_gradient_direction(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + 2.0 * j + 3.0 * k
        dataset.act_add_field("scalar", values)

        d_dy = dataset.act_derivative("scalar", direction="y", coord="index")
        d_dz = dataset.act_derivative("scalar", direction=2, coord="index")

        self.assertTrue(np.allclose(d_dy, 2.0))
        self.assertTrue(np.allclose(d_dz, 3.0))

    def test_derivative_accepts_temporary_arrays_for_chained_expressions(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(5, 4, 3)))
        i, j, k = np.indices((5, 4, 3), dtype=float)
        A = i**2
        B = 2.0 + j * 0.0 + k * 0.0
        dataset.act_add_field("A", A)
        dataset.act_add_field("B", B)

        dA_dx = dataset.act_derivative("A", direction="x", coord="index")
        result = dataset.act_derivative(
            dA_dx * dataset["B"].raw_values,
            direction="x",
            coord="index",
        )

        expected = np.zeros((5, 4, 3), dtype=float)
        expected[0] = 2.0
        expected[1] = 3.0
        expected[2] = 4.0
        expected[3] = 3.0
        expected[4] = 2.0
        self.assertTrue(np.allclose(result, expected))

    def test_derivative_rejects_invalid_direction(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        dataset.act_add_field("scalar", np.zeros((3, 3, 3), dtype=float))

        with self.assertRaises(ValueError):
            dataset.act_derivative("scalar", direction="theta")

    def test_divergence_contracts_vector_component_with_derivative_axis(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = i
        values[..., 1] = 2.0 * j
        values[..., 2] = 3.0 * k
        dataset.act_add_field("vector", values)

        div = dataset.act_divergence("vector", coord="index")

        self.assertEqual(div.shape, (3, 4, 5))
        self.assertTrue(np.allclose(div, 6.0))

    def test_divergence_uses_physical_coordinate_gradient(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(3, 4, 5),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = 4.0 * i
        values[..., 1] = 6.0 * j
        values[..., 2] = 8.0 * k

        div = dataset.act_divergence(values)

        self.assertTrue(np.allclose(div, 6.0))

    def test_divergence_accepts_temporary_vector_arrays(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        i, j, k = np.indices((3, 3, 3), dtype=float)
        values = np.stack((i, j, k), axis=-1)

        div = dataset.act_divergence(values, coord="index")

        self.assertTrue(np.allclose(div, 3.0))

    def test_divergence_rejects_non_vector_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        dataset.act_add_field("scalar", np.zeros((3, 3, 3), dtype=float))

        with self.assertRaises(ValueError):
            dataset.act_divergence("scalar")


if __name__ == "__main__":
    unittest.main()
