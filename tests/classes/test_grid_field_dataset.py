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
    SpatialDerivativeInfo,
    SpatialDerivativeResult,
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

    def test_dataset_bounds_opts_are_protected_but_copies_are_editable(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        bounds = dataset.calc_bounds
        origin_before = bounds.opts.origin.copy()

        bounds.act_commit(origin=(10.0, 20.0, 30.0))
        bounds.opts.origin = (10.0, 20.0, 30.0)

        self.assertTrue(np.allclose(bounds.opts.origin, origin_before))
        self.assertTrue(set(type(bounds.opts).__attrs__) <= bounds.attrs_protected)

        bounds_copy = bounds.act_copy()
        bounds_copy.act_commit(origin=(10.0, 20.0, 30.0))

        self.assertTrue(np.allclose(bounds_copy.opts.origin, (10.0, 20.0, 30.0)))

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

    def test_field_values_are_fixed_and_replace_creates_new_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        field = dataset.act_add_field("scalar", np.zeros((2, 2, 2)))

        with self.assertRaises(AttributeError):
            field.values = np.ones((2, 2, 2))

        field_new = dataset.act_add_field(
            "scalar",
            np.ones((2, 2, 2)),
            is_replace=True,
        )

        self.assertIsNot(field_new, field)
        self.assertIs(dataset["scalar"], field_new)
        self.assertTrue(np.allclose(field_new.raw_values, 1.0))

    def test_field_can_store_optional_user_info(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        values = np.zeros((2, 2, 2), dtype=float)
        info = {"source": "synthetic", "note": "unit test"}

        field = dataset.act_add_field("scalar", values, info=info)

        self.assertIs(field.raw_info, info)
        self.assertIs(dataset["scalar"].raw_info, info)

    def test_field_info_defaults_to_none(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        field = dataset.act_add_field("scalar", np.zeros((2, 2, 2), dtype=float))

        self.assertIsNone(field.raw_info)

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

    def test_gradient_can_return_spatial_derivative_result_metadata(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(3, 4, 5),
                box_periodic_flag=(True, False, True),
                grid_offset=(1.0, 2.0, 3.0),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + j + k
        dataset.act_add_field("scalar", values)

        result = dataset.act_gradient("scalar", is_result=True)

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertIsInstance(result.raw_info, SpatialDerivativeInfo)
        self.assertEqual(result.raw_info.operator, "gradient")
        self.assertEqual(result.raw_info.source, "scalar")
        self.assertEqual(result.raw_info.source_shape, (3, 4, 5))
        self.assertEqual(result.raw_info.coord, "physical")
        self.assertIsNone(result.raw_info.derivative_axis)
        self.assertEqual(result.raw_info.input_component_shape, ())
        self.assertEqual(result.raw_info.output_shape, (3, 4, 5, 3))
        self.assertEqual(result.raw_info.box_periodic_flag, (True, False, True))
        self.assertEqual(result.raw_info.edge_order, 1)
        self.assertTrue(
            np.allclose(result.raw_info.grid_offset, np.array([1.0, 2.0, 3.0]))
        )
        self.assertIn("raw_values", result)
        self.assertTrue(np.allclose(result["raw_values"], result.raw_values))
        self.assertIn("operator", result.raw_info)
        self.assertNotIn("raw_values", result.raw_info)

    def test_spatial_derivative_result_can_be_registered_as_field_info(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + j + k
        dataset.act_add_field("scalar", values)
        result = dataset.act_gradient("scalar", is_result=True)

        field = dataset.act_add_field(
            "grad_scalar",
            result.raw_values,
            info=result.raw_info,
        )

        self.assertIs(field.raw_info, result.raw_info)
        self.assertEqual(field.raw_info.operator, "gradient")
        self.assertEqual(field.raw_info.source, "scalar")
        self.assertTrue(np.allclose(field.raw_values, result.raw_values))

    def test_spatial_derivative_result_can_be_registered_with_convenience_method(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + j + k
        dataset.act_add_field("scalar", values)
        result = dataset.act_gradient("scalar", is_result=True)

        field = dataset.act_add_result_field("grad_scalar", result)

        self.assertIs(field, dataset["grad_scalar"])
        self.assertIs(field.raw_info, result.raw_info)
        self.assertIsNot(field.raw_info, result)
        self.assertTrue(np.allclose(field.raw_values, result.raw_values))

    def test_result_field_registration_supports_replace(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        dataset.act_add_field("scalar", i + j + k)
        result_first = dataset.act_gradient("scalar", coord="index", is_result=True)
        dataset.act_add_result_field("grad_scalar", result_first)

        result_second = dataset.act_derivative(
            "scalar",
            direction="x",
            coord="index",
            is_result=True,
        )
        field = dataset.act_add_result_field(
            "grad_scalar",
            result_second,
            is_replace=True,
        )

        self.assertEqual(field.raw_info.operator, "derivative")
        self.assertEqual(field.raw_info.derivative_axis, 0)
        self.assertEqual(field.raw_values.shape, (3, 4, 5))

    def test_result_field_registration_rejects_non_result(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))

        with self.assertRaises(TypeError):
            dataset.act_add_result_field("bad", np.zeros((3, 3, 3), dtype=float))

    def test_derivative_selects_one_gradient_direction(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + 2.0 * j + 3.0 * k
        dataset.act_add_field("scalar", values)

        d_dy = dataset.act_derivative("scalar", direction="y", coord="index")
        d_dx = dataset.act_derivative("scalar", direction="X", coord="index")
        d_dz = dataset.act_derivative("scalar", direction=2, coord="index")

        self.assertTrue(np.allclose(d_dx, 1.0))
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

    def test_derivative_result_records_selected_axis_and_temporary_source(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + 2.0 * j + 3.0 * k

        result = dataset.act_derivative(
            values,
            direction="z",
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "derivative")
        self.assertIsNone(result.raw_info.source)
        self.assertEqual(result.raw_info.coord, "index")
        self.assertEqual(result.raw_info.derivative_axis, 2)
        self.assertEqual(result.raw_info.source_shape, (3, 4, 5))
        self.assertEqual(result.raw_info.output_shape, (3, 4, 5))
        self.assertTrue(np.allclose(result.raw_values, 3.0))

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

    def test_divergence_can_return_spatial_derivative_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        i, j, k = np.indices((3, 3, 3), dtype=float)
        values = np.stack((i, j, k), axis=-1)
        dataset.act_add_field("vector", values)

        result = dataset.act_divergence("vector", coord="index", is_result=True)

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "divergence")
        self.assertEqual(result.raw_info.source, "vector")
        self.assertEqual(result.raw_info.source_shape, (3, 3, 3, 3))
        self.assertEqual(result.raw_info.input_component_shape, (3,))
        self.assertEqual(result.raw_info.output_shape, (3, 3, 3))
        self.assertTrue(np.allclose(result.raw_values, 3.0))

    def test_curl_returns_vector_field_rotation(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = -j
        values[..., 1] = i
        dataset.act_add_field("vector", values)

        curl = dataset.act_curl("vector", coord="index")

        self.assertEqual(curl.shape, (3, 4, 5, 3))
        self.assertTrue(np.allclose(curl[..., 0], 0.0))
        self.assertTrue(np.allclose(curl[..., 1], 0.0))
        self.assertTrue(np.allclose(curl[..., 2], 2.0))

    def test_curl_uses_physical_coordinate_gradient(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(3, 4, 5),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((3, 4, 5), dtype=float)
        x = 2.0 * i
        y = 3.0 * j
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = -y
        values[..., 1] = x

        curl = dataset.act_curl(values)

        self.assertTrue(np.allclose(curl[..., 0], 0.0))
        self.assertTrue(np.allclose(curl[..., 1], 0.0))
        self.assertTrue(np.allclose(curl[..., 2], 2.0))

    def test_curl_accepts_temporary_vector_arrays(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.stack((-j, i, np.zeros_like(k)), axis=-1)

        curl = dataset.act_curl(values, coord="index")

        self.assertTrue(np.allclose(curl[..., 2], 2.0))

    def test_curl_rejects_non_vector_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        dataset.act_add_field("scalar", np.zeros((3, 3, 3), dtype=float))

        with self.assertRaises(ValueError):
            dataset.act_curl("scalar")

    def test_curl_rejects_tensor_field_with_specific_message(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        dataset.act_add_field("tensor", np.zeros((3, 3, 3, 3, 3), dtype=float))

        with self.assertRaisesRegex(ValueError, "tensor-specific curl"):
            dataset.act_curl("tensor")

    def test_curl_can_return_spatial_derivative_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.stack((-j, i, np.zeros_like(k)), axis=-1)
        dataset.act_add_field("vector", values)

        result = dataset.act_curl("vector", coord="index", is_result=True)

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "curl")
        self.assertEqual(result.raw_info.source, "vector")
        self.assertEqual(result.raw_info.source_shape, (3, 4, 5, 3))
        self.assertEqual(result.raw_info.input_component_shape, (3,))
        self.assertEqual(result.raw_info.output_shape, (3, 4, 5, 3))
        self.assertTrue(np.allclose(result.raw_values[..., 2], 2.0))

    def test_tensor_curl_applies_curl_along_default_last_axis(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 2, 3), dtype=float)
        values[..., 0, 0] = -j
        values[..., 0, 1] = i
        values[..., 1, 0] = -2.0 * j
        values[..., 1, 1] = 2.0 * i

        curl = dataset.act_tensor_curl(values, coord="index")

        self.assertEqual(curl.shape, values.shape)
        self.assertTrue(np.allclose(curl[..., 0, 0], 0.0))
        self.assertTrue(np.allclose(curl[..., 0, 1], 0.0))
        self.assertTrue(np.allclose(curl[..., 0, 2], 2.0))
        self.assertTrue(np.allclose(curl[..., 1, 2], 4.0))

    def test_tensor_curl_can_use_non_last_component_axis(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3, 2), dtype=float)
        values[..., 0, 0] = -j
        values[..., 1, 0] = i
        values[..., 0, 1] = -2.0 * j
        values[..., 1, 1] = 2.0 * i

        curl = dataset.act_tensor_curl(values, vector_axis=-2, coord="index")

        self.assertEqual(curl.shape, values.shape)
        self.assertTrue(np.allclose(curl[..., 0, 0], 0.0))
        self.assertTrue(np.allclose(curl[..., 1, 0], 0.0))
        self.assertTrue(np.allclose(curl[..., 2, 0], 2.0))
        self.assertTrue(np.allclose(curl[..., 2, 1], 4.0))

    def test_tensor_curl_rejects_non_vector_component_axis(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        values = np.zeros((3, 3, 3, 2, 3), dtype=float)

        with self.assertRaises(ValueError):
            dataset.act_tensor_curl(values, vector_axis=-2)

    def test_tensor_curl_result_records_component_axis(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3, 2), dtype=float)
        values[..., 0, 0] = -j
        values[..., 1, 0] = i
        dataset.act_add_field("tensor", values)

        result = dataset.act_tensor_curl(
            "tensor",
            vector_axis=-2,
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "tensor_curl")
        self.assertEqual(result.raw_info.source, "tensor")
        self.assertEqual(result.raw_info.source_shape, (3, 4, 5, 3, 2))
        self.assertEqual(result.raw_info.output_shape, (3, 4, 5, 3, 2))
        self.assertEqual(result.raw_info.component_axis, 3)
        self.assertTrue(np.allclose(result.raw_values[..., 2, 0], 2.0))

    def test_symmetric_and_antisymmetric_gradient_split_vector_gradient(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = j
        values[..., 1] = 0.0 * i
        dataset.act_add_field("velocity", values)

        symmetric = dataset.act_symmetric_gradient("velocity", coord="index")
        antisymmetric = dataset.act_antisymmetric_gradient(
            "velocity",
            coord="index",
        )

        self.assertEqual(symmetric.shape, (3, 4, 5, 3, 3))
        self.assertEqual(antisymmetric.shape, (3, 4, 5, 3, 3))
        self.assertTrue(np.allclose(symmetric[..., 0, 1], 0.5))
        self.assertTrue(np.allclose(symmetric[..., 1, 0], 0.5))
        self.assertTrue(np.allclose(antisymmetric[..., 0, 1], 0.5))
        self.assertTrue(np.allclose(antisymmetric[..., 1, 0], -0.5))
        self.assertTrue(
            np.allclose(
                symmetric + antisymmetric,
                dataset.act_gradient("velocity", coord="index"),
            )
        )

    def test_symmetric_gradient_uses_physical_coordinate_derivatives(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(3, 4, 5),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((3, 4, 5), dtype=float)
        y = 3.0 * j
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = y

        symmetric = dataset.act_symmetric_gradient(values)
        antisymmetric = dataset.act_antisymmetric_gradient(values)

        self.assertTrue(np.allclose(symmetric[..., 0, 1], 0.5))
        self.assertTrue(np.allclose(symmetric[..., 1, 0], 0.5))
        self.assertTrue(np.allclose(antisymmetric[..., 0, 1], 0.5))
        self.assertTrue(np.allclose(antisymmetric[..., 1, 0], -0.5))

    def test_symmetric_gradient_rejects_non_vector_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        dataset.act_add_field("scalar", np.zeros((3, 3, 3), dtype=float))

        with self.assertRaises(ValueError):
            dataset.act_symmetric_gradient("scalar")

    def test_antisymmetric_gradient_can_return_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = j
        dataset.act_add_field("velocity", values)

        result = dataset.act_antisymmetric_gradient(
            "velocity",
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "antisymmetric_gradient")
        self.assertEqual(result.raw_info.source, "velocity")
        self.assertEqual(result.raw_info.source_shape, (3, 4, 5, 3))
        self.assertEqual(result.raw_info.output_shape, (3, 4, 5, 3, 3))
        self.assertTrue(np.allclose(result.raw_values[..., 0, 1], 0.5))

    def test_laplacian_returns_scalar_second_derivative_sum(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = i**2 + j**2 + k**2
        dataset.act_add_field("scalar", values)

        laplacian = dataset.act_laplacian("scalar", coord="index")

        self.assertEqual(laplacian.shape, (7, 7, 7))
        self.assertTrue(np.allclose(laplacian[2:-2, 2:-2, 2:-2], 6.0))

    def test_laplacian_uses_physical_coordinate_derivatives(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(7, 7, 7),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = (2.0 * i) ** 2 + (3.0 * j) ** 2 + (4.0 * k) ** 2

        laplacian = dataset.act_laplacian(values)

        self.assertTrue(np.allclose(laplacian[2:-2, 2:-2, 2:-2], 6.0))

    def test_laplacian_accepts_temporary_scalar_arrays(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = i**2 + j**2 + k**2

        laplacian = dataset.act_laplacian(values, coord="index")

        self.assertTrue(np.allclose(laplacian[2:-2, 2:-2, 2:-2], 6.0))

    def test_laplacian_rejects_non_scalar_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        dataset.act_add_field("vector", np.zeros((3, 3, 3, 3), dtype=float))

        with self.assertRaisesRegex(ValueError, "act_componentwise_laplacian"):
            dataset.act_laplacian("vector")

    def test_laplacian_can_return_spatial_derivative_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = i**2 + j**2 + k**2
        dataset.act_add_field("scalar", values)

        result = dataset.act_laplacian("scalar", coord="index", is_result=True)

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "laplacian")
        self.assertEqual(result.raw_info.source, "scalar")
        self.assertEqual(result.raw_info.coord, "index")
        self.assertEqual(result.raw_info.source_shape, (7, 7, 7))
        self.assertEqual(result.raw_info.output_shape, (7, 7, 7))
        self.assertIsNone(result.raw_info.derivative_axis)
        self.assertTrue(np.allclose(result.raw_values[2:-2, 2:-2, 2:-2], 6.0))

    def test_componentwise_laplacian_preserves_component_axes(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = np.zeros((7, 7, 7, 2), dtype=float)
        values[..., 0] = i**2 + j**2 + k**2
        values[..., 1] = 2.0 * i**2 + 3.0 * j**2 + 4.0 * k**2
        dataset.act_add_field("vector_like", values)

        laplacian = dataset.act_componentwise_laplacian(
            "vector_like",
            coord="index",
        )

        self.assertEqual(laplacian.shape, (7, 7, 7, 2))
        self.assertTrue(np.allclose(laplacian[2:-2, 2:-2, 2:-2, 0], 6.0))
        self.assertTrue(np.allclose(laplacian[2:-2, 2:-2, 2:-2, 1], 18.0))

    def test_componentwise_laplacian_matches_scalar_laplacian_for_scalar_input(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = i**2 + j**2 + k**2

        scalar_laplacian = dataset.act_laplacian(values, coord="index")
        componentwise_laplacian = dataset.act_componentwise_laplacian(
            values,
            coord="index",
        )

        self.assertTrue(np.allclose(componentwise_laplacian, scalar_laplacian))

    def test_componentwise_laplacian_uses_physical_coordinate_derivatives(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(7, 7, 7),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = np.zeros((7, 7, 7, 2), dtype=float)
        values[..., 0] = (2.0 * i) ** 2 + (3.0 * j) ** 2 + (4.0 * k) ** 2
        values[..., 1] = 2.0 * (2.0 * i) ** 2 + 3.0 * (3.0 * j) ** 2

        laplacian = dataset.act_componentwise_laplacian(values)

        self.assertTrue(np.allclose(laplacian[2:-2, 2:-2, 2:-2, 0], 6.0))
        self.assertTrue(np.allclose(laplacian[2:-2, 2:-2, 2:-2, 1], 10.0))

    def test_componentwise_laplacian_can_return_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = np.stack((i**2 + j**2 + k**2, i + j + k), axis=-1)
        dataset.act_add_field("field", values)

        result = dataset.act_componentwise_laplacian(
            "field",
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "componentwise_laplacian")
        self.assertEqual(result.raw_info.source, "field")
        self.assertEqual(result.raw_info.source_shape, (7, 7, 7, 2))
        self.assertEqual(result.raw_info.input_component_shape, (2,))
        self.assertEqual(result.raw_info.output_shape, (7, 7, 7, 2))
        self.assertTrue(np.allclose(result.raw_values[2:-2, 2:-2, 2:-2, 0], 6.0))
        self.assertTrue(np.allclose(result.raw_values[2:-2, 2:-2, 2:-2, 1], 0.0))


if __name__ == "__main__":
    unittest.main()
