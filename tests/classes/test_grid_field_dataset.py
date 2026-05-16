import sys
from pathlib import Path
import tempfile
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
    GaussianSmoothResult,
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

    def test_dataset_grid_transform_parameters_are_readonly_snapshots(self):
        grid_offset = np.array([1.0, 2.0, 3.0])
        grid_transform = np.diag((2.0, 3.0, 4.0))
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(2, 2, 2),
                grid_offset=grid_offset,
                grid_transform=grid_transform,
            )
        )

        grid_offset[:] = 99.0
        grid_transform[0, 0] = 99.0

        self.assertTrue(np.allclose(dataset.raw_grid_offset, (1.0, 2.0, 3.0)))
        self.assertTrue(
            np.allclose(dataset.raw_grid_transform, np.diag((2.0, 3.0, 4.0)))
        )
        with self.assertRaises(ValueError):
            dataset.raw_grid_offset[0] = 0.0
        with self.assertRaises(ValueError):
            dataset.raw_grid_transform[0, 0] = 0.0

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

    def test_gradient_can_return_norm_without_derivative_axis(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + 2.0 * j + 2.0 * k

        grad_norm = dataset.act_gradient(values, coord="index", is_norm=True)

        self.assertEqual(grad_norm.shape, (3, 4, 5))
        self.assertTrue(np.allclose(grad_norm, 3.0))

    def test_gradient_norm_preserves_component_axes(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 2), dtype=float)
        values[..., 0] = i + 2.0 * j
        values[..., 1] = 2.0 * j + 3.0 * k

        grad = dataset.act_gradient(values, coord="index")
        grad_norm = dataset.act_gradient(values, coord="index", is_norm=True)

        self.assertEqual(grad_norm.shape, (3, 4, 5, 2))
        self.assertTrue(np.allclose(grad_norm, np.linalg.norm(grad, axis=-1)))

    def test_gradient_norm_can_return_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        dataset.act_add_field("scalar", i + j + k)

        result = dataset.act_gradient(
            "scalar", coord="index", is_norm=True, is_result=True
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "gradient_norm")
        self.assertEqual(result.raw_info.source, "scalar")
        self.assertEqual(result.raw_info.output_shape, (3, 4, 5))
        self.assertTrue(np.allclose(result.raw_values, np.sqrt(3.0)))

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

    def test_spatial_derivative_info_grid_transform_is_readonly_snapshot(self):
        grid_transform = np.diag((2.0, 3.0, 4.0))
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(3, 4, 5),
                grid_transform=grid_transform,
            )
        )
        values = np.zeros((3, 4, 5), dtype=float)

        result = dataset.act_gradient(values, is_result=True)
        grid_transform[0, 0] = 99.0

        self.assertTrue(
            np.allclose(result.raw_info.grid_transform, np.diag((2.0, 3.0, 4.0)))
        )
        with self.assertRaises(ValueError):
            result.raw_info.grid_transform[0, 0] = 0.0

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

    def test_spatial_derivative_result_can_save_release_and_read_with_context(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        dataset.act_add_field("scalar", i + j + k)
        result = dataset.act_gradient("scalar", coord="index", is_result=True)
        expected = result.raw_values.copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = result.act_save_values(
                Path(tmp_dir) / "grad_scalar",
                is_release=True,
            )

            self.assertIsNone(saved.raw_values)
            self.assertTrue(saved.raw_path.endswith(".npy"))
            self.assertTrue(Path(saved.raw_path).exists())

            with saved.act_with_values() as loaded:
                self.assertTrue(np.allclose(loaded, expected))

            loaded_result = saved.act_load_values()
            self.assertTrue(np.allclose(loaded_result.raw_values, expected))

    def test_result_field_registration_loads_released_saved_values(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        dataset.act_add_field("scalar", i + j + k)
        result = dataset.act_gradient("scalar", coord="index", is_result=True)
        expected = result.raw_values.copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = result.act_save_values(
                Path(tmp_dir) / "grad_scalar.npy",
                is_release=True,
            )
            field = dataset.act_add_result_field("grad_scalar", saved)

        self.assertTrue(np.allclose(field.raw_values, expected))
        self.assertIs(field.raw_info, result.raw_info)

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

    def test_derivative_matches_gradient_slice_for_rotated_physical_grid(self):
        grid_transform = np.array(
            [
                [0.0, -2.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(5, 5, 5),
                grid_transform=grid_transform,
            )
        )
        i, j, k = np.indices((5, 5, 5), dtype=float)
        x = 3.0 * j
        y = -2.0 * i
        z = 4.0 * k
        values = 7.0 * x + 11.0 * y + 13.0 * z

        grad = dataset.act_gradient(values)
        d_dx = dataset.act_derivative(values, direction="x")
        d_dy = dataset.act_derivative(values, direction="y")
        d_dz = dataset.act_derivative(values, direction="z")

        self.assertTrue(np.allclose(d_dx, grad[..., 0]))
        self.assertTrue(np.allclose(d_dy, grad[..., 1]))
        self.assertTrue(np.allclose(d_dz, grad[..., 2]))
        self.assertTrue(np.allclose(grad[1:-1, 1:-1, 1:-1, 0], 7.0))
        self.assertTrue(np.allclose(grad[1:-1, 1:-1, 1:-1, 1], 11.0))
        self.assertTrue(np.allclose(grad[1:-1, 1:-1, 1:-1, 2], 13.0))

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

    def test_second_derivative_returns_repeated_direction_derivative(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = 2.0 * i**2 + 3.0 * j**2 + 4.0 * k**2

        d2_dx2 = dataset.act_second_derivative(values, direction="x", coord="index")
        d2_dy2 = dataset.act_second_derivative(values, direction="y", coord="index")
        d2_dz2 = dataset.act_second_derivative(values, direction="z", coord="index")

        self.assertTrue(np.allclose(d2_dx2, 4.0))
        self.assertTrue(np.allclose(d2_dy2, 6.0))
        self.assertTrue(np.allclose(d2_dz2, 8.0))

    def test_second_derivative_can_return_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        i, j, k = np.indices((7, 7, 7), dtype=float)
        values = i**2 + j + k
        dataset.act_add_field("scalar", values)

        result = dataset.act_second_derivative(
            "scalar",
            direction="x",
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "second_derivative")
        self.assertEqual(result.raw_info.source, "scalar")
        self.assertEqual(result.raw_info.derivative_axis, 0)
        self.assertTrue(np.allclose(result.raw_values, 2.0))

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

    def test_tensor_divergence_contracts_selected_component_axis(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3, 2), dtype=float)
        values[..., 0, 0] = i
        values[..., 1, 0] = 2.0 * j
        values[..., 2, 0] = 3.0 * k
        values[..., 0, 1] = 4.0 * i
        values[..., 1, 1] = 5.0 * j
        values[..., 2, 1] = 6.0 * k

        div = dataset.act_tensor_divergence(
            values,
            vector_axis=-2,
            coord="index",
        )

        self.assertEqual(div.shape, (3, 4, 5, 2))
        self.assertTrue(np.allclose(div[..., 0], 6.0))
        self.assertTrue(np.allclose(div[..., 1], 15.0))

    def test_tensor_divergence_can_return_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        i, j, k = np.indices((3, 3, 3), dtype=float)
        values = np.stack((i, j, k), axis=-1)
        dataset.act_add_field("vector", values)

        result = dataset.act_tensor_divergence(
            "vector",
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "tensor_divergence")
        self.assertEqual(result.raw_info.source, "vector")
        self.assertEqual(result.raw_info.component_axis, 3)
        self.assertEqual(result.raw_info.output_shape, (3, 3, 3))
        self.assertTrue(np.allclose(result.raw_values, 3.0))

    def test_directional_derivative_projects_gradient_onto_direction(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = i + 2.0 * j + 3.0 * k

        directional = dataset.act_directional_derivative(
            values,
            direction=(1.0, 1.0, 0.0),
            coord="index",
        )

        self.assertTrue(np.allclose(directional, 3.0))

    def test_directional_derivative_preserves_component_axes(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 2), dtype=float)
        values[..., 0] = i + j
        values[..., 1] = 2.0 * j + 3.0 * k

        directional = dataset.act_directional_derivative(
            values,
            direction=(0.0, 1.0, 0.0),
            coord="index",
        )

        self.assertEqual(directional.shape, (3, 4, 5, 2))
        self.assertTrue(np.allclose(directional[..., 0], 1.0))
        self.assertTrue(np.allclose(directional[..., 1], 2.0))

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

    def test_strain_rate_and_vorticity_tensor_split_velocity_gradient_once(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = j
        values[..., 1] = 2.0 * i
        values[..., 2] = k
        dataset.act_add_field("velocity", values)

        strain_rate, vorticity_tensor = dataset.act_strain_rate_and_vorticity_tensor(
            "velocity",
            coord="index",
        )

        self.assertEqual(strain_rate.shape, (3, 4, 5, 3, 3))
        self.assertEqual(vorticity_tensor.shape, (3, 4, 5, 3, 3))
        self.assertTrue(
            np.allclose(
                strain_rate + vorticity_tensor,
                dataset.act_gradient("velocity", coord="index"),
            )
        )
        self.assertTrue(np.allclose(strain_rate[..., 0, 1], 1.5))
        self.assertTrue(np.allclose(strain_rate[..., 1, 0], 1.5))
        self.assertTrue(np.allclose(vorticity_tensor[..., 0, 1], -0.5))
        self.assertTrue(np.allclose(vorticity_tensor[..., 1, 0], 0.5))

    def test_strain_rate_and_vorticity_tensor_use_physical_coordinates(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(3, 4, 5),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((3, 4, 5), dtype=float)
        x = 2.0 * i
        y = 3.0 * j
        z = 4.0 * k
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = y
        values[..., 1] = 2.0 * x
        values[..., 2] = z

        strain_rate, vorticity_tensor = dataset.act_strain_rate_and_vorticity_tensor(
            values
        )

        self.assertTrue(np.allclose(strain_rate[..., 0, 1], 1.5))
        self.assertTrue(np.allclose(strain_rate[..., 1, 0], 1.5))
        self.assertTrue(np.allclose(vorticity_tensor[..., 0, 1], -0.5))
        self.assertTrue(np.allclose(vorticity_tensor[..., 1, 0], 0.5))

    def test_strain_rate_and_vorticity_tensor_can_return_result_metadata(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = j
        dataset.act_add_field("velocity", values)

        strain_result, vorticity_result = dataset.act_strain_rate_and_vorticity_tensor(
            "velocity",
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(strain_result, SpatialDerivativeResult)
        self.assertIsInstance(vorticity_result, SpatialDerivativeResult)
        self.assertEqual(strain_result.raw_info.operator, "strain_rate")
        self.assertEqual(vorticity_result.raw_info.operator, "vorticity_tensor")
        self.assertEqual(strain_result.raw_info.source, "velocity")
        self.assertEqual(vorticity_result.raw_info.source, "velocity")
        self.assertEqual(strain_result.raw_info.output_shape, (3, 4, 5, 3, 3))
        self.assertEqual(vorticity_result.raw_info.output_shape, (3, 4, 5, 3, 3))

    def test_strain_rate_and_vorticity_tensor_can_select_one_output(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = j
        values[..., 1] = 2.0 * i

        strain_rate = dataset.act_strain_rate_and_vorticity_tensor(
            values,
            which="strain_rate",
            coord="index",
        )
        vorticity_tensor = dataset.act_strain_rate_and_vorticity_tensor(
            values,
            which="vorticity_tensor",
            coord="index",
        )

        self.assertEqual(strain_rate.shape, (3, 4, 5, 3, 3))
        self.assertEqual(vorticity_tensor.shape, (3, 4, 5, 3, 3))
        self.assertTrue(np.allclose(strain_rate[..., 0, 1], 1.5))
        self.assertTrue(np.allclose(vorticity_tensor[..., 0, 1], -0.5))

    def test_strain_rate_and_vorticity_tensor_can_select_one_result(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 4, 5)))
        i, j, k = np.indices((3, 4, 5), dtype=float)
        values = np.zeros((3, 4, 5, 3), dtype=float)
        values[..., 0] = j
        dataset.act_add_field("velocity", values)

        result = dataset.act_strain_rate_and_vorticity_tensor(
            "velocity",
            which="vorticity_tensor",
            coord="index",
            is_result=True,
        )

        self.assertIsInstance(result, SpatialDerivativeResult)
        self.assertEqual(result.raw_info.operator, "vorticity_tensor")
        self.assertEqual(result.raw_info.source, "velocity")
        self.assertEqual(result.raw_info.output_shape, (3, 4, 5, 3, 3))

    def test_strain_rate_and_vorticity_tensor_rejects_non_vector_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        dataset.act_add_field("scalar", np.zeros((3, 3, 3), dtype=float))

        with self.assertRaises(ValueError):
            dataset.act_strain_rate_and_vorticity_tensor("scalar")

    def test_strain_rate_and_vorticity_tensor_rejects_invalid_which(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 3)))
        values = np.zeros((3, 3, 3, 3), dtype=float)

        with self.assertRaises(ValueError):
            dataset.act_strain_rate_and_vorticity_tensor(values, which="spin")

    def test_elastic_deformation_returns_full_grid_outputs_for_q5_input(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(4, 5, 6)))
        i, j, k = np.indices((4, 5, 6), dtype=float)
        q_values = np.zeros((4, 5, 6, 5), dtype=float)
        q_values[..., 0] = i
        q_values[..., 1] = j
        q_values[..., 2] = k
        q_values[..., 3] = i - 2.0 * j
        q_values[..., 4] = 3.0 * k
        dataset.act_add_field("Q", q_values)

        result = dataset.act_elastic_deformation("Q", coord="index")

        q_tensor = np.zeros((4, 5, 6, 3, 3), dtype=float)
        q_tensor[..., 0, 0] = q_values[..., 0]
        q_tensor[..., 0, 1] = q_values[..., 1]
        q_tensor[..., 0, 2] = q_values[..., 2]
        q_tensor[..., 1, 0] = q_values[..., 1]
        q_tensor[..., 1, 1] = q_values[..., 3]
        q_tensor[..., 1, 2] = q_values[..., 4]
        q_tensor[..., 2, 0] = q_values[..., 2]
        q_tensor[..., 2, 1] = q_values[..., 4]
        q_tensor[..., 2, 2] = -q_values[..., 0] - q_values[..., 3]

        diff_q = np.zeros((4, 5, 6, 3, 3, 3), dtype=float)
        diff_q[..., 0, :, :] = np.gradient(q_tensor, axis=0)
        diff_q[..., 1, :, :] = np.gradient(q_tensor, axis=1)
        diff_q[..., 2, :, :] = np.gradient(q_tensor, axis=2)

        levi = np.zeros((3, 3, 3), dtype=float)
        levi[0, 1, 2], levi[1, 2, 0], levi[2, 0, 1] = 1.0, 1.0, 1.0
        levi[1, 0, 2], levi[2, 1, 0], levi[0, 2, 1] = -1.0, -1.0, -1.0

        twist_linear = np.einsum("abc,...ad,...bcd->...", levi, q_tensor, diff_q)
        temp1 = np.einsum("...ab,...aib->...i", q_tensor, diff_q)
        temp2 = np.einsum("...ia,...bab->...i", q_tensor, diff_q)
        splay_vector = temp1 + 2.0 * temp2
        bend_vector = -2.0 * temp1 - temp2

        self.assertEqual(set(result), {
            "splay_vector",
            "twist_linear",
            "bend_vector",
            "splay",
            "twist",
            "bend",
        })
        self.assertEqual(result["splay_vector"].shape, (4, 5, 6, 3))
        self.assertEqual(result["twist_linear"].shape, (4, 5, 6))
        self.assertEqual(result["bend_vector"].shape, (4, 5, 6, 3))
        self.assertEqual(result["splay"].shape, (4, 5, 6))
        self.assertEqual(result["twist"].shape, (4, 5, 6))
        self.assertEqual(result["bend"].shape, (4, 5, 6))
        self.assertTrue(np.allclose(result["splay_vector"], splay_vector))
        self.assertTrue(np.allclose(result["twist_linear"], twist_linear))
        self.assertTrue(np.allclose(result["bend_vector"], bend_vector))
        self.assertTrue(np.allclose(result["splay"], np.sum(splay_vector**2, axis=-1)))
        self.assertTrue(np.allclose(result["twist"], twist_linear**2))
        self.assertTrue(np.allclose(result["bend"], np.sum(bend_vector**2, axis=-1)))

    def test_elastic_deformation_uses_physical_gradient_and_output_flags(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(4, 4, 4),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        i, j, k = np.indices((4, 4, 4), dtype=float)
        x = 2.0 * i
        y = 3.0 * j
        z = 4.0 * k

        q_tensor = np.zeros((4, 4, 4, 3, 3), dtype=float)
        q_tensor[..., 0, 0] = x + y
        q_tensor[..., 0, 1] = y
        q_tensor[..., 0, 2] = z
        q_tensor[..., 1, 0] = y
        q_tensor[..., 1, 1] = x - z
        q_tensor[..., 1, 2] = x + z
        q_tensor[..., 2, 0] = z
        q_tensor[..., 2, 1] = x + z
        q_tensor[..., 2, 2] = -q_tensor[..., 0, 0] - q_tensor[..., 1, 1]
        dataset.act_add_field("Q", q_tensor)

        vector_result = dataset.act_elastic_deformation(
            "Q",
            is_return_scalar=False,
            is_return_vector=True,
        )
        scalar_result = dataset.act_elastic_deformation(
            "Q",
            is_return_scalar=True,
            is_return_vector=False,
        )

        diff_q = np.zeros((4, 4, 4, 3, 3, 3), dtype=float)
        spacing = (2.0, 3.0, 4.0)
        diff_q[..., 0, :, :] = np.gradient(q_tensor, axis=0) / spacing[0]
        diff_q[..., 1, :, :] = np.gradient(q_tensor, axis=1) / spacing[1]
        diff_q[..., 2, :, :] = np.gradient(q_tensor, axis=2) / spacing[2]

        levi = np.zeros((3, 3, 3), dtype=float)
        levi[0, 1, 2], levi[1, 2, 0], levi[2, 0, 1] = 1.0, 1.0, 1.0
        levi[1, 0, 2], levi[2, 1, 0], levi[0, 2, 1] = -1.0, -1.0, -1.0

        twist_linear = np.einsum("abc,...ad,...bcd->...", levi, q_tensor, diff_q)
        temp1 = np.einsum("...ab,...aib->...i", q_tensor, diff_q)
        temp2 = np.einsum("...ia,...bab->...i", q_tensor, diff_q)
        splay_vector = temp1 + 2.0 * temp2
        bend_vector = -2.0 * temp1 - temp2

        self.assertEqual(set(vector_result), {
            "splay_vector",
            "twist_linear",
            "bend_vector",
        })
        self.assertEqual(set(scalar_result), {"splay", "twist", "bend"})
        self.assertTrue(np.allclose(vector_result["splay_vector"], splay_vector))
        self.assertTrue(np.allclose(vector_result["twist_linear"], twist_linear))
        self.assertTrue(np.allclose(vector_result["bend_vector"], bend_vector))
        self.assertTrue(
            np.allclose(scalar_result["splay"], np.sum(splay_vector**2, axis=-1))
        )
        self.assertTrue(np.allclose(scalar_result["twist"], twist_linear**2))
        self.assertTrue(
            np.allclose(scalar_result["bend"], np.sum(bend_vector**2, axis=-1))
        )

    def test_elastic_deformation_requires_some_output(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        dataset.act_add_field("Q", np.zeros((2, 2, 2, 5), dtype=float))

        with self.assertRaises(ValueError):
            dataset.act_elastic_deformation(
                "Q",
                is_return_scalar=False,
                is_return_vector=False,
            )

    def test_gaussian_smooth_preserves_constant_field(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(5, 6, 7),
                box_periodic_flag=(True, False, True),
            )
        )
        values = np.full((5, 6, 7), 3.5, dtype=float)

        smoothed = dataset.act_gaussian_smooth(values, sigma=1.2, coord="index")

        self.assertEqual(smoothed.shape, values.shape)
        self.assertTrue(np.allclose(smoothed, values))

    def test_gaussian_smooth_sigma_zero_returns_original_values(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(4, 5, 6)))
        i, j, k = np.indices((4, 5, 6), dtype=float)
        values = i + 2.0 * j + 3.0 * k

        smoothed = dataset.act_gaussian_smooth(values, sigma=0.0, coord="index")

        self.assertTrue(np.allclose(smoothed, values))

    def test_gaussian_smooth_wraps_periodic_boundaries(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(5, 3, 3),
                box_periodic_flag=(True, False, False),
            )
        )
        values = np.zeros((5, 3, 3), dtype=float)
        values[0, 1, 1] = 1.0

        smoothed = dataset.act_gaussian_smooth(
            values,
            sigma=(1.0, 0.0, 0.0),
            coord="index",
            truncate=2.0,
        )

        self.assertGreater(smoothed[4, 1, 1], 0.0)
        self.assertTrue(np.allclose(smoothed[:, 0, :], 0.0))
        self.assertTrue(np.allclose(smoothed[:, 2, :], 0.0))

    def test_gaussian_smooth_uses_physical_spacing(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(7, 3, 3),
                grid_transform=np.diag((2.0, 1.0, 1.0)),
            )
        )
        values = np.zeros((7, 3, 3), dtype=float)
        values[3, 1, 1] = 1.0

        smoothed_physical = dataset.act_gaussian_smooth(
            values,
            sigma=2.0,
            coord="physical",
        )
        smoothed_index = dataset.act_gaussian_smooth(
            values,
            sigma=(1.0, 2.0, 2.0),
            coord="index",
        )

        self.assertTrue(np.allclose(smoothed_physical, smoothed_index))

    def test_gaussian_smooth_can_return_result_metadata(self):
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(4, 5, 6),
                box_periodic_flag=(True, False, False),
                grid_transform=np.diag((2.0, 3.0, 4.0)),
            )
        )
        values = np.zeros((4, 5, 6), dtype=float)
        dataset.act_add_field("scalar", values)

        result = dataset.act_gaussian_smooth(
            "scalar",
            sigma=2.0,
            coord="physical",
            is_result=True,
        )

        self.assertIsInstance(result, GaussianSmoothResult)
        self.assertEqual(result.raw_info.operator, "gaussian_smooth")
        self.assertEqual(result.raw_info.source, "scalar")
        self.assertEqual(result.raw_info.coord, "physical")
        self.assertEqual(result.raw_info.source_shape, (4, 5, 6))
        self.assertEqual(result.raw_info.output_shape, (4, 5, 6))
        self.assertEqual(result.raw_info.sigma, (2.0, 2.0, 2.0))
        self.assertEqual(result.raw_info.sigma_index, (1.0, 2.0 / 3.0, 0.5))
        self.assertEqual(result.raw_info.boundary, ("wrap", "reflect", "reflect"))

    def test_gaussian_smooth_preserves_trailing_component_axes(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(5, 5, 5)))
        values = np.zeros((5, 5, 5, 2), dtype=float)
        values[2, 2, 2, 0] = 1.0
        values[2, 2, 2, 1] = 2.0

        smoothed = dataset.act_gaussian_smooth(values, sigma=1.0, coord="index")

        self.assertEqual(smoothed.shape, values.shape)
        ratio = smoothed[..., 1] / 2.0
        self.assertTrue(np.allclose(smoothed[..., 0], ratio))

    def test_gaussian_smooth_supports_anisotropic_sigma(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(9, 9, 9)))
        values = np.zeros((9, 9, 9), dtype=float)
        values[4, 4, 4] = 1.0

        smoothed = dataset.act_gaussian_smooth(
            values,
            sigma=(2.0, 0.5, 0.5),
            coord="index",
        )

        self.assertGreater(smoothed[3, 4, 4], smoothed[4, 3, 4])
        self.assertGreater(smoothed[5, 4, 4], smoothed[4, 5, 4])

    def test_gaussian_smooth_result_can_be_registered_as_field(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(5, 5, 5)))
        values = np.zeros((5, 5, 5), dtype=float)
        values[2, 2, 2] = 1.0
        dataset.act_add_field("scalar", values)

        result = dataset.act_gaussian_smooth("scalar", sigma=1.0, is_result=True)
        field = dataset.act_add_result_field("scalar_smooth", result)

        self.assertIs(field, dataset["scalar_smooth"])
        self.assertIs(field.raw_info, result.raw_info)
        self.assertTrue(np.allclose(field.raw_values, result.raw_values))

    def test_gaussian_smooth_result_can_save_release_and_load(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(5, 5, 5)))
        values = np.zeros((5, 5, 5), dtype=float)
        values[2, 2, 2] = 1.0
        result = dataset.act_gaussian_smooth(values, sigma=1.0, is_result=True)
        expected = result.raw_values.copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            saved = result.act_save_values(
                Path(tmp_dir) / "gaussian_smooth",
                is_release=True,
            )

            self.assertIsNone(saved.raw_values)
            self.assertTrue(saved.raw_path.endswith(".npy"))
            with saved.act_with_values() as loaded:
                self.assertTrue(np.allclose(loaded, expected))

            loaded_result = saved.act_load_values()
            self.assertTrue(np.allclose(loaded_result.raw_values, expected))

    def test_gaussian_smooth_registered_field_can_feed_gradient(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(7, 7, 7)))
        values = np.zeros((7, 7, 7), dtype=float)
        values[3, 3, 3] = 1.0
        dataset.act_add_field("scalar", values)

        smooth_result = dataset.act_gaussian_smooth("scalar", sigma=1.0, is_result=True)
        dataset.act_add_result_field("scalar_smooth", smooth_result)
        grad = dataset.act_gradient("scalar_smooth", coord="index")

        self.assertEqual(grad.shape, (7, 7, 7, 3))
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertTrue(np.allclose(grad[3, 3, 3], 0.0, atol=1e-12))

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

    def test_laplacian_uses_physical_spacing_on_rotated_grid(self):
        grid_transform = np.array(
            [
                [0.0, -2.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=(7, 7, 7),
                grid_transform=grid_transform,
            )
        )
        i, j, k = np.indices((7, 7, 7), dtype=float)
        x = 3.0 * j
        y = -2.0 * i
        z = 4.0 * k
        values = x**2 + y**2 + z**2

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
