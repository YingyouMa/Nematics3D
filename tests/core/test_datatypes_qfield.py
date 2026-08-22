import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from nematics3d.datatypes import as_qfield5, as_qfield9  # noqa: E402


class TestAsQField9Validation(unittest.TestCase):
    def test_converts_compact_float32_batch_to_symmetric_traceless_tensors(self):
        compact = np.array(
            [
                [0.4, 0.1, -0.2, -0.3, 0.05],
                [-0.2, 0.0, 0.3, 0.1, -0.1],
            ],
            dtype=np.float32,
        )
        original = compact.copy()

        result = as_qfield9(compact, is_strict_3d_field=False)

        self.assertEqual(result.shape, (2, 3, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(compact, original)
        np.testing.assert_array_equal(result, np.swapaxes(result, -2, -1))
        np.testing.assert_allclose(np.trace(result, axis1=-2, axis2=-1), 0.0)

    def test_strict_3d_field_rejects_zero_length_spatial_axes(self):
        shapes = (
            (0, 2, 3, 5),
            (2, 0, 3, 3, 3),
        )

        for shape in shapes:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(
                    ValueError,
                    r"nonzero spatial dimensions.*is_strict_3d_field=True",
                ):
                    as_qfield9(np.zeros(shape))

    def test_non_strict_mode_accepts_empty_tensor_batches(self):
        shape_pairs = (
            ((0, 5), (0, 3, 3)),
            ((0, 3, 3), (0, 3, 3)),
        )

        for shape, expected_shape in shape_pairs:
            with self.subTest(shape=shape):
                result = as_qfield9(
                    np.zeros(shape),
                    is_strict_3d_field=False,
                )

                self.assertEqual(result.shape, expected_shape)

    def test_accepts_symmetric_traceless_tensor(self):
        qtensor = np.array(
            [
                [0.4, 0.1, -0.2],
                [0.1, -0.3, 0.05],
                [-0.2, 0.05, -0.1],
            ]
        )

        result = as_qfield9(qtensor, is_strict_3d_field=False)

        self.assertIs(result, qtensor)

    def test_rejects_asymmetric_tensor_with_indices(self):
        off_diagonal_pairs = (
            ((0, 1), (1, 0)),
            ((0, 2), (2, 0)),
            ((1, 2), (2, 1)),
        )

        for upper_index, lower_index in off_diagonal_pairs:
            with self.subTest(pair=(upper_index, lower_index)):
                qtensor = np.zeros((2, 3, 3))
                qtensor[(1, *upper_index)] = 0.1

                with self.assertRaisesRegex(ValueError, r"symmetric.*\[\[1\]\]"):
                    as_qfield9(qtensor, is_strict_3d_field=False)

    def test_rejects_nonzero_trace(self):
        qtensor = np.eye(3)

        with self.assertRaisesRegex(ValueError, "traceless"):
            as_qfield9(qtensor, is_strict_3d_field=False)

    def test_rejects_non_finite_five_component_input(self):
        qtensor = np.zeros((2, 5))
        qtensor[1, 0] = np.nan

        with self.assertRaisesRegex(ValueError, r"finite.*\[\[1\]\]"):
            as_qfield9(qtensor, is_strict_3d_field=False)

    def test_rejects_non_finite_full_tensors(self):
        for invalid_value in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid_value=invalid_value):
                qtensor = np.zeros((2, 3, 3))
                qtensor[1, 0, 0] = invalid_value

                with self.assertRaisesRegex(ValueError, r"finite.*\[\[1\]\]"):
                    as_qfield9(qtensor, is_strict_3d_field=False)

    def test_rejects_invalid_dtype_and_shape(self):
        invalid_inputs = (
            (np.zeros((3, 3), dtype=int), TypeError, "float-type"),
            (np.zeros((2, 4), dtype=float), ValueError, "Invalid QField shape"),
        )

        for qtensor, error_type, message in invalid_inputs:
            with self.subTest(shape=qtensor.shape, dtype=qtensor.dtype):
                with self.assertRaisesRegex(error_type, message):
                    as_qfield9(qtensor, is_strict_3d_field=False)

    def test_validation_can_be_explicitly_skipped(self):
        qtensor = np.eye(3)

        result = as_qfield9(
            qtensor,
            is_strict_3d_field=False,
            is_validate_tensor=False,
        )

        self.assertIs(result, qtensor)

    def test_custom_absolute_tolerances_are_applied(self):
        qtensor = np.array(
            [
                [0.4, 1e-5, 0.0],
                [0.0, -0.2, 0.0],
                [0.0, 0.0, -0.20001],
            ]
        )

        result = as_qfield9(
            qtensor,
            is_strict_3d_field=False,
            symmetry_tolerance=1e-4,
            trace_tolerance=1e-4,
        )

        self.assertIs(result, qtensor)

        with self.assertRaisesRegex(ValueError, "symmetric"):
            as_qfield9(
                qtensor,
                is_strict_3d_field=False,
                symmetry_tolerance=1e-6,
                trace_tolerance=1e-4,
            )

        with self.assertRaisesRegex(ValueError, "traceless"):
            as_qfield9(
                qtensor,
                is_strict_3d_field=False,
                symmetry_tolerance=1e-4,
                trace_tolerance=1e-6,
            )

        invalid_tolerances = (
            ("symmetry_tolerance", -1.0),
            ("symmetry_tolerance", np.nan),
            ("trace_tolerance", np.inf),
        )
        compact_qtensor = np.zeros((2, 5))
        for tolerance_name, tolerance in invalid_tolerances:
            with self.subTest(
                tolerance_name=tolerance_name,
                tolerance=tolerance,
            ):
                with self.assertRaisesRegex(ValueError, tolerance_name):
                    as_qfield9(
                        compact_qtensor,
                        is_strict_3d_field=False,
                        is_validate_tensor=False,
                        **{tolerance_name: tolerance},
                    )


class TestAsQField5Validation(unittest.TestCase):
    def test_extracts_compact_components_without_tensor_property_checks(self):
        qtensor = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

        result = as_qfield5(qtensor, is_strict_3d_field=False)

        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 5.0, 6.0])

    def test_rejects_non_finite_values_in_both_representations(self):
        invalid_inputs = (
            (np.full((2, 5), np.inf), r"finite.*\[\[0\], \[1\]\]"),
            (np.full((2, 3, 3), np.nan), r"finite.*\[\[0\], \[1\]\]"),
        )

        for qtensor, message in invalid_inputs:
            with self.subTest(shape=qtensor.shape):
                with self.assertRaisesRegex(ValueError, message):
                    as_qfield5(qtensor, is_strict_3d_field=False)

    def test_strict_empty_fields_are_rejected_but_non_strict_batches_are_allowed(self):
        strict_shapes = (
            (0, 2, 3, 5),
            (2, 0, 3, 3, 3),
        )
        for shape in strict_shapes:
            with self.subTest(mode="strict", shape=shape):
                with self.assertRaisesRegex(ValueError, "nonzero spatial dimensions"):
                    as_qfield5(np.zeros(shape))

        non_strict_shapes = (
            ((0, 5), (0, 5)),
            ((0, 3, 3), (0, 5)),
        )
        for shape, expected_shape in non_strict_shapes:
            with self.subTest(mode="non-strict", shape=shape):
                result = as_qfield5(
                    np.zeros(shape),
                    is_strict_3d_field=False,
                )
                self.assertEqual(result.shape, expected_shape)

    def test_validation_can_be_skipped_but_dtype_and_shape_checks_remain(self):
        compact = np.full((2, 5), np.nan)
        self.assertIs(
            as_qfield5(
                compact,
                is_strict_3d_field=False,
                is_validate_tensor=False,
            ),
            compact,
        )

        invalid_inputs = (
            (np.zeros((3, 3), dtype=int), TypeError, "float-type"),
            (np.zeros((2, 4), dtype=float), ValueError, "Invalid QField shape"),
        )
        for qtensor, error_type, message in invalid_inputs:
            with self.subTest(shape=qtensor.shape, dtype=qtensor.dtype):
                with self.assertRaisesRegex(error_type, message):
                    as_qfield5(
                        qtensor,
                        is_strict_3d_field=False,
                        is_validate_tensor=False,
                    )


if __name__ == "__main__":
    unittest.main()
