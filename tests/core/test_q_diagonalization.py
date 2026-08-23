import io
import logging
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from nematics3d import q_diagonalize  # noqa: E402
from nematics3d.classes.result_base import ResultBase  # noqa: E402


def make_uniaxial_q_tensor(director, scalar_order):
    director = np.asarray(director, dtype=float)
    director /= np.linalg.norm(director)
    return scalar_order * (np.outer(director, director) - np.eye(3) / 3)


class TestQDiagonalizeResult(unittest.TestCase):
    def test_default_result_uses_named_fast_outputs(self):
        input_director = np.array([1.0, 1.0, 1.0], dtype=float)
        input_director /= np.linalg.norm(input_director)
        input_scalar_order = np.array(0.75)
        q_tensor = make_uniaxial_q_tensor(input_director, input_scalar_order)

        result = q_diagonalize(q_tensor, log_mode="none")

        self.assertIsInstance(result, ResultBase)
        self.assertTrue(np.allclose(result.S, input_scalar_order))
        self.assertTrue(
            np.allclose(
                np.abs(np.dot(result.n, input_director)),
                1.0,
            )
        )
        self.assertEqual(result.isotropic_indices, [])
        self.assertEqual(result.uniaxial_indices, [])
        self.assertIsNone(result.eigenvalues)
        self.assertIsNone(result.eigenvectors)
        self.assertIsNone(result.biaxial_order)

    def test_default_path_does_not_compute_complete_eigensystem(self):
        q_tensor = make_uniaxial_q_tensor([1.0, 0.0, 0.0], 1.0)

        with patch(
            "nematics3d.q_diagonalization._eigh3_q_sd",
            side_effect=AssertionError("The default path must stay principal-only."),
        ):
            result = q_diagonalize(q_tensor, log_mode="none")

        self.assertTrue(np.allclose(result.S, 1.0))
        self.assertTrue(np.allclose(np.abs(result.n), [1.0, 0.0, 0.0]))

    def test_result_base_attribute_and_dictionary_interfaces(self):
        q_tensor = make_uniaxial_q_tensor([1.0, 1.0, 1.0], 0.75)
        result = q_diagonalize(q_tensor, log_mode="none")

        expected_keys = (
            "S",
            "n",
            "isotropic_indices",
            "uniaxial_indices",
            "eigenvalues",
            "eigenvectors",
            "biaxial_order",
        )
        self.assertEqual(result.keys(), expected_keys)
        self.assertEqual(tuple(result), expected_keys)
        self.assertEqual(len(result), len(expected_keys))
        self.assertIn("S", result)
        self.assertNotIn("missing", result)

        self.assertIs(result.S, result["S"])
        self.assertIs(result.n, result["n"])
        self.assertIs(result.get("n"), result.n)
        self.assertEqual(result.get("missing", "fallback"), "fallback")
        with self.assertRaises(KeyError):
            result["missing"]

        values = result.values()
        items = result.items()
        self.assertEqual(len(values), len(expected_keys))
        self.assertEqual(tuple(key for key, _ in items), expected_keys)
        self.assertIs(values[1], result.n)
        self.assertIs(items[1][1], result.n)

        result_dict = result.asdict()
        self.assertEqual(tuple(result_dict), expected_keys)
        self.assertIs(result_dict["n"], result.n)

    def test_result_base_representation_and_field_descriptions(self):
        q_tensor = make_uniaxial_q_tensor([1.0, 1.0, 1.0], 0.75)
        result = q_diagonalize(q_tensor, log_mode="none")

        representation = repr(result)
        self.assertTrue(
            representation.startswith(
                "QDiagonalizationResult: Q-tensor diagonalization\n"
            )
        )
        self.assertIn("S", representation)
        self.assertIn("isotropic_indices", representation)

        descriptions = result.show_readable_attrs(
            is_return=True,
            log_mode="none",
        )
        self.assertIn("- S", descriptions)
        self.assertIn("Scalar order", descriptions)
        self.assertEqual(
            result.show_attr_doc("n", is_return=True, log_mode="none"),
            "Unit eigenvector for the largest eigenvalue.",
        )
        with self.assertRaises(KeyError):
            result.show_attr_doc("missing", log_mode="none")

    def test_result_base_field_description_logging(self):
        q_tensor = make_uniaxial_q_tensor([1.0, 1.0, 1.0], 0.75)
        result = q_diagonalize(q_tensor, log_mode="none")

        screen_output = io.StringIO()
        with redirect_stdout(screen_output):
            returned_doc = result.show_attr_doc("S", is_return=True)

        self.assertIn("Scalar order", returned_doc)
        self.assertIn("[INFO]", screen_output.getvalue())
        self.assertIn("Scalar order", screen_output.getvalue())
        self.assertNotIn("STARTED", screen_output.getvalue())

        suppressed_output = io.StringIO()
        with redirect_stdout(suppressed_output):
            returned_doc = result.show_attr_doc(
                "S",
                is_return=True,
                log_mode="none",
            )

        self.assertIn("Scalar order", returned_doc)
        self.assertEqual(suppressed_output.getvalue(), "")

    def test_biaxial_result_returns_complete_descending_eigensystem(self):
        random_generator = np.random.default_rng(0)
        input_axes, _ = np.linalg.qr(random_generator.normal(size=(3, 3)))
        expected_eigenvalues = np.array([0.6, -0.1, -0.5])
        q_tensor = input_axes @ np.diag(expected_eigenvalues) @ input_axes.T

        with patch(
            "nematics3d.q_diagonalization.np.linalg.eigh",
            side_effect=AssertionError(
                "The stable biaxial path must remain fully analytic."
            ),
        ):
            result = q_diagonalize(
                q_tensor,
                is_biaxial=True,
                is_right_handed=True,
                log_mode="none",
            )

        self.assertTrue(np.allclose(result.eigenvalues, expected_eigenvalues))
        self.assertTrue(np.allclose(result.S, 0.9))
        self.assertTrue(np.allclose(result.biaxial_order, 0.6))
        self.assertTrue(np.allclose(result.n, result.eigenvectors[..., :, 0]))
        reconstructed_tensor = (
            result.eigenvectors @ np.diag(result.eigenvalues) @ result.eigenvectors.T
        )
        self.assertTrue(np.allclose(reconstructed_tensor, q_tensor))

    def test_complete_eigenvector_frames_can_be_made_right_handed(self):
        random_generator = np.random.default_rng(42)
        q_tensors = []
        expected_eigenvalues = np.array([0.6, -0.1, -0.5])
        for _ in range(8):
            axes, _ = np.linalg.qr(random_generator.normal(size=(3, 3)))
            q_tensors.append(axes @ np.diag(expected_eigenvalues) @ axes.T)

        result = q_diagonalize(
            np.stack(q_tensors),
            is_biaxial=True,
            is_right_handed=True,
            log_mode="none",
        )

        self.assertTrue(np.all(np.linalg.det(result.eigenvectors) > 0.0))
        reconstructed_tensors = np.einsum(
            "...ik,...k,...jk->...ij",
            result.eigenvectors,
            result.eigenvalues,
            result.eigenvectors,
        )
        self.assertTrue(np.allclose(reconstructed_tensors, q_tensors))

    def test_random_symmetric_traceless_tensors_match_numpy_eigh(self):
        random_generator = np.random.default_rng(20260819)
        random_matrices = random_generator.normal(size=(64, 3, 3))
        symmetric_tensors = 0.5 * (
            random_matrices + np.swapaxes(random_matrices, -1, -2)
        )
        traces = np.trace(symmetric_tensors, axis1=-2, axis2=-1)
        q_tensors = symmetric_tensors - traces[..., None, None] * np.eye(3) / 3.0

        result = q_diagonalize(
            q_tensors,
            is_biaxial=True,
            is_right_handed=True,
            log_mode="none",
        )
        expected_eigenvalues, expected_eigenvectors = np.linalg.eigh(q_tensors)
        expected_eigenvalues = expected_eigenvalues[..., ::-1]
        expected_eigenvectors = expected_eigenvectors[..., :, ::-1]

        self.assertTrue(np.allclose(result.eigenvalues, expected_eigenvalues))
        axis_overlaps = np.einsum(
            "...ij,...ij->...j",
            result.eigenvectors,
            expected_eigenvectors,
        )
        self.assertTrue(np.allclose(np.abs(axis_overlaps), 1.0))
        self.assertTrue(np.allclose(result.S, 1.5 * expected_eigenvalues[..., 0]))
        self.assertTrue(np.all(np.linalg.det(result.eigenvectors) > 0.0))

    def test_right_handed_frames_require_complete_biaxial_output(self):
        q_tensor = make_uniaxial_q_tensor([1.0, 0.0, 0.0], 1.0)

        with self.assertRaisesRegex(
            ValueError,
            "is_right_handed=True.*is_biaxial=True",
        ):
            q_diagonalize(
                q_tensor,
                is_right_handed=True,
                log_mode="none",
            )

    def test_biaxial_isotropic_result_uses_deterministic_frame(self):
        q_tensor = np.zeros((2, 3, 3), dtype=float)

        result = q_diagonalize(
            q_tensor,
            is_biaxial=True,
            log_mode="none",
        )

        self.assertTrue(np.allclose(result.S, 0.0))
        self.assertEqual(result.isotropic_indices, [(0,), (1,)])
        self.assertTrue(np.allclose(result.biaxial_order, 0.0))
        self.assertTrue(
            np.allclose(result.eigenvectors, np.broadcast_to(np.eye(3), (2, 3, 3)))
        )
        self.assertTrue(np.allclose(result.n, np.array([[1.0, 0.0, 0.0]] * 2)))

    def test_single_isotropic_tensor_uses_empty_coordinate_tuple(self):
        result = q_diagonalize(
            np.zeros((3, 3), dtype=float),
            log_mode="none",
        )

        self.assertEqual(result.isotropic_indices, [()])
        self.assertEqual(result.uniaxial_indices, [])

    def test_empty_q_tensor_input_is_rejected(self):
        empty_inputs = (
            np.empty((0, 5), dtype=float),
            np.empty((0, 3, 3), dtype=float),
        )

        for empty_input in empty_inputs:
            with self.subTest(shape=empty_input.shape):
                with self.assertRaisesRegex(
                    ValueError,
                    "must contain at least one Q tensor",
                ):
                    q_diagonalize(empty_input, log_mode="none")

    def test_debug_logging_reports_stages_counts_and_elapsed_time(self):
        q_tensor = np.stack([make_uniaxial_q_tensor([1.0, 1.0, 1.0], 0.75)] * 2)
        screen_output = io.StringIO()

        with redirect_stdout(screen_output):
            q_diagonalize(q_tensor, log_level=logging.DEBUG)

        debug_output = screen_output.getvalue()
        self.assertIn("Computing tensor invariants for 2 Q tensor(s)", debug_output)
        self.assertIn("Computed tensor invariants for 2 Q tensor(s) in", debug_output)
        self.assertIn(
            "Computing the largest eigenvalue for 2 Q tensor(s)", debug_output
        )
        self.assertIn("Computed the director for 2 Q tensor(s) in", debug_output)
        self.assertIn("seconds", debug_output)

    def test_isotropic_warning_points_to_result_indices(self):
        screen_output = io.StringIO()

        with redirect_stdout(screen_output):
            q_diagonalize(np.zeros((3, 3), dtype=float))

        warning_output = screen_output.getvalue()
        self.assertIn("[WARNING]", warning_output)
        self.assertIn("1 near-isotropic grid point(s)", warning_output)
        self.assertIn("result.isotropic_indices", warning_output)

    def test_x_aligned_tensor_uses_robust_analytic_path(self):
        x_aligned_biaxial_tensor = np.diag([0.6, -0.1, -0.5])

        with patch(
            "nematics3d.q_diagonalization.np.linalg.eigh",
            side_effect=AssertionError("The robust analytic path must not fall back."),
        ):
            result = q_diagonalize(
                x_aligned_biaxial_tensor,
                is_biaxial=True,
                log_mode="none",
            )

        self.assertTrue(np.allclose(result.eigenvalues, [0.6, -0.1, -0.5]))
        self.assertTrue(
            np.allclose(result.eigenvectors.T @ result.eigenvectors, np.eye(3))
        )
        reconstructed_tensor = (
            result.eigenvectors @ np.diag(result.eigenvalues) @ result.eigenvectors.T
        )
        self.assertTrue(np.allclose(reconstructed_tensor, x_aligned_biaxial_tensor))

    def test_x_aligned_perfect_uniaxial_tensor_has_orthonormal_frame(self):
        x_aligned_uniaxial_tensor = make_uniaxial_q_tensor([1.0, 0.0, 0.0], 1.0)

        with patch(
            "nematics3d.q_diagonalization.np.linalg.eigh",
            side_effect=AssertionError("Perfect uniaxial input must remain analytic."),
        ):
            result = q_diagonalize(
                x_aligned_uniaxial_tensor,
                is_biaxial=True,
                is_right_handed=True,
                log_mode="none",
            )

        self.assertTrue(np.allclose(result.S, 1.0))
        self.assertTrue(np.allclose(np.abs(result.n), [1.0, 0.0, 0.0]))
        self.assertTrue(
            np.allclose(result.eigenvectors.T @ result.eigenvectors, np.eye(3))
        )
        self.assertGreater(np.linalg.det(result.eigenvectors), 0.0)

    def test_near_degenerate_secondary_axes_remain_orthonormal(self):
        random_generator = np.random.default_rng(20260823)
        q_tensors = []
        for gap in np.logspace(-14, -3, 32):
            axes, _ = np.linalg.qr(random_generator.normal(size=(3, 3)))
            eigenvalues = np.array([0.5, -0.25 + gap / 2, -0.25 - gap / 2])
            q_tensors.append(axes @ np.diag(eigenvalues) @ axes.T)

        q_tensors = np.stack(q_tensors)
        result = q_diagonalize(q_tensors, is_biaxial=True, log_mode="none")
        reconstructed = np.einsum(
            "...ik,...k,...jk->...ij",
            result.eigenvectors,
            result.eigenvalues,
            result.eigenvectors,
        )

        self.assertTrue(np.allclose(reconstructed, q_tensors, atol=1e-12))
        frames = np.swapaxes(result.eigenvectors, -1, -2) @ result.eigenvectors
        self.assertTrue(np.allclose(frames, np.eye(3), atol=1e-12))
        secondary_overlap = np.abs(
            np.einsum(
                "...i,...i->...",
                result.eigenvectors[..., :, 1],
                result.eigenvectors[..., :, 2],
            )
        )
        self.assertTrue(np.all(secondary_overlap < 1e-12))

    def test_positive_uniaxial_result_canonicalizes_degenerate_eigensystem(self):
        input_director = np.array([0.3, 0.4, np.sqrt(0.75)])
        input_scalar_order = 1.0
        q_tensor = make_uniaxial_q_tensor(input_director, input_scalar_order)

        with patch(
            "nematics3d.q_diagonalization.np.linalg.eigh",
            side_effect=AssertionError(
                "A stable positive-S uniaxial point must remain analytic."
            ),
        ):
            result = q_diagonalize(
                q_tensor,
                is_biaxial=True,
                is_right_handed=True,
                log_mode="none",
            )

        self.assertTrue(np.allclose(result.S, input_scalar_order))
        self.assertEqual(result.biaxial_order, 0.0)
        self.assertTrue(
            np.allclose(result.eigenvalues, np.array([2 / 3, -1 / 3, -1 / 3]))
        )
        self.assertTrue(
            np.allclose(result.eigenvectors.T @ result.eigenvectors, np.eye(3))
        )
        self.assertGreater(np.linalg.det(result.eigenvectors), 0.0)
        reconstructed_tensor = (
            result.eigenvectors @ np.diag(result.eigenvalues) @ result.eigenvectors.T
        )
        self.assertTrue(np.allclose(reconstructed_tensor, q_tensor))
        self.assertTrue(np.allclose(abs(np.dot(result.n, input_director)), 1.0))

    def test_uniaxial_indices_identify_only_canonicalized_grid_points(self):
        input_director = np.array([0.3, 0.4, np.sqrt(0.75)])
        uniaxial_tensor = make_uniaxial_q_tensor(input_director, 1.0)
        biaxial_tensor = np.diag([0.6, -0.1, -0.5])
        q_tensor = np.stack([uniaxial_tensor, biaxial_tensor])

        result = q_diagonalize(
            q_tensor,
            is_biaxial=True,
            log_mode="none",
        )

        self.assertEqual(result.uniaxial_indices, [(0,)])


if __name__ == "__main__":
    unittest.main()
