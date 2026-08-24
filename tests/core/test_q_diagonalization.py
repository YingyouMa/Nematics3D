import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from nematics3d import q_diagonalize  # noqa: E402
from nematics3d.analysis.q_diagonalization._backend import (  # noqa: E402
    is_c_backend_available,
)
from nematics3d.classes.result_base import ResultBase  # noqa: E402


def make_uniaxial_q_tensor(director, scalar_order):
    director = np.asarray(director, dtype=float)
    director /= np.linalg.norm(director)
    return scalar_order * (np.outer(director, director) - np.eye(3) / 3)


class TestQDiagonalizeResult(unittest.TestCase):
    @unittest.skipUnless(is_c_backend_available(), "compiled backend unavailable")
    def test_c_and_python_backends_match_for_full_and_principal_outputs(self):
        random_generator = np.random.default_rng(20260823)
        qfield5 = random_generator.uniform(-0.4, 0.4, size=(9, 7, 5)).astype(np.float32)

        for is_biaxial in (False, True):
            with self.subTest(is_biaxial=is_biaxial):
                c_result = q_diagonalize(
                    qfield5,
                    is_biaxial=is_biaxial,
                    is_use_c_backend=True,
                    worker_count=4,
                    log_mode="none",
                )
                python_result = q_diagonalize(
                    qfield5,
                    is_biaxial=is_biaxial,
                    is_use_c_backend=False,
                    worker_count=1,
                    log_mode="none",
                )

                self.assertTrue(np.allclose(c_result.S, python_result.S))
                director_overlap = np.abs(np.sum(c_result.n * python_result.n, axis=-1))
                self.assertTrue(np.allclose(director_overlap, 1.0))
                if is_biaxial:
                    self.assertTrue(
                        np.allclose(c_result.eigenvalues, python_result.eigenvalues)
                    )
                    c_reconstructed = np.einsum(
                        "...ik,...k,...jk->...ij",
                        c_result.eigenvectors,
                        c_result.eigenvalues,
                        c_result.eigenvectors,
                    )
                    python_reconstructed = np.einsum(
                        "...ik,...k,...jk->...ij",
                        python_result.eigenvectors,
                        python_result.eigenvalues,
                        python_result.eigenvectors,
                    )
                    self.assertTrue(np.allclose(c_reconstructed, python_reconstructed))

    @unittest.skipUnless(is_c_backend_available(), "compiled backend unavailable")
    def test_c_worker_counts_produce_identical_results(self):
        random_generator = np.random.default_rng(13579)
        qfield5 = random_generator.uniform(-0.4, 0.4, size=(17, 11, 5)).astype(
            np.float32
        )

        single = q_diagonalize(
            qfield5,
            is_biaxial=True,
            is_use_c_backend=True,
            worker_count=1,
            log_mode="none",
        )
        parallel = q_diagonalize(
            qfield5,
            is_biaxial=True,
            is_use_c_backend=True,
            worker_count=4,
            log_mode="none",
        )

        np.testing.assert_array_equal(single.eigenvalues, parallel.eigenvalues)
        np.testing.assert_array_equal(single.eigenvectors, parallel.eigenvectors)

    def test_backend_selection_and_worker_validation(self):
        q_tensor = make_uniaxial_q_tensor([1.0, 0.0, 0.0], 1.0)

        with patch(
            "nematics3d.analysis.q_diagonalization._solver.is_c_backend_available",
            return_value=False,
        ):
            fallback = q_diagonalize(q_tensor, log_mode="none")
            self.assertTrue(np.allclose(fallback.S, 1.0))
            with self.assertRaisesRegex(ImportError, "is_use_c_backend=True"):
                q_diagonalize(
                    q_tensor,
                    is_use_c_backend=True,
                    log_mode="none",
                )

        invalid_options = (0, -1, 1.5, True)
        for worker_count in invalid_options:
            with self.subTest(worker_count=worker_count):
                with self.assertRaises((TypeError, ValueError)):
                    q_diagonalize(
                        q_tensor,
                        worker_count=worker_count,
                        log_mode="none",
                    )

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
        self.assertIsNone(result.eigenvalues)
        self.assertIsNone(result.eigenvectors)

    def test_default_path_does_not_compute_complete_eigensystem(self):
        q_tensor = make_uniaxial_q_tensor([1.0, 0.0, 0.0], 1.0)

        with patch(
            "nematics3d.analysis.q_diagonalization._solver._eigh3_q_sd",
            side_effect=AssertionError("The default path must stay principal-only."),
        ):
            result = q_diagonalize(q_tensor, log_mode="none")

        self.assertTrue(np.allclose(result.S, 1.0))
        self.assertTrue(np.allclose(np.abs(result.n), [1.0, 0.0, 0.0]))

    def test_random_symmetric_traceless_tensors_match_numpy_eigh(self):
        random_generator = np.random.default_rng(20260819)
        random_matrices = random_generator.normal(size=(64, 3, 3))
        symmetric_tensors = 0.5 * (
            random_matrices + np.swapaxes(random_matrices, -1, -2)
        )
        traces = np.trace(symmetric_tensors, axis1=-2, axis2=-1)
        q_tensors = symmetric_tensors - traces[..., None, None] * np.eye(3) / 3.0

        with patch(
            "nematics3d.analysis.q_diagonalization._solver.np.linalg.eigh",
            side_effect=AssertionError("The analytic path must not fall back."),
        ):
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
        self.assertTrue(np.all(np.diff(result.eigenvalues, axis=-1) <= 0.0))
        self.assertTrue(np.all(np.linalg.det(result.eigenvectors) > 0.0))
        reconstructed = np.einsum(
            "...ik,...k,...jk->...ij",
            result.eigenvectors,
            result.eigenvalues,
            result.eigenvectors,
        )
        self.assertTrue(np.allclose(reconstructed, q_tensors))

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

    def test_isotropic_results_use_deterministic_conventions(self):
        q_tensor = np.zeros((2, 3, 3), dtype=float)

        result = q_diagonalize(
            q_tensor,
            is_biaxial=True,
            log_mode="none",
        )

        self.assertTrue(np.allclose(result.S, 0.0))
        self.assertEqual(result.isotropic_indices, [(0,), (1,)])
        self.assertTrue(
            np.allclose(result.eigenvectors, np.broadcast_to(np.eye(3), (2, 3, 3)))
        )
        self.assertTrue(np.allclose(result.n, np.array([[1.0, 0.0, 0.0]] * 2)))

        screen_output = io.StringIO()
        with redirect_stdout(screen_output):
            single_result = q_diagonalize(np.zeros((3, 3), dtype=float))
        self.assertEqual(single_result.isotropic_indices, [()])
        self.assertIn("1 near-isotropic grid point(s)", screen_output.getvalue())
        self.assertIn("result.isotropic_indices", screen_output.getvalue())

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


if __name__ == "__main__":
    unittest.main()
