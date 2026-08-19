import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from nematics3d.datatypes import as_qfield9  # noqa: E402


class TestAsQField9Validation(unittest.TestCase):
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
        qtensor = np.zeros((2, 3, 3))
        qtensor[1, 0, 1] = 0.1

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


if __name__ == "__main__":
    unittest.main()
