import io
import sys
from contextlib import redirect_stdout
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

from nematics3d.field import Q_diagonalize, getQ


class TestQDiagonalize(unittest.TestCase):
    def test_zero_q_uses_isotropic_recovery(self):
        q = np.zeros((1, 1, 1, 3, 3), dtype=float)

        stream = io.StringIO()
        with redirect_stdout(stream):
            S, n = Q_diagonalize(q)

        self.assertTrue(np.allclose(S, 0.0))
        self.assertTrue(np.allclose(n, np.array([[[[1.0, 0.0, 0.0]]]])))
        self.assertIn("near-isotropic grid point(s)", stream.getvalue())
        self.assertIn("Set S = 0", stream.getvalue())

    def test_axis_aligned_uniaxial_q_uses_director_fallback(self):
        n_in = np.array([[[[1.0, 0.0, 0.0]]]], dtype=float)
        q = getQ(n_in, S=np.array([[[1.0]]], dtype=float), log_mode="none")

        stream = io.StringIO()
        with redirect_stdout(stream):
            S, n = Q_diagonalize(q)

        self.assertTrue(np.allclose(S, 1.0))
        self.assertTrue(np.isfinite(n).all())
        self.assertTrue(np.allclose(np.abs(n), np.abs(n_in)))
        self.assertIn("analytic director formula", stream.getvalue())
        self.assertIn("np.linalg.eigh", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
