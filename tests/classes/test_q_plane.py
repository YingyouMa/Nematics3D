import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.grid_field import GridFieldDataset, InputGridField
from nematics3d.classes.plane_grid import OptsPlaneGrid
from nematics3d.classes.q_plane import OmegaResult, QPlane, QPlanePolar
from nematics3d.field import get_q


class TestQPlane(unittest.TestCase):
    def test_q_plane_maps_detected_defect_centers_into_physical_plane_coords(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(3, 3, 2)))
        director = np.zeros((3, 3, 2, 3), dtype=float)
        director[..., 0] = 1.0
        q_values = get_q(director, S=np.ones((3, 3, 2), dtype=float))
        field = dataset.act_add_field("Q", q_values)
        interpolator = field.act_add_interpolator()

        opts = OptsPlaneGrid(
            normal=(0.0, 0.0, 1.0),
            axis1=(1.0, 0.0, 0.0),
            origin=(10.0, 20.0, 30.0),
            spacing=2.0,
            size=4.0,
            alignment="bottom-left",
        )

        with (
            patch(
                "nematics3d.classes.q_plane.defect_detect",
                return_value=np.array([[1, 1, 0]], dtype=int),
            ),
            patch(
                "nematics3d.classes.q_plane.defect_vicinity_grid",
                return_value=np.array([[[1, 1, 0]]], dtype=int),
            ),
        ):
            plane = QPlane(interpolator=interpolator, opts=opts)

        self.assertIsNotNone(plane.calc_defect_pos_all)
        self.assertTrue(
            np.allclose(plane.calc_defect_pos_all, np.array([[12.0, 22.0, 30.0]]))
        )
        self.assertEqual(int(np.sum(plane.calc_is_near_defect)), 1)

    def test_q_plane_polar_maps_structured_rotation_axis_result(self):
        angles = np.linspace(0.0, np.pi / 2.0, 6)
        directors = np.column_stack(
            (np.cos(angles), np.sin(angles), np.zeros_like(angles))
        )
        grid_opts = {"source": "bridge regression"}
        grid = types.SimpleNamespace(
            calc_ring_offsets=np.array([0, len(directors)]),
            entity_polar=np.column_stack((np.full(len(directors), 2.5), angles)),
            entity_grid_all=np.zeros((len(directors), 3)),
            opts=grid_opts,
        )
        interpolator = types.SimpleNamespace(
            interpolate=lambda points, is_out_warning: (
                np.zeros((len(points), 5)),
                np.empty((0, 3)),
            )
        )
        plane = types.SimpleNamespace(
            grid=grid,
            interpolator=interpolator,
            _helper_get_omega_metric_flags=lambda radius, out_points: {
                "is_out_of_domain": False,
                "is_defect_inside_R": False,
                "is_defect_at_center": True,
            },
        )
        logger = types.SimpleNamespace(warning=lambda message: None)

        with patch(
            "nematics3d.classes.q_plane.q_diagonalize",
            return_value=types.SimpleNamespace(n=directors),
        ):
            result = QPlanePolar.act_calc_omega.__wrapped__(
                plane,
                layer=0,
                logger=logger,
            )

        self.assertIsInstance(result, OmegaResult)
        self.assertTrue(np.allclose(result.omega, np.array([0.0, 0.0, 1.0])))
        self.assertEqual(result.layer, 0)
        self.assertEqual(result.num_directors, len(directors))
        self.assertEqual(result.R, 2.5)
        self.assertEqual(result.metric["orthogonality_score"], 1.0)
        self.assertEqual(result.metric["rms_sin_theta"], 0.0)
        self.assertEqual(result.metric["tilt_angle_degrees"], 0.0)
        self.assertEqual(result.metric["rotation_consistency"], 1.0)
        self.assertEqual(result.metric["eigenvalues"].shape, (3,))
        self.assertTrue(np.all(np.diff(result.metric["eigenvalues"]) >= 0.0))
        self.assertFalse(result.metric["is_out_of_domain"])
        self.assertFalse(result.metric["is_defect_inside_R"])
        self.assertTrue(result.metric["is_defect_at_center"])
        self.assertEqual(result.opts, grid_opts)
        self.assertIsNot(result.opts, grid_opts)


if __name__ == "__main__":
    unittest.main()
