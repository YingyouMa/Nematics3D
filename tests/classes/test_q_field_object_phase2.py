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

from nematics3d.classes.q_field_object import InputQ, QFieldObject
from nematics3d.classes.grid_field import (
    GridFieldDataset,
    GridInterpolator,
    InputGridField,
)
from nematics3d.classes.plane_grid_polar import OptsPlaneGridPolar
from nematics3d.classes.disclination_line import (
    DefectSectionOmegaResult,
    _helper_sample_beta_from_smooth,
)
from nematics3d.classes.q_plane import OmegaResult
from nematics3d.grid import apply_linear_transform


class TestQFieldObjectPhase2(unittest.TestCase):
    def test_q_registers_canonical_dataset_bounds_once(self):
        q_values = np.zeros((2, 2, 2, 5), dtype=float)
        q_values[..., 0] = 2.0 / 3.0
        q_values[..., 3] = -1.0 / 3.0
        q_values[..., 4] = -1.0 / 3.0

        q = QFieldObject(
            inputValue=InputQ(Q=q_values),
            is_detect_defects=False,
            is_classify_lines=False,
        )

        bounds = q.dataset.calc_bounds
        self.assertIs(q.calc_bounds, bounds)
        self.assertEqual(sum(item is bounds for item in q.objects), 1)

    def test_init_excludes_defects_touching_invalid_mask_voxels(self):
        shape = (3, 3, 3)
        n = np.zeros(shape + (3,), dtype=float)
        n[..., 0] = 1.0
        mask = np.ones(shape, dtype=bool)
        mask[0, 0, 0] = False
        detected = np.array(((0.0, 0.5, 0.5), (2.0, 0.5, 0.5)))

        with patch(
            "nematics3d.classes.q_field_object.defect_detect",
            return_value=detected,
        ):
            q = QFieldObject(
                inputValue=InputQ(n=n, mask=mask),
                is_classify_lines=False,
                name="masked-defect-test",
            )

        np.testing.assert_array_equal(
            q.calc_defect_indices, np.array(((2.0, 0.5, 0.5),))
        )
        np.testing.assert_array_equal(
            q.calc_defect_indices_masked, np.array(((0.0, 0.5, 0.5),))
        )
        self.assertIs(q.mask, q.dataset.mask)
        self.assertIs(q.mask, q.dataset.fields["mask"].raw_values)
        self.assertEqual(q.mask.dtype, np.dtype(bool))
        self.assertFalse(q.mask.flags.writeable)

    def test_act_get_beta_interpolator_new_smooth_matches_direct_beta(self):
        data_path = (
            Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
        )
        q_data = np.load(data_path)[0]
        q_data = q_data[168:185, 5:32, 10:35]

        q = QFieldObject(Q=q_data, name="beta-test")

        self.assertEqual(len(q.lines), 1)
        self.assertEqual(len(q.lines[0].smooths), 0)

        beta_func = q.act_get_beta_interpolator(
            index_line=0,
            is_new_smooth=True,
            u_samples=np.array([0.0, 25.0, 50.0, 75.0], dtype=float),
            smooth_window_length=28,
            name="beta-test-func",
        )

        smooth = q.lines[0].smooth
        self.assertIsNotNone(smooth)
        self.assertIs(beta_func.owner, smooth)

        u_query = np.array([0.0, 25.0, 50.0, 75.0], dtype=float)
        expected = np.array([smooth.act_calc_omega(u)["beta"] for u in u_query])

        self.assertTrue(np.allclose(beta_func(u_query), expected, equal_nan=True))
        self.assertIs(beta_func.raw_func_kwargs["smooth"], smooth)
        self.assertEqual(beta_func.raw_result_value_attr, "beta")
        self.assertIsInstance(beta_func.calc_results, tuple)
        self.assertEqual(len(beta_func.calc_results), len(beta_func.raw_u_samples))
        result0 = beta_func.calc_results[0]
        self.assertIsInstance(result0, DefectSectionOmegaResult)
        self.assertIsInstance(result0.metric, dict)
        self.assertEqual(result0.omega.shape, (3,))
        self.assertEqual(result0.tangent.shape, (3,))
        self.assertGreater(result0.R, 0)
        self.assertGreater(result0.num_directors, 0)
        self.assertGreaterEqual(result0.layer, 0)

    def test_act_get_beta_interpolator_requires_existing_smooth_when_not_new(self):
        data_path = (
            Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
        )
        q_data = np.load(data_path)[0]
        q_data = q_data[168:185, 5:32, 10:35]

        q = QFieldObject(Q=q_data, name="beta-no-smooth")

        with self.assertRaisesRegex(ValueError, "is_new_smooth=True"):
            q.act_get_beta_interpolator(
                index_line=0,
                u_samples=np.array([0.0, 50.0], dtype=float),
            )

    def test_act_get_beta_interpolator_stores_omega_grid_kwargs_on_linefunc(self):
        data_path = (
            Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
        )
        q_data = np.load(data_path)[0]
        q_data = q_data[168:185, 5:32, 10:35]

        q = QFieldObject(Q=q_data, name="beta-grid-kwargs")
        opts_grid = OptsPlaneGridPolar(dr=0.4, layers=8)
        beta_func = q.act_get_beta_interpolator(
            index_line=0,
            is_new_smooth=True,
            u_samples=np.array([0.0, 50.0], dtype=float),
            smooth_window_length=28,
            opts_grid=opts_grid,
            grid_arc_dist=0.5,
        )

        self.assertIs(beta_func.raw_func, _helper_sample_beta_from_smooth)
        self.assertIsInstance(
            beta_func.raw_func_kwargs["opts_grid"], OptsPlaneGridPolar
        )
        self.assertEqual(beta_func.raw_func_kwargs["opts_grid"].dr, 0.4)
        self.assertEqual(beta_func.raw_func_kwargs["opts_grid"].arc_dist, 0.5)
        self.assertIn("smooth", beta_func.raw_func_kwargs)
        self.assertEqual(beta_func.name, "beta_line_0_smooth_0")

    def test_act_get_beta_interpolator_new_smooth_appends_even_when_cached_exists(self):
        data_path = (
            Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
        )
        q_data = np.load(data_path)[0]
        q_data = q_data[168:185, 5:32, 10:35]

        q = QFieldObject(Q=q_data, name="beta-force-new")
        q.act_lines_smooth(window_length=28)
        self.assertEqual(len(q.lines[0].smooths), 1)

        beta_func = q.act_get_beta_interpolator(
            index_line=0,
            is_new_smooth=True,
            u_samples=np.array([0.0, 50.0], dtype=float),
            smooth_window_length=31,
        )

        self.assertEqual(len(q.lines[0].smooths), 2)
        self.assertIs(beta_func.owner, q.lines[0].smooths[-1])
        self.assertEqual(beta_func.name, "beta_line_0_smooth_1")

    def test_smoothed_line_act_add_beta_interpolator_builds_linefunc_directly(self):
        data_path = (
            Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
        )
        q_data = np.load(data_path)[0]
        q_data = q_data[168:185, 5:32, 10:35]

        q = QFieldObject(Q=q_data, name="beta-smooth-direct")
        q.act_lines_smooth(window_length=28)
        smooth = q.lines[0].smooth

        beta_func = smooth.act_add_beta_interpolator(
            u_samples=np.array([0.0, 50.0], dtype=float),
            grid_arc_dist=0.5,
        )

        self.assertIs(beta_func.owner, smooth)
        self.assertIs(beta_func.raw_func, _helper_sample_beta_from_smooth)
        self.assertEqual(beta_func.name, "beta_smooth_0")
        self.assertEqual(beta_func.raw_func_kwargs["opts_grid"].arc_dist, 0.5)

    def test_smoothed_line_act_add_beta_interpolator_wrap_drops_100_endpoint(self):
        data_path = (
            Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
        )
        q_data = np.load(data_path)[0]
        q_data = q_data[168:185, 5:32, 10:35]

        q = QFieldObject(Q=q_data, name="beta-wrap-endpoint")
        q.act_lines_smooth(window_length=28)
        smooth = q.lines[0].smooth
        object.__setattr__(smooth.owner, "calc_end2end_kind", "loop")

        beta_func = smooth.act_add_beta_interpolator(
            u_samples=np.array([0.0, 25.0, 50.0, 100.0], dtype=float),
        )

        self.assertTrue(
            np.allclose(beta_func.raw_u_samples, np.array([0.0, 25.0, 50.0]))
        )

    def test_act_calc_omega_returns_result_base_objects(self):
        data_path = (
            Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
        )
        q_data = np.load(data_path)[0]
        q_data = q_data[168:185, 5:32, 10:35]

        q = QFieldObject(Q=q_data, name="beta-result-test")
        q.act_lines_smooth(window_length=28)
        smooth = q.lines[0].smooth

        section_result = smooth.act_calc_omega(5.0)
        self.assertIsInstance(section_result, DefectSectionOmegaResult)
        self.assertIsInstance(section_result, OmegaResult)
        self.assertTrue("beta" in section_result)
        self.assertTrue("omega" in section_result)
        self.assertTrue(
            np.isfinite(section_result["beta"]) or np.isnan(section_result["beta"])
        )

    def test_legacy_init_builds_dataset_owned_q_field(self):
        shape = (2, 2, 2)
        n = np.zeros(shape + (3,), dtype=float)
        n[..., 0] = 1.0
        S = np.ones(shape, dtype=float)
        grid_transform = np.diag((2.0, 3.0, 4.0))
        grid_offset = (10.0, 20.0, 30.0)

        q = QFieldObject(
            inputValue=InputQ(
                n=n,
                S=S,
                box_periodic_flag=(True, False, True),
                grid_offset=grid_offset,
                grid_transform=grid_transform,
            ),
            name="phase2-q",
            is_detect_defects=False,
            is_classify_lines=False,
        )

        self.assertIsNotNone(q.dataset)
        self.assertIsNotNone(q.field)
        self.assertIs(q.field, q.dataset["Q"])
        self.assertIs(q.field.owner, q.dataset)
        self.assertIs(q.raw_Q, q.field.raw_values)
        self.assertEqual(tuple(q.dataset.raw_shape), shape)
        self.assertTrue(
            np.allclose(
                q.dataset.act_generate_grid(),
                apply_linear_transform(
                    q.dataset.act_generate_grid(coord="index"),
                    transform=grid_transform,
                    offset=grid_offset,
                ),
            )
        )
        self.assertTrue(np.allclose(q.calc_corners, q.dataset.calc_corners))
        self.assertTrue(
            np.allclose(
                q.calc_box_size_periodic_index,
                q.dataset.calc_box_size_periodic_index,
            )
        )
        self.assertIs(q.calc_bounds, q.dataset.calc_bounds)
        self.assertEqual(tuple(q.raw_box_periodic_flag), (True, False, True))
        self.assertEqual(tuple(q.raw_grid_offset), grid_offset)
        self.assertTrue(np.allclose(q.raw_grid_transform, grid_transform))
        self.assertIsInstance(q.interpolator, GridInterpolator)
        self.assertIs(q.interpolator, q.field.interpolator)

    def test_n_init_uses_broadcast_unit_s_and_direct_q5(self):
        n = np.zeros((2, 3, 4, 3), dtype=np.float32)
        n[..., 0] = 1.0

        q = QFieldObject(
            n=n,
            is_detect_defects=False,
            is_classify_lines=False,
        )

        self.assertEqual(q.raw_S.shape, n.shape[:-1])
        self.assertEqual(q.raw_S.strides, (0, 0, 0))
        self.assertFalse(q.raw_S.flags.writeable)
        self.assertEqual(q.raw_Q.shape, n.shape[:-1] + (5,))
        np.testing.assert_allclose(q.raw_S, 1.0)
        np.testing.assert_allclose(
            q.raw_Q,
            np.broadcast_to(
                np.array([2.0 / 3.0, 0.0, 0.0, -1.0 / 3.0, 0.0]),
                q.raw_Q.shape,
            ),
        )

    def test_attached_analysis_init_reuses_existing_dataset_owned_q_field(self):
        shape = (2, 2, 2)
        q_values = np.zeros(shape + (5,), dtype=float)
        q_values[..., 0] = 2.0 / 3.0
        q_values[..., 3] = -1.0 / 3.0
        q_values[..., 4] = -1.0 / 3.0

        dataset = GridFieldDataset(
            inputValue=InputGridField(
                shape=shape,
                box_periodic_flag=(False, True, False),
                grid_offset=(1.0, 2.0, 3.0),
                grid_transform=np.diag((1.5, 2.5, 3.5)),
            ),
            name="attached-dataset",
        )
        field = dataset.act_add_field("Q", q_values)

        q = QFieldObject(
            field=field,
            name="attached-q",
            is_detect_defects=False,
            is_classify_lines=False,
            default_miminum_line_length_smooth=101,
            default_smooth_window_length=51,
        )

        self.assertIs(q.dataset, dataset)
        self.assertIs(q.field, field)
        self.assertIs(q.raw_Q, field.raw_values)
        self.assertEqual(len(dataset.fields), 1)
        self.assertIs(dataset["Q"], field)
        self.assertEqual(dataset.act_generate_grid().shape, shape + (3,))
        self.assertTrue(np.allclose(q.calc_corners, dataset.calc_corners))
        self.assertIs(q.calc_bounds, dataset.calc_bounds)
        self.assertEqual(tuple(q.raw_box_periodic_flag), (False, True, False))
        self.assertEqual(tuple(q.raw_grid_offset), (1.0, 2.0, 3.0))
        self.assertTrue(np.allclose(q.raw_grid_transform, np.diag((1.5, 2.5, 3.5))))
        self.assertEqual(q.default_miminum_line_length_smooth, 101)
        self.assertEqual(q.default_smooth_window_length, 51)
        self.assertEqual(q.default_miminum_line_length_visual, 75)

    def test_attached_analysis_init_ignores_extra_raw_grid_input(self):
        dataset = GridFieldDataset(inputValue=InputGridField(shape=(2, 2, 2)))
        field = dataset.act_add_field("Q", np.zeros((2, 2, 2, 5), dtype=float))

        q = QFieldObject(
            field=field,
            inputValue=InputQ(
                Q=np.ones((2, 2, 2, 5), dtype=float),
                box_periodic_flag=(True, True, True),
                grid_offset=(9.0, 9.0, 9.0),
            ),
            is_detect_defects=False,
            is_classify_lines=False,
        )

        self.assertIs(q.field, field)
        self.assertIs(q.dataset, dataset)
        self.assertEqual(
            tuple(q.raw_box_periodic_flag), tuple(dataset.raw_box_periodic_flag)
        )
        self.assertIs(q.raw_grid_offset, dataset.raw_grid_offset)

    def test_legacy_init_keeps_default_grid_offset_as_none(self):
        shape = (2, 2, 2)
        n = np.zeros(shape + (3,), dtype=float)
        n[..., 0] = 1.0
        S = np.ones(shape, dtype=float)

        q = QFieldObject(
            inputValue=InputQ(n=n, S=S),
            is_detect_defects=False,
            is_classify_lines=False,
        )

        self.assertIsNone(q.raw_grid_offset)
        self.assertIsNone(q.dataset.raw_grid_offset)
        self.assertTrue(
            np.allclose(
                q.dataset.act_generate_grid(),
                q.dataset.act_generate_grid(coord="index"),
            )
        )


if __name__ == "__main__":
    unittest.main()
