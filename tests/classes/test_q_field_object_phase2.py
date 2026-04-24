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

from nematics3d.classes.q_field_object import InputQ, QFieldObject
from nematics3d.classes.grid_field import (
    GridFieldDataset,
    GridInterpolator,
    InputGridField,
)


class TestQFieldObjectPhase2(unittest.TestCase):
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
        self.assertTrue(np.allclose(q.calc_grid, q.dataset.calc_grid))
        self.assertTrue(np.allclose(q.calc_grid_index, q.dataset.calc_grid_index))
        self.assertTrue(
            np.allclose(
                q.calc_box_size_periodic_index,
                q.dataset.calc_box_size_periodic_index,
            )
        )
        self.assertIs(q.calc_corners, q.dataset.calc_corners)
        self.assertEqual(tuple(q.raw_box_periodic_flag), (True, False, True))
        self.assertEqual(tuple(q.raw_grid_offset), grid_offset)
        self.assertTrue(np.allclose(q.raw_grid_transform, grid_transform))
        self.assertIsInstance(q.interpolator, GridInterpolator)
        self.assertIs(q.interpolator, q.field.interpolator)

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
        self.assertTrue(np.allclose(q.calc_grid, dataset.calc_grid))
        self.assertIs(q.calc_corners, dataset.calc_corners)
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
        self.assertEqual(tuple(q.raw_grid_offset), tuple(dataset.raw_grid_offset))


if __name__ == "__main__":
    unittest.main()
