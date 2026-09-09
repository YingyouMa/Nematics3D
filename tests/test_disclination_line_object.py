import numpy as np

from nematics3d.analysis.disclination import (
    DisclinationLine,
    DisclinationLineInput,
)


def test_disclination_line_classifies_segment_loop_and_periodic_crossing():
    segment = DisclinationLine(
        defect_indices=[[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [2.5, 0.5, 0.0]]
    )
    assert segment.kind == "seg"
    assert len(segment) == 3

    loop = DisclinationLine(
        defect_indices=[
            [0.5, 0.5, 0.0],
            [1.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    assert loop.kind == "loop"
    assert len(loop) == 2

    cross = DisclinationLine(
        defect_indices=[[9.5, 0.5, 0.0], [10.5, 0.5, 0.0], [19.5, 0.5, 0.0]],
        box_size_periodic_index=(10, np.inf, np.inf),
    )
    assert cross.kind == "cross"
    assert len(cross) == 2


def test_disclination_line_input_validates_transform_data_and_keywords_override():
    input_value = DisclinationLineInput(
        defect_indices=[[0.5, 0.5, 0.0], [1.5, 0.5, 0.0]],
        grid_offset=(1.0, 2.0, 3.0),
    )
    line = DisclinationLine(input_value, grid_offset=(3.0, 2.0, 1.0))
    np.testing.assert_allclose(line.raw_grid_offset, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(
        line.calc_defect_coords,
        np.asarray(line.raw_defect_indices) + np.array([3.0, 2.0, 1.0]),
    )
