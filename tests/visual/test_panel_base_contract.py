import math

import pytest
from qtpy import QtWidgets

from nematics3d.visual.qt.panel_base import (
    LogTickMapper,
    PressHoldButtonItem,
    make_labeled_slider_row,
)


@pytest.fixture(scope="module", autouse=True)
def qapplication():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_log_tick_mapper_round_trip_and_validation():
    mapper = LogTickMapper(value_min=0.01, value_max=100.0)
    for value in (0.01, 0.1, 1.0, 10.0, 100.0):
        tick = mapper.value_to_tick(value)
        recovered = mapper.tick_to_value(tick)
        assert math.isclose(recovered, value, rel_tol=0.02)

    with pytest.raises(ValueError):
        LogTickMapper(value_min=0.0, value_max=1.0)
    with pytest.raises(ValueError):
        LogTickMapper(value_min=1.0, value_max=1.0)
    with pytest.raises(ValueError):
        LogTickMapper(value_min=1.0, value_max=10.0, base=1.0)


def test_labeled_slider_clamps_and_can_expand_maximum():
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)

    clamped = make_labeled_slider_row(
        parent=parent,
        layout=layout,
        name="clamped",
        value_min=0,
        value_max=10,
        value_init=5,
    )
    clamped.set_tick(20)
    assert clamped.get_value() == 10.0

    expanding = make_labeled_slider_row(
        parent=parent,
        layout=layout,
        name="expanding",
        value_min=0,
        value_max=10,
        value_init=5,
        input_out_of_range="expand_max",
    )
    expanding.set_tick(20)
    assert expanding.get_value() == 20.0
    assert expanding.value_max == 20.0


def test_press_hold_button_requires_positive_timings():
    button = QtWidgets.QPushButton()
    with pytest.raises(ValueError, match="long_press_ms"):
        PressHoldButtonItem(button, lambda: None, long_press_ms=0)
    with pytest.raises(ValueError, match="repeat_ms"):
        PressHoldButtonItem(button, lambda: None, repeat_ms=0)
