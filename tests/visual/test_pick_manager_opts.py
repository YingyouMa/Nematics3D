import pytest
import vtk

from nematics3d.classes.visual.pick_manager import OptsPickManager


def test_pick_manager_integer_options_remain_integers():
    opts = OptsPickManager(
        marker_size=18,
        marker_font_size=16,
        slider_throttle_ms=25,
    )

    assert opts.marker_size == 18
    assert type(opts.marker_size) is int
    assert opts.marker_font_size == 16
    assert type(opts.marker_font_size) is int
    assert opts.slider_throttle_ms == 25
    assert type(opts.slider_throttle_ms) is int

    vtk.vtkTextProperty().SetFontSize(opts.marker_font_size)


@pytest.mark.parametrize("option_name", ["marker_size", "marker_font_size"])
def test_pick_manager_marker_integer_options_replace_fractional_values(option_name):
    opts = OptsPickManager(**{option_name: 1.5})

    assert getattr(opts, option_name) == 14
    assert type(getattr(opts, option_name)) is int


def test_pick_manager_slider_throttle_rejects_fractional_values():
    with pytest.raises(TypeError, match="must be an integer-valued finite number"):
        OptsPickManager(slider_throttle_ms=1.5)
