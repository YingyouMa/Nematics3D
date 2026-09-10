from pathlib import Path

import numpy as np
from PIL import Image

from nematics3d.visual.plot_figure import PlotFigure


def test_savefig_returns_screenshot_result_and_writes_expected_size(tmp_path):
    figure = PlotFigure(is_off_screen=True, size=(320, 240))
    path = tmp_path / "figure.png"

    result = figure.act_savefig(path, window_size=(200, 150), scale=2)

    assert path.is_file()
    assert isinstance(result, np.ndarray)
    assert result.shape[:2] == (300, 400)
    with Image.open(path) as image:
        assert image.size == (400, 300)


def test_savefig_default_size_uses_figure_opts(tmp_path):
    figure = PlotFigure(is_off_screen=True, size=(210, 130))
    path = tmp_path / "default-size.png"

    figure.act_savefig(path)

    with Image.open(path) as image:
        assert image.size == (210, 130)


def test_savefig_custom_size_does_not_mutate_figure_state(tmp_path):
    figure = PlotFigure(is_off_screen=True, size=(320, 240))
    original_opts_size = np.asarray(figure.opts.size).copy()
    original_plotter_size = tuple(figure.pl.window_size)

    figure.act_savefig(tmp_path / "custom-size.png", window_size=(180, 120))

    np.testing.assert_array_equal(figure.opts.size, original_opts_size)
    assert tuple(figure.pl.window_size) == original_plotter_size


def test_savefig_rejects_non_pixel_window_sizes(tmp_path):
    figure = PlotFigure(is_off_screen=True)
    path = tmp_path / "invalid.png"

    for window_size in ((200.5, 100), (0, 100), (-200, 100)):
        try:
            figure.act_savefig(path, window_size=window_size)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected invalid window_size {window_size!r} to fail"
            )

    assert not path.exists()


def test_savefig_accepts_pathlike_filename(tmp_path):
    figure = PlotFigure(is_off_screen=True, size=(120, 80))
    path = Path(tmp_path) / "pathlike.png"

    figure.act_savefig(path)

    assert path.is_file()
