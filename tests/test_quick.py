import numpy as np
import pytest

import nematics3d.quick as quick


def test_quick_public_surface_is_intentional():
    assert quick.__all__ == ["quick_visualize_q"]


def test_auto_visual_params_scale_from_spatial_shape():
    params = quick._auto_quick_q_visual_params(np.zeros((128, 128, 128, 5)))
    assert params["smooth_min_line_length"] == 61
    assert params["smooth_window_length"] == 41
    assert params["visual_min_line_length"] == 75
    assert params["grid_origin"] == (64.0, 64.0, 64.0)
    assert params["grid_spacing"] == pytest.approx(2.5)


@pytest.mark.parametrize("field", [np.empty((3, 3)), np.empty((0, 3, 3, 3))])
def test_auto_visual_params_reject_invalid_spatial_shape(field):
    with pytest.raises(ValueError):
        quick._auto_quick_q_visual_params(field)


@pytest.mark.parametrize("normal", [(0, 0, 0), (1, 2), (1, np.nan, 0)])
def test_grid_normal_validation_rejects_invalid_values(normal):
    with pytest.raises(ValueError):
        quick._validate_grid_normal(normal)


def test_director_spacing_rejects_invalid_level():
    with pytest.raises(ValueError, match="director_spacing"):
        quick._resolve_director_spacing_level("very_sparse")


def test_quick_rejects_ambiguous_field_inputs():
    q = np.zeros((2, 2, 2, 5))
    n = np.zeros((2, 2, 2, 3))

    with pytest.raises(ValueError, match="either `Q`"):
        quick.quick_visualize_q(Q=q, n=n)
    with pytest.raises(ValueError, match="either `Q` or `n`"):
        quick.quick_visualize_q()
    with pytest.raises(ValueError, match="only be provided together"):
        quick.quick_visualize_q(S=np.ones((2, 2, 2)))


def test_off_screen_without_save_path_is_noop():
    q = np.zeros((2, 2, 2, 5))
    assert quick.quick_visualize_q(Q=q, is_off_screen=True) == (None, None)


def test_quick_delegates_to_qfield_workflow(monkeypatch, tmp_path):
    calls = []

    class FakeBounds:
        def act_visualize(self, **kwargs):
            calls.append(("bounds", kwargs))

    class FakeQObject:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.calc_bounds = FakeBounds()
            self.calc_defect_grid = np.array([[1.0, 2.0, 3.0]])

        def act_lines_smooth(self, **kwargs):
            calls.append(("smooth", kwargs))

        def act_visualize_disclination_lines(self, **kwargs):
            calls.append(("lines", kwargs))

        def act_visualize_n_plane(self, **kwargs):
            calls.append(("plane", kwargs))

    class FakeFigure:
        def __init__(self, **kwargs):
            calls.append(("figure", kwargs))

        def act_savefig(self, path):
            calls.append(("save", path))

    monkeypatch.setattr(quick, "QFieldObject", FakeQObject)
    monkeypatch.setattr(quick, "PlotFigure", FakeFigure)

    save_path = tmp_path / "nested" / "quick.png"
    n = np.zeros((8, 8, 8, 3))
    n[..., 0] = 1.0
    q_obj, figure = quick.quick_visualize_q(
        n=n,
        director_spacing="sparse",
        grid_normal=(1, 0, 0),
        save_path=save_path,
    )

    assert isinstance(q_obj, FakeQObject)
    assert isinstance(figure, FakeFigure)
    assert save_path.parent.is_dir()
    assert any(name == "smooth" for name, _ in calls)
    assert any(name == "lines" for name, _ in calls)
    plane_kwargs = next(kwargs for name, kwargs in calls if name == "plane")
    assert plane_kwargs["grid_normal"] == (1.0, 0.0, 0.0)
    assert plane_kwargs["grid_spacing"] > 0
    assert any(name == "save" for name, _ in calls)
