from nematics3d.datatypes import UNSET
from nematics3d.classes.visual.plot_figure import OptsFigure


def test_opts_finalize_allow_unset_skips_unset_validators():
    opts = OptsFigure()

    opts.act_finalize(is_allow_unset=True)

    assert opts.azimuth is UNSET
    assert opts.elevation is UNSET
    assert opts.roll is UNSET
    assert opts.distance is UNSET
    assert opts.focal_point is UNSET
    assert tuple(opts.size) == (1900, 1000)
    assert tuple(opts.bg_color) == (1, 1, 1)
