from pathlib import Path
import sys
import types

import numpy as np

PKG_DIR = Path(__file__).resolve().parents[2] / "src" / "nematics3d"
if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.q_field.q_field_object import QFieldObject  # noqa: E402
from nematics3d.sample.defect_section import DefectSectionGrid  # noqa: E402
from nematics3d.sample.plane_grid_polar import PlaneGridPolar  # noqa: E402


def _make_smooth():
    data_path = (
        Path(__file__).resolve().parents[1] / "disclination" / "beta" / "Q_1630.npy"
    )
    q_data = np.load(data_path)[0][168:185, 5:32, 10:35]
    q = QFieldObject(Q=q_data, name="defect-section-test")
    q.act_lines_smooth(window_length=28)
    return q, q.lines[0].smooth


def test_cross_section_builds_canonical_polar_grid_wrapper():
    q, smooth = _make_smooth()
    section = smooth.act_cross_section(25.0, dr=0.4, layers=6)

    assert isinstance(section, DefectSectionGrid)
    assert isinstance(section.wrapped, PlaneGridPolar)
    assert section.wrapped.wrapper is section
    assert section.owner is smooth
    assert section.opts.u_percent == 25.0
    assert np.allclose(section.calc_normal, section.wrapped.opts.normal)
    assert q is not None


def test_cross_section_commit_updates_wrapped_pose():
    q, smooth = _make_smooth()
    section = smooth.act_cross_section(20.0, dr=0.5, layers=4)
    origin_old = np.asarray(section.wrapped.opts.origin).copy()

    section.act_commit(u_percent=60.0)

    assert not np.allclose(section.wrapped.opts.origin, origin_old)
    assert np.allclose(section.calc_normal, section.wrapped.opts.normal)
    assert q is not None


def test_normal_registry_keeps_tangent_builtin_separate():
    q, smooth = _make_smooth()
    section = smooth.act_cross_section(50.0)

    assert "tangent" not in section.impl_normals
    section.act_register_normal("x", (1.0, 0.0, 0.0))
    section.act_commit(state_normal="x")

    assert np.allclose(section.calc_normal, (1.0, 0.0, 0.0))
    assert np.allclose(section.wrapped.opts.normal, (1.0, 0.0, 0.0))
    assert q is not None
