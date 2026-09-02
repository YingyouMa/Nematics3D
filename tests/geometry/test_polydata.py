import numpy as np
import pytest
import pyvista as pv

from nematics3d.classes.visual.plot_polydata import (
    as_polydata_input as legacy_as_polydata_input,
)
from nematics3d.geometry.polydata import (
    as_polydata_input,
    copy_polydata_geometry,
)


def test_visual_polydata_converter_uses_canonical_geometry_implementation():
    assert legacy_as_polydata_input is as_polydata_input
    assert as_polydata_input.__module__ == "nematics3d.geometry.polydata"
    assert copy_polydata_geometry.__module__ == "nematics3d.geometry.polydata"


def test_as_polydata_input_returns_existing_polydata_without_copying():
    poly = pv.Plane()

    assert as_polydata_input(poly) is poly


def test_as_polydata_input_name_is_keyword_only():
    with pytest.raises(TypeError):
        as_polydata_input(pv.Plane(), "surface")


def test_copy_polydata_geometry_returns_independent_geometry_only_copy():
    poly = pv.Plane()
    poly.point_data["values"] = np.arange(poly.n_points)

    copied = copy_polydata_geometry(poly)

    assert copied is not poly
    assert copied.n_points == poly.n_points
    assert copied.n_cells == poly.n_cells
    assert list(copied.point_data.keys()) == []
    assert "values" in poly.point_data


def test_copy_polydata_geometry_rejects_non_polydata():
    with pytest.raises(TypeError, match="poly must be a pyvista.PolyData"):
        copy_polydata_geometry(object())
