"""Normalization and geometry-only copying helpers for PolyData inputs."""

from __future__ import annotations

import pyvista as pv
import vtk


def as_polydata_input(data, *, name: str = "polydata input") -> pv.PolyData:
    """
    Normalize supported mesh-like inputs to ``pyvista.PolyData``.

    Supported inputs currently include:

    - ``pyvista.PolyData``
    - ``vtk.vtkPolyData``
    - surface-like ``pyvista.DataSet`` / ``vtk.vtkDataSet`` objects that can
      be converted through ``extract_surface()`` or ``cast_to_polydata()``

    If ``data`` is already ``pyvista.PolyData``, the original object is
    returned without copying. Call :func:`copy_polydata_geometry` when an
    independent, geometry-only deep copy is required.
    """

    if isinstance(data, pv.PolyData):
        return data

    if isinstance(data, vtk.vtkPolyData):
        wrapped = pv.wrap(data)
    elif isinstance(data, pv.DataSet):
        wrapped = data
    elif isinstance(data, vtk.vtkDataSet):
        wrapped = pv.wrap(data)
    else:
        raise TypeError(
            f"{name} must be a pyvista.PolyData, vtkPolyData, or another "
            f"surface-like PyVista/VTK dataset. Got {type(data).__name__}."
        )

    if isinstance(wrapped, pv.PolyData):
        return wrapped

    last_error: Exception | None = None
    for method_name in ("extract_surface", "cast_to_polydata"):
        method = getattr(wrapped, method_name, None)
        if method is None:
            continue
        try:
            candidate = method()
        except Exception as error:
            last_error = error
            continue
        if isinstance(candidate, pv.PolyData):
            return candidate
        if candidate is not None:
            candidate = pv.wrap(candidate)
            if isinstance(candidate, pv.PolyData):
                return candidate

    error = TypeError(
        f"{name} with type {type(data).__name__} could not be converted to "
        "pyvista.PolyData."
    )
    if last_error is not None:
        raise error from last_error
    raise error


def copy_polydata_geometry(poly: pv.PolyData) -> pv.PolyData:
    """
    Return an independent PolyData deep copy containing only geometry/topology.

    All attached point, cell, and field arrays are removed from the returned
    copy. This function does not run ``pyvista.PolyData.clean``: it does not
    merge duplicate points, remove unused points, or repair topology.
    """

    if not isinstance(poly, pv.PolyData):
        raise TypeError(
            "poly must be a pyvista.PolyData; normalize other supported "
            f"dataset types with as_polydata_input first. Got {type(poly).__name__}."
        )

    poly_clean = poly.copy(deep=True)
    for attr_name in ("point_data", "cell_data", "field_data"):
        data_attr = getattr(poly_clean, attr_name, None)
        if data_attr is None:
            continue
        clear_method = getattr(data_attr, "clear", None)
        if callable(clear_method):
            clear_method()
    return poly_clean


__all__ = [
    "as_polydata_input",
    "copy_polydata_geometry",
]
