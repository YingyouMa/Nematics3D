"""Compatibility checks for a vtkGlyph3DMapper-based PlotSphere backend.

This file is intentionally isolated under ``dev/benchmarks``.  It does not
modify the production PlotSphere implementation.  The goal is to establish
which current PlotSphere behaviors map naturally to an instanced mapper and
which behaviors still require a materialized fallback.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from nematics3d.classes.visual.plot_figure import PlotFigure


def _build_instanced_spheres(
    fig,
    coords,
    radius,
    *,
    rgba=None,
    scalars=None,
    sides=8,
):
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    radius = np.ascontiguousarray(radius, dtype=np.float32)

    points = vtk.vtkPoints()
    points_array = numpy_to_vtk(coords, deep=False)
    points.SetData(points_array)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    radius_array = numpy_to_vtk(radius, deep=False)
    radius_array.SetName("radius")
    poly.GetPointData().AddArray(radius_array)

    source = vtk.vtkSphereSource()
    source.SetRadius(1.0)
    source.SetThetaResolution(sides)
    source.SetPhiResolution(sides)

    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(poly)
    mapper.SetSourceConnection(source.GetOutputPort())
    mapper.SetScaleArray("radius")
    mapper.SetScaleModeToScaleByMagnitude()
    mapper.ScalingOn()

    rgba_array = None
    scalar_array = None
    if rgba is not None:
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        rgba_array = numpy_to_vtk(
            rgba,
            deep=False,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        rgba_array.SetName("rgba")
        rgba_array.SetNumberOfComponents(4)
        poly.GetPointData().AddArray(rgba_array)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("rgba")
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()

    if scalars is not None:
        scalars = np.ascontiguousarray(scalars, dtype=np.float32)
        scalar_array = numpy_to_vtk(scalars, deep=False)
        scalar_array.SetName("scalars")
        poly.GetPointData().AddArray(scalar_array)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("scalars")
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarRange(float(np.min(scalars)), float(np.max(scalars)))
        mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    fig.pl.renderer.AddActor(actor)
    return {
        "coords": coords,
        "radius": radius,
        "rgba": rgba,
        "scalars": scalars,
        "points": points,
        "points_array": points_array,
        "poly": poly,
        "radius_array": radius_array,
        "rgba_array": rgba_array,
        "scalar_array": scalar_array,
        "source": source,
        "mapper": mapper,
        "actor": actor,
    }


def test_per_instance_rgba_radius_and_transparency_render():
    fig = PlotFigure(is_off_screen=True, name="vtk_mapper_rgba_compat")
    try:
        coords = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        radius = np.array([0.2, 0.35, 0.5], dtype=np.float32)
        rgba = np.array(
            [[255, 0, 0, 64], [0, 255, 0, 160], [0, 0, 255, 255]],
            dtype=np.uint8,
        )
        state = _build_instanced_spheres(fig, coords, radius, rgba=rgba)
        fig.pl.reset_camera()
        image = fig.pl.screenshot(return_img=True)

        assert state["poly"].GetNumberOfPoints() == 3
        assert state["poly"].GetPointData().GetArray("radius").GetNumberOfTuples() == 3
        assert (
            state["poly"].GetPointData().GetArray("rgba").GetNumberOfComponents() == 4
        )
        assert state["mapper"].GetColorMode() == vtk.VTK_COLOR_MODE_DIRECT_SCALARS
        assert np.asarray(image).sum() > 0
    finally:
        fig.act_close()


def test_scalar_colormap_pipeline_and_scalar_bar_can_share_lookup_table():
    fig = PlotFigure(is_off_screen=True, name="vtk_mapper_scalar_compat")
    try:
        coords = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        radius = np.full(3, 0.35, dtype=np.float32)
        scalars = np.array([-2.0, 0.5, 4.0], dtype=np.float32)
        state = _build_instanced_spheres(fig, coords, radius, scalars=scalars)

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()
        state["mapper"].SetLookupTable(lut)
        state["mapper"].SetScalarRange(-2.0, 4.0)

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(state["mapper"].GetLookupTable())
        scalar_bar.SetTitle("test scalars")
        fig.pl.renderer.AddActor2D(scalar_bar)
        fig.pl.reset_camera()
        image = fig.pl.screenshot(return_img=True)

        assert state["mapper"].GetArrayName() == "scalars"
        assert state["mapper"].GetLookupTable() is lut
        assert scalar_bar.GetLookupTable() is lut
        assert tuple(state["mapper"].GetScalarRange()) == (-2.0, 4.0)
        assert np.asarray(image).sum() > 0
    finally:
        fig.act_close()


def test_center_clipping_preserves_pointwise_alignment():
    rng = np.random.default_rng(12)
    coords = rng.uniform(-2.0, 2.0, size=(50, 3)).astype(np.float32)
    radius = np.arange(50, dtype=np.float32) + 0.25
    rgba = np.column_stack(
        [
            np.arange(50, dtype=np.uint8),
            np.full(50, 20, dtype=np.uint8),
            np.full(50, 40, dtype=np.uint8),
            np.full(50, 255, dtype=np.uint8),
        ]
    )
    scalars = np.arange(50, dtype=np.float32) * 10.0

    keep = np.flatnonzero(np.all(np.abs(coords) <= 1.0, axis=1))
    clipped_coords = coords[keep]
    clipped_radius = radius[keep]
    clipped_rgba = rgba[keep]
    clipped_scalars = scalars[keep]

    # The first color channel and scalar deliberately encode original indices.
    assert np.array_equal(clipped_rgba[:, 0], keep.astype(np.uint8))
    assert np.array_equal(clipped_scalars / 10.0, keep.astype(np.float32))
    assert np.array_equal(clipped_radius - 0.25, keep.astype(np.float32))

    fig = PlotFigure(is_off_screen=True, name="vtk_mapper_clip_compat")
    try:
        state = _build_instanced_spheres(
            fig,
            clipped_coords,
            clipped_radius,
            rgba=clipped_rgba,
        )
        assert state["poly"].GetNumberOfPoints() == len(keep)
        assert state["radius_array"].GetNumberOfTuples() == len(keep)
        assert state["rgba_array"].GetNumberOfTuples() == len(keep)
    finally:
        fig.act_close()


def test_actor_lighting_and_pbr_properties_are_available():
    fig = PlotFigure(is_off_screen=True, name="vtk_mapper_property_compat")
    try:
        state = _build_instanced_spheres(
            fig,
            np.zeros((1, 3), dtype=np.float32),
            np.ones(1, dtype=np.float32),
            rgba=np.array([[200, 100, 50, 255]], dtype=np.uint8),
        )
        prop = state["actor"].GetProperty()
        prop.SetAmbient(0.2)
        prop.SetDiffuse(0.7)
        prop.SetSpecular(0.4)
        prop.SetSpecularPower(20.0)
        prop.SetInterpolationToPBR()
        prop.SetMetallic(0.3)
        prop.SetRoughness(0.6)

        assert np.isclose(prop.GetAmbient(), 0.2)
        assert np.isclose(prop.GetDiffuse(), 0.7)
        assert np.isclose(prop.GetSpecular(), 0.4)
        assert np.isclose(prop.GetSpecularPower(), 20.0)
        assert np.isclose(prop.GetMetallic(), 0.3)
        assert np.isclose(prop.GetRoughness(), 0.6)
    finally:
        fig.act_close()


def test_highlight_can_be_represented_by_a_separate_selected_instance_actor():
    fig = PlotFigure(is_off_screen=True, name="vtk_mapper_highlight_compat")
    try:
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        radius = np.array([0.4, 0.7], dtype=np.float32)
        rgba = np.array([[100, 100, 255, 255], [255, 100, 100, 255]], dtype=np.uint8)
        main = _build_instanced_spheres(fig, coords, radius, rgba=rgba)

        selected_index = 1
        highlight = _build_instanced_spheres(
            fig,
            coords[selected_index : selected_index + 1],
            radius[selected_index : selected_index + 1] * 1.08,
            rgba=np.array([[0, 0, 0, 255]], dtype=np.uint8),
            sides=8,
        )
        highlight["actor"].GetProperty().SetRepresentationToWireframe()
        highlight["actor"].GetProperty().SetLineWidth(5.0)

        assert main["poly"].GetNumberOfPoints() == 2
        assert highlight["poly"].GetNumberOfPoints() == 1
        assert highlight["actor"].GetProperty().GetRepresentation() == vtk.VTK_WIREFRAME
    finally:
        fig.act_close()


def test_materialized_final_mesh_is_not_present_in_mapper_input():
    fig = PlotFigure(is_off_screen=True, name="vtk_mapper_materialization_contract")
    try:
        coords = np.zeros((5, 3), dtype=np.float32)
        radius = np.ones(5, dtype=np.float32)
        state = _build_instanced_spheres(
            fig,
            coords,
            radius,
            rgba=np.tile(np.array([[255, 255, 255, 255]], dtype=np.uint8), (5, 1)),
        )
        state["source"].Update()

        # Mapper input is only the five instance centers; the rendered sphere
        # triangles live in the separate source and are not expanded here.
        assert state["poly"].GetNumberOfPoints() == 5
        assert state["poly"].GetNumberOfCells() == 0
        assert state["source"].GetOutput().GetNumberOfCells() > 0

        # Therefore mesh-space operations such as exact sphere clipping,
        # extract_surface(), or exporting the final combined polygon mesh need
        # a separate materialization/fallback path by design.
    finally:
        fig.act_close()
