import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def plot_custom_quivers_glyph3dmapper(positions, directions, lengths, radii, colors, geom_type="rod"):
    pos = np.atleast_2d(positions).astype(np.float32)
    orient = np.atleast_2d(directions).astype(np.float32)
    scale_vec = np.column_stack((lengths, 2.0 * radii, 2.0 * radii)).astype(np.float32)

    # ---- input polydata ----
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(pos, deep=True))
    poly.SetPoints(pts)

    pd = poly.GetPointData()

    arr_orient = numpy_to_vtk(orient, deep=True)
    arr_orient.SetName("orient_vectors")
    pd.AddArray(arr_orient)

    arr_scale = numpy_to_vtk(scale_vec, deep=True)
    arr_scale.SetName("scale_vectors")
    pd.AddArray(arr_scale)

    arr_colors = numpy_to_vtk(colors, deep=True)
    arr_colors.SetName("colors")
    pd.AddArray(arr_colors)

    # ---- source geometry (unit, centered, +X axis) ----
    if geom_type == "rod":
        src = pv.Cylinder(height=1.0, radius=0.5, direction=(1, 0, 0), resolution=20)
    elif geom_type == "grain":
        src = pv.Sphere(radius=0.5, theta_resolution=20, phi_resolution=20)
    else:
        raise ValueError(f"Unsupported geom_type: {geom_type!r}")

    src_vtk = src  # PyVista PolyData is vtkPolyData-compatible

    # ---- vtkGlyph3DMapper: separate arrays for orientation and scaling ----
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(poly)
    mapper.SetSourceData(src_vtk)

    # orientation: use direction vectors
    mapper.SetOrientationArray("orient_vectors")
    mapper.SetOrientationModeToDirection()

    # scaling: use 3-component scale vector (x,y,z)
    mapper.SetScaleArray("scale_vectors")
    mapper.SetScaleModeToScaleByVectorComponents()
    mapper.SetScaleFactor(1.0)

    # colors: direct RGB from point field data
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("colors")
    mapper.SetColorModeToDirectScalars()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # ---- render in PyVista ----
    pl = pv.Plotter()
    pl.add_actor(actor)
    pl.show_bounds(grid="front", location="outer")
    pl.add_axes()
    pl.reset_camera()
    pl.show(interactive_update=True)


# --- 测试 ---
pts = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
dirs = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])

lengths = np.array([0.5, 2.0, 1.0])
radii = np.array([0.4, 0.05, 0.2])

cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

plot_custom_quivers_glyph3dmapper(pts, dirs, lengths, radii, cols, geom_type="rod")
