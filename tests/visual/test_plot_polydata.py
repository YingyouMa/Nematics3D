import sys
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
import vtk

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes.visual import OptsPolyData as ExportedOptsPolyData
from nematics3d.classes.visual import PlotPolyData as ExportedPlotPolyData
from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_polydata import OptsPolyData, PlotPolyData
from nematics3d.classes.visual.qt.interact_polydata import InteractPolyData


class TestPlotPolyData(unittest.TestCase):
    def _make_figure(self, *, is_off_screen=True):
        return PlotFigure(is_off_screen=is_off_screen)

    def test_visual_subpackage_exports_polydata_classes(self):
        self.assertIs(ExportedOptsPolyData, OptsPolyData)
        self.assertIs(ExportedPlotPolyData, PlotPolyData)

    def test_plot_polydata_accepts_pyvista_polydata(self):
        fig = self._make_figure()
        try:
            poly = pv.Cube().extract_surface().triangulate().clean()
            mesh = PlotPolyData(
                poly,
                figure=fig,
                color=(0.2, 0.4, 0.8),
                opacity=0.6,
            )

            self.assertIsInstance(mesh.raw_poly, pv.PolyData)
            self.assertEqual(mesh.raw_coords.shape[1], 3)
            self.assertFalse(mesh.calc_is_empty)
            self.assertIsNotNone(mesh.entity_actor)
        finally:
            fig.act_close()

    def test_plot_polydata_strips_input_data_from_internal_template_only(self):
        fig = self._make_figure()
        try:
            poly = pv.Plane(i_resolution=1, j_resolution=1).triangulate().clean()
            poly.point_data["source_point"] = np.arange(poly.n_points, dtype=np.float32)
            poly.cell_data["source_cell"] = np.arange(poly.n_cells, dtype=np.float32)
            poly.field_data["source_field"] = np.array([1.0], dtype=np.float32)

            mesh = PlotPolyData(poly, figure=fig, opacity=0.5)

            self.assertIn("source_point", poly.point_data)
            self.assertIn("source_cell", poly.cell_data)
            self.assertIn("source_field", poly.field_data)

            self.assertEqual(len(mesh.raw_poly.point_data), 0)
            self.assertEqual(len(mesh.raw_poly.cell_data), 0)
            self.assertEqual(len(mesh.raw_poly.field_data), 0)
        finally:
            fig.act_close()

    def test_plot_polydata_core_geometry_inputs_are_protected(self):
        fig = self._make_figure()
        try:
            poly = pv.Plane(i_resolution=1, j_resolution=1).triangulate().clean()
            mesh = PlotPolyData(poly, figure=fig)

            with self.assertRaisesRegex(AttributeError, "protected"):
                mesh.coords = np.zeros_like(mesh.raw_coords)

            with self.assertRaisesRegex(AttributeError, "protected"):
                mesh.raw_coords = np.zeros_like(mesh.raw_coords)

            with self.assertRaisesRegex(AttributeError, "protected"):
                mesh.poly = pv.Sphere()

            with self.assertRaisesRegex(AttributeError, "protected"):
                mesh.raw_poly = pv.Sphere()
        finally:
            fig.act_close()

    def test_plot_polydata_accepts_vtk_polydata(self):
        fig = self._make_figure()
        try:
            source_poly = pv.Sphere(theta_resolution=8, phi_resolution=8)
            vtk_polydata = vtk.vtkPolyData()
            vtk_polydata.DeepCopy(source_poly)

            mesh = PlotPolyData(vtk_polydata, figure=fig, opacity=0.8)

            self.assertIsInstance(mesh.raw_poly, pv.PolyData)
            self.assertGreater(mesh.entity_actor.mapper.dataset.n_points, 0)
        finally:
            fig.act_close()

    def test_plot_polydata_accepts_dataset_convertible_to_surface(self):
        fig = self._make_figure()
        try:
            image = pv.ImageData(dimensions=(2, 2, 2))
            mesh = PlotPolyData(image, figure=fig, color=(0.7, 0.3, 0.2))

            self.assertIsInstance(mesh.raw_poly, pv.PolyData)
            self.assertGreater(mesh.entity_actor.mapper.dataset.n_cells, 0)
        finally:
            fig.act_close()

    def test_plot_polydata_mesh_has_visual_arrays(self):
        fig = self._make_figure()
        try:
            poly = pv.Plane(i_resolution=1, j_resolution=1).triangulate().clean()
            mesh = PlotPolyData(
                poly,
                figure=fig,
                color=lambda pts: np.column_stack(
                    [
                        np.clip(pts[:, 0] + 0.5, 0.0, 1.0),
                        np.full(len(pts), 0.25),
                        np.clip(pts[:, 1] + 0.5, 0.0, 1.0),
                    ]
                ),
                opacity=lambda pts: np.linspace(0.3, 0.9, len(pts)),
                scalars=lambda pts: pts[:, 0] + pts[:, 1],
                paint_by="scalars",
            )

            dataset = mesh.entity_actor.mapper.dataset
            self.assertIn("rgba", dataset.point_data)
            self.assertIn("opacity", dataset.point_data)
            self.assertIn("scalars", dataset.point_data)
            self.assertEqual(dataset.point_data["rgba"].shape[1], 4)
        finally:
            fig.act_close()

    def test_polydata_opts_actor_attrs_update_live(self):
        fig = self._make_figure()
        try:
            poly = pv.Plane(i_resolution=1, j_resolution=1).triangulate().clean()
            mesh = PlotPolyData(poly, figure=fig)

            mesh.opts.is_show_edges = True
            mesh.opts.edge_color = (1.0, 0.0, 0.0)
            mesh.opts.line_width = 3.0
            mesh.opts.style = "wireframe"

            actor_prop = mesh.entity_actor.prop
            self.assertTrue(actor_prop.show_edges)
            self.assertEqual(tuple(actor_prop.edge_color)[:3], (1.0, 0.0, 0.0))
            self.assertAlmostEqual(float(actor_prop.line_width), 3.0)
            self.assertEqual(actor_prop.style, "Wireframe")
        finally:
            fig.act_close()

    def test_plot_polydata_interact_func_opens_polydata_panel(self):
        fig = self._make_figure()
        try:
            poly = pv.Plane(i_resolution=1, j_resolution=1).triangulate().clean()
            mesh = PlotPolyData(poly, figure=fig)

            sentinel = object()
            with patch.object(
                InteractPolyData,
                "show_once",
                return_value=sentinel,
            ) as mocked_show_once:
                panel = mesh.impl_interact_func()

            self.assertIs(panel, sentinel)
            mocked_show_once.assert_called_once_with(mesh, mesh.fig)
        finally:
            fig.act_close()

    def test_plot_polydata_rejects_unsupported_input(self):
        fig = self._make_figure()
        try:
            with self.assertRaisesRegex(TypeError, "must be a pyvista.PolyData"):
                PlotPolyData([1, 2, 3], figure=fig)
        finally:
            fig.act_close()


if __name__ == "__main__":
    unittest.main()
