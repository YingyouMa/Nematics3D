import sys
from pathlib import Path
import types
import unittest

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
PKG_DIR = SRC_DIR / "nematics3d"

sys.path.insert(0, str(SRC_DIR))

if "nematics3d" not in sys.modules:
    pkg = types.ModuleType("nematics3d")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["nematics3d"] = pkg

from nematics3d.classes import ContourSurface, ContourSurfaceSet


class TestContourSurface(unittest.TestCase):
    def test_set_builds_one_surface_per_unique_level(self):
        values = np.zeros((4, 5, 6), dtype=float)
        surface_set = ContourSurfaceSet(values, levels=[0.5, 1.0, 0.5], name="demo")

        self.assertEqual(surface_set.impl_init_levels, (0.5, 1.0))
        self.assertEqual(surface_set.calc_levels, (0.5, 1.0))
        self.assertEqual(surface_set.raw_values.shape, (4, 5, 6))
        self.assertEqual(len(surface_set), 2)
        self.assertIsInstance(surface_set[0], ContourSurface)
        self.assertEqual(surface_set[0].raw_level, 0.5)
        self.assertEqual(surface_set[1].raw_level, 1.0)

    def test_surface_extracts_and_caches_mesh(self):
        values = np.zeros((5, 6, 7), dtype=float)
        values[2, 3, 4] = 1.0
        surface_set = ContourSurfaceSet(values, levels=[0.5], name="spike")
        surface = surface_set[0]

        mesh = surface.act_extract()

        self.assertTrue(surface.is_extracted)
        self.assertIs(mesh, surface.mesh)
        self.assertGreater(mesh.n_points, 0)
        self.assertGreater(mesh.n_cells, 0)
        self.assertTrue(np.allclose(mesh.field_data["contour_level"], [0.5]))
        self.assertIs(mesh, surface.act_extract())

    def test_surface_level_update_syncs_container_levels_and_refreshes_mesh(self):
        values = np.zeros((5, 6, 7), dtype=float)
        values[2, 3, 4] = 1.0
        surface_set = ContourSurfaceSet(values, levels=[0.5], name="sync")
        surface = surface_set[0]
        mesh = surface.act_extract()

        self.assertIs(mesh, surface.mesh)

        surface.act_set_level(0.25)

        self.assertEqual(surface.raw_level, 0.25)
        self.assertEqual(surface_set.calc_levels, (0.25,))
        self.assertTrue(surface.is_extracted)
        self.assertIsNotNone(surface.mesh)
        self.assertIsNot(surface.mesh, mesh)
        self.assertTrue(np.allclose(surface.mesh.field_data["contour_level"], [0.25]))
        self.assertIn("0.25", surface.name)

    def test_extracted_mesh_points_follow_grid_transform_and_offset(self):
        values = np.zeros((5, 6, 7), dtype=float)
        values[2, 3, 4] = 1.0
        surface_set = ContourSurfaceSet(
            values,
            levels=[0.5],
            grid_offset=(10.0, 20.0, 30.0),
            grid_transform=np.diag((2.0, 3.0, 4.0)),
            name="world",
        )

        mesh = surface_set.act_extract_surface_by_level(0.5)
        center = np.asarray(mesh.points, dtype=float).mean(axis=0)

        self.assertTrue(np.allclose(center, (14.0, 29.0, 46.0)))

    def test_extract_all_returns_meshes_in_level_order(self):
        values = np.zeros((7, 7, 7), dtype=float)
        values[2, 3, 3] = 1.0
        values[4, 3, 3] = 2.0
        surface_set = ContourSurfaceSet(values, levels=[0.5, 1.5], name="multi")

        meshes = surface_set.act_extract_all()

        self.assertEqual(len(meshes), 2)
        self.assertTrue(all(mesh.n_points > 0 for mesh in meshes))
        self.assertTrue(surface_set[0].is_extracted)
        self.assertTrue(surface_set[1].is_extracted)


if __name__ == "__main__":
    unittest.main()
