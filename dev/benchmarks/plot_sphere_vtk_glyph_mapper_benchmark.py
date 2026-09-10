"""A/B benchmark: current PlotSphere versus raw vtkGlyph3DMapper prototype.

This benchmark intentionally leaves production code unchanged.  It generates
deterministic per-point positions, radii, RGBA colors, and update arrays, then
compares initial construction plus radius/color/coords live updates.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from nematics3d.visual.plot_figure import PlotFigure
from nematics3d.visual.plot_sphere import PlotSphere


RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_JSON = RESULT_DIR / "plot_sphere_vtk_glyph_mapper_benchmark.json"
RESULT_MD = RESULT_DIR / "plot_sphere_vtk_glyph_mapper_benchmark.md"


def _make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    radius = rng.uniform(0.04, 0.18, size=n).astype(np.float32)
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, :3] = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    rgba[:, 3] = rng.integers(96, 256, size=n, dtype=np.uint8)

    coords2 = (coords + rng.normal(0.0, 0.03, size=coords.shape)).astype(np.float32)
    radius2 = (radius * rng.uniform(0.75, 1.25, size=n)).astype(np.float32)
    rgba2 = rgba.copy()
    rgba2[:, :3] = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    return coords, radius, rgba, coords2, radius2, rgba2


class VtkGlyphSpherePrototype:
    def __init__(self, fig, coords, radius, rgba, sides=8):
        self.fig = fig
        self.coords = np.ascontiguousarray(coords, dtype=np.float32)
        self.radius = np.ascontiguousarray(radius, dtype=np.float32)
        self.rgba = np.ascontiguousarray(rgba, dtype=np.uint8)

        self.points = vtk.vtkPoints()
        self.points_array = numpy_to_vtk(self.coords, deep=False)
        self.points.SetData(self.points_array)

        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.points)

        self.radius_array = numpy_to_vtk(self.radius, deep=False)
        self.radius_array.SetName("radius")
        self.poly.GetPointData().AddArray(self.radius_array)
        self.poly.GetPointData().SetActiveScalars("radius")

        self.rgba_array = numpy_to_vtk(
            self.rgba.reshape(-1, 4), deep=False, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        self.rgba_array.SetName("rgba")
        self.rgba_array.SetNumberOfComponents(4)
        self.poly.GetPointData().AddArray(self.rgba_array)

        self.source = vtk.vtkSphereSource()
        self.source.SetRadius(1.0)
        self.source.SetThetaResolution(sides)
        self.source.SetPhiResolution(sides)
        self.source.Update()

        self.mapper = vtk.vtkGlyph3DMapper()
        self.mapper.SetInputData(self.poly)
        self.mapper.SetSourceConnection(self.source.GetOutputPort())
        self.mapper.SetScaleArray("radius")
        self.mapper.SetScaleModeToScaleByMagnitude()
        self.mapper.ScalingOn()
        self.mapper.SetScalarModeToUsePointFieldData()
        self.mapper.SelectColorArray("rgba")
        self.mapper.SetColorModeToDirectScalars()
        self.mapper.ScalarVisibilityOn()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.fig.pl.renderer.AddActor(self.actor)
        self.fig.pl.render()

    def update_radius(self, radius):
        self.radius[...] = radius
        self.radius_array.Modified()
        self.poly.Modified()
        self.fig.pl.render()

    def update_color(self, rgba):
        self.rgba[...] = rgba
        self.rgba_array.Modified()
        self.poly.Modified()
        self.fig.pl.render()

    def update_coords(self, coords):
        self.coords[...] = coords
        self.points_array.Modified()
        self.points.Modified()
        self.poly.Modified()
        self.fig.pl.render()


def _construct_current(fig, coords, radius, rgba):
    return PlotSphere(
        coords,
        figure=fig,
        radius=radius,
        color=rgba[:, :3].astype(np.float32) / 255.0,
        opacity=rgba[:, 3].astype(np.float32) / 255.0,
        sides=8,
        is_reset_camera=False,
    )


def _run_current(n, seed):
    data = _make_data(n, seed)
    coords, radius, rgba, coords2, radius2, rgba2 = data
    fig = PlotFigure(is_off_screen=True, name=f"current_{n}_{seed}")
    try:
        gc.collect()
        t0 = time.perf_counter()
        obj = _construct_current(fig, coords, radius, rgba)
        t_construct = time.perf_counter() - t0

        t0 = time.perf_counter()
        obj.opts.radius = radius2
        t_radius = time.perf_counter() - t0

        t0 = time.perf_counter()
        obj.act_commit(
            color=rgba2[:, :3].astype(np.float32) / 255.0,
            opacity=rgba2[:, 3].astype(np.float32) / 255.0,
        )
        t_color = time.perf_counter() - t0

        t0 = time.perf_counter()
        obj.coords = coords2
        t_coords = time.perf_counter() - t0
        mesh = obj.entity_actor.mapper.dataset
        t0 = time.perf_counter()
        image = fig.pl.screenshot(return_img=True)
        t_screenshot = time.perf_counter() - t0
        return {
            "construct": t_construct,
            "radius_update": t_radius,
            "color_update": t_color,
            "coords_update": t_coords,
            "screenshot": t_screenshot,
            "image_checksum": int(np.asarray(image, dtype=np.uint64).sum()),
            "mesh_points": int(mesh.n_points),
            "mesh_cells": int(mesh.n_cells),
        }
    finally:
        fig.act_close()


def _run_vtk(n, seed):
    data = _make_data(n, seed)
    coords, radius, rgba, coords2, radius2, rgba2 = data
    fig = PlotFigure(is_off_screen=True, name=f"vtk_{n}_{seed}")
    try:
        gc.collect()
        t0 = time.perf_counter()
        obj = VtkGlyphSpherePrototype(fig, coords, radius, rgba, sides=8)
        t_construct = time.perf_counter() - t0

        t0 = time.perf_counter()
        obj.update_radius(radius2)
        t_radius = time.perf_counter() - t0

        t0 = time.perf_counter()
        obj.update_color(rgba2)
        t_color = time.perf_counter() - t0

        t0 = time.perf_counter()
        obj.update_coords(coords2)
        t_coords = time.perf_counter() - t0
        t0 = time.perf_counter()
        image = fig.pl.screenshot(return_img=True)
        t_screenshot = time.perf_counter() - t0
        return {
            "construct": t_construct,
            "radius_update": t_radius,
            "color_update": t_color,
            "coords_update": t_coords,
            "screenshot": t_screenshot,
            "image_checksum": int(np.asarray(image, dtype=np.uint64).sum()),
            "mesh_points": int(n),
            "mesh_cells": 0,
        }
    finally:
        fig.act_close()


def _summary(rows, n, metric):
    current = [r[metric] for r in rows if r["n"] == n and r["mode"] == "current"]
    vtk_rows = [r[metric] for r in rows if r["n"] == n and r["mode"] == "vtk"]
    a = statistics.median(current)
    b = statistics.median(vtk_rows)
    return {
        "current_median_seconds": a,
        "vtk_median_seconds": b,
        "speedup_x": a / b,
        "time_saved_percent": 100.0 * (a - b) / a,
    }


def test_plot_sphere_vtk_glyph_mapper_benchmark():
    sizes = [500, 2000, 5000, 10000, 50000]
    repeats = 5
    rows = []

    _run_vtk(50, 1)
    _run_current(50, 1)

    for n in sizes:
        for rep in range(repeats):
            order = ("current", "vtk") if rep % 2 == 0 else ("vtk", "current")
            for mode in order:
                seed = 10000 + n + rep * 31
                stats = (
                    _run_current(n, seed) if mode == "current" else _run_vtk(n, seed)
                )
                rows.append({"n": n, "repeat": rep, "mode": mode, **stats})

    metrics = [
        "construct",
        "radius_update",
        "color_update",
        "coords_update",
        "screenshot",
    ]
    summary = {str(n): {m: _summary(rows, n, m) for m in metrics} for n in sizes}
    payload = {"sizes": sizes, "repeats": repeats, "summary": summary, "samples": rows}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# PlotSphere vtkGlyph3DMapper prototype benchmark",
        "",
        "Production code is unchanged. Values are medians of 5 interleaved repeats.",
        "",
        "| N | metric | current (s) | vtk mapper (s) | speedup | time saved |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for n in sizes:
        for metric in metrics:
            s = summary[str(n)][metric]
            lines.append(
                f"| {n} | {metric} | {s['current_median_seconds']:.6f} | "
                f"{s['vtk_median_seconds']:.6f} | {s['speedup_x']:.2f}x | "
                f"{s['time_saved_percent']:.1f}% |"
            )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert len(rows) == len(sizes) * repeats * 2
