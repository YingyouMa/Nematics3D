"""Benchmark eager versus lazy PlotGlyph silhouette creation.

Run with pytest so it uses the project environment.  The benchmark generates
deterministic synthetic point clouds, rod orientations, and sampled surfaces,
then measures only glyph construction time (PlotFigure construction is outside
the timed interval).  Results are written beside this script under
``dev/benchmarks/results``.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np

from nematics3d.classes.visual.glyph import PlotGlyph
from nematics3d.classes.visual.plot_delaunay import PlotDelaunay
from nematics3d.classes.visual.plot_figure import PlotFigure
from nematics3d.classes.visual.plot_rod import PlotRod
from nematics3d.classes.visual.plot_sphere import PlotSphere


RESULT_DIR = Path(__file__).resolve().parent / "results"
RESULT_JSON = RESULT_DIR / "plot_glyph_lazy_silhouette_benchmark.json"
RESULT_MD = RESULT_DIR / "plot_glyph_lazy_silhouette_benchmark.md"


def _make_benchmark_data(n_target: int, seed: int):
    """Generate one shared sampled surface and one rod orientation per point."""
    side = max(3, int(round(np.sqrt(n_target))))
    u = np.linspace(-5.0, 5.0, side)
    x, y = np.meshgrid(u, u, indexing="xy")
    z = 0.65 * np.sin(0.8 * x) * np.cos(0.65 * y)
    coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    rng = np.random.default_rng(seed)
    orient = rng.normal(size=(len(coords), 3))
    orient /= np.linalg.norm(orient, axis=1, keepdims=True)
    return coords, orient


def _construct(kind: str, n: int, seed: int):
    fig = PlotFigure(is_off_screen=True, name=f"benchmark_{kind}_{n}")
    coords, orient = _make_benchmark_data(n, seed)

    try:
        gc.collect()
        start = time.perf_counter()
        if kind == "points":
            obj = PlotSphere(
                coords,
                figure=fig,
                radius=0.08,
                sides=8,
                is_reset_camera=False,
            )
        elif kind == "rods":
            obj = PlotRod(
                coords,
                orient,
                figure=fig,
                length=0.35,
                radius=0.035,
                sides=8,
                is_reset_camera=False,
            )
        elif kind == "surface":
            obj = PlotDelaunay(
                coords,
                figure=fig,
                color=(0.35, 0.55, 0.8),
                is_reset_camera=False,
            )
        elif kind == "combined":
            objects = [
                PlotSphere(
                    coords,
                    figure=fig,
                    radius=0.08,
                    sides=8,
                    is_reset_camera=False,
                ),
                PlotRod(
                    coords,
                    orient,
                    figure=fig,
                    length=0.35,
                    radius=0.035,
                    sides=8,
                    is_reset_camera=False,
                ),
                PlotDelaunay(
                    coords,
                    figure=fig,
                    color=(0.35, 0.55, 0.8),
                    is_reset_camera=False,
                ),
            ]
            elapsed = time.perf_counter() - start
            meshes = [obj.entity_actor.mapper.dataset for obj in objects]
            return {
                "seconds": elapsed,
                "input_points": int(len(coords)),
                "mesh_points": int(sum(mesh.n_points for mesh in meshes)),
                "mesh_cells": int(sum(mesh.n_cells for mesh in meshes)),
                "silhouette_created": any(
                    obj.entity_silhouette is not None for obj in objects
                ),
            }
        else:
            raise ValueError(kind)
        elapsed = time.perf_counter() - start
        mesh = obj.entity_actor.mapper.dataset
        stats = {
            "seconds": elapsed,
            "input_points": int(len(coords)),
            "mesh_points": int(mesh.n_points),
            "mesh_cells": int(mesh.n_cells),
            "silhouette_created": obj.entity_silhouette is not None,
        }
        return stats
    finally:
        fig.act_close()


class _EagerSilhouetteMode:
    """Restore the pre-lazy initial-construction behavior for benchmarking."""

    def __enter__(self):
        self._make_figure = PlotGlyph._helper_make_figure

        def eager_make_figure(glyph, logger=None):
            self._make_figure(glyph, logger=logger)
            if (
                glyph.state_is_silhouette
                and getattr(glyph, "entity_actor", None) is not None
                and not glyph.calc_is_empty
            ):
                glyph._helper_add_silhouette()

        PlotGlyph._helper_make_figure = eager_make_figure
        return self

    def __exit__(self, exc_type, exc, tb):
        PlotGlyph._helper_make_figure = self._make_figure


def _run_one(mode: str, kind: str, n: int, seed: int):
    if mode == "lazy":
        return _construct(kind, n, seed)
    if mode == "eager":
        with _EagerSilhouetteMode():
            return _construct(kind, n, seed)
    raise ValueError(mode)


def _summarize(samples):
    eager = [row["seconds"] for row in samples if row["mode"] == "eager"]
    lazy = [row["seconds"] for row in samples if row["mode"] == "lazy"]
    eager_median = statistics.median(eager)
    lazy_median = statistics.median(lazy)
    return {
        "eager_median_seconds": eager_median,
        "lazy_median_seconds": lazy_median,
        "speedup_x": eager_median / lazy_median,
        "time_saved_percent": 100.0 * (eager_median - lazy_median) / eager_median,
        "eager_samples_seconds": eager,
        "lazy_samples_seconds": lazy,
    }


def test_plot_glyph_lazy_silhouette_benchmark():
    sizes = [500, 2000, 5000]
    kinds = ["points", "rods", "surface", "combined"]
    repeats = 5
    rows = []

    # One unrecorded warm-up for VTK/PyVista initialization.
    _run_one("lazy", "points", 50, 1)

    for kind in kinds:
        for n in sizes:
            for rep in range(repeats):
                # Alternate order each repeat to suppress systematic cache/order bias.
                modes = ("eager", "lazy") if rep % 2 == 0 else ("lazy", "eager")
                for mode in modes:
                    stats = _run_one(mode, kind, n, seed=1000 + 17 * rep + n)
                    rows.append(
                        {
                            "kind": kind,
                            "requested_n": n,
                            "repeat": rep,
                            "mode": mode,
                            **stats,
                        }
                    )

    summary = {}
    for kind in kinds:
        summary[kind] = {}
        for n in sizes:
            samples = [
                row for row in rows if row["kind"] == kind and row["requested_n"] == n
            ]
            summary[kind][str(n)] = _summarize(samples)

    payload = {
        "description": "Synthetic PlotGlyph construction benchmark: eager vs lazy silhouette",
        "repeats": repeats,
        "sizes": sizes,
        "summary": summary,
        "samples": rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# PlotGlyph lazy-silhouette benchmark",
        "",
        f"Each condition uses {repeats} timed repeats; figure construction is excluded.",
        "",
        "| object | N | eager median (s) | lazy median (s) | speedup | time saved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in kinds:
        for n in sizes:
            s = summary[kind][str(n)]
            lines.append(
                f"| {kind} | {n} | {s['eager_median_seconds']:.6f} | "
                f"{s['lazy_median_seconds']:.6f} | {s['speedup_x']:.3f}x | "
                f"{s['time_saved_percent']:.1f}% |"
            )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # This is a measurement, not a performance threshold test.
    assert len(rows) == len(kinds) * len(sizes) * repeats * 2
