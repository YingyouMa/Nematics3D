"""Exploratory benchmark for a specialized uniform-grid trilinear backend.

This is intentionally kept under dev/onGoing.  It compares the current
RegularGridInterpolator-style path against two narrower implementations:

1. a direct NumPy trilinear evaluator, and
2. scipy.ndimage.map_coordinates(order=1) as a compiled proxy for the kind of
   uniform-grid kernel a future Nematics3D C backend could implement.

The benchmark writes RESULTS.md so timings remain inspectable after pytest.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates


HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260909)
GRID_SHAPE = (72, 68, 64)
QUERY_COUNTS = (100_000, 500_000)
REPEATS = 3
PERIODIC = np.array((True, False, True), dtype=bool)


def _time_call(func, *args):
    times = []
    result = None
    for _ in range(REPEATS):
        start = perf_counter()
        result = func(*args)
        times.append(perf_counter() - start)
    return min(times), result


def _make_points(count):
    upper = np.asarray(GRID_SHAPE, dtype=float) - 1.000001
    return RNG.random((count, 3)) * upper


def _numpy_trilinear(values, points):
    base = np.floor(points).astype(np.intp)
    frac = points - base
    nxt = base + 1

    x0, y0, z0 = base.T
    x1, y1, z1 = nxt.T
    tx, ty, tz = frac.T

    if values.ndim == 3:
        result = np.zeros(len(points), dtype=float)
    else:
        result = np.zeros((len(points), values.shape[3]), dtype=float)
        tx = tx[:, None]
        ty = ty[:, None]
        tz = tz[:, None]

    wx0, wx1 = 1.0 - tx, tx
    wy0, wy1 = 1.0 - ty, ty
    wz0, wz1 = 1.0 - tz, tz

    result += values[x0, y0, z0] * wx0 * wy0 * wz0
    result += values[x1, y0, z0] * wx1 * wy0 * wz0
    result += values[x0, y1, z0] * wx0 * wy1 * wz0
    result += values[x1, y1, z0] * wx1 * wy1 * wz0
    result += values[x0, y0, z1] * wx0 * wy0 * wz1
    result += values[x1, y0, z1] * wx1 * wy0 * wz1
    result += values[x0, y1, z1] * wx0 * wy1 * wz1
    result += values[x1, y1, z1] * wx1 * wy1 * wz1
    return result


def _map_coordinates_linear(values, points):
    coords = points.T
    if values.ndim == 3:
        return map_coordinates(values, coords, order=1, mode="nearest", prefilter=False)
    return np.stack(
        [
            map_coordinates(
                values[..., component],
                coords,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            for component in range(values.shape[3])
        ],
        axis=-1,
    )


def _rgi(values):
    axes = tuple(np.arange(size, dtype=float) for size in values.shape[:3])
    return RegularGridInterpolator(axes, values, method="linear", bounds_error=True)


def _periodic_extended(values):
    extended = values
    axes = []
    for axis, (size, periodic) in enumerate(zip(GRID_SHAPE, PERIODIC)):
        if periodic:
            axes.append(np.arange(size + 1, dtype=float))
            extended = np.concatenate(
                (extended, np.take(extended, [0], axis=axis)), axis=axis
            )
        else:
            axes.append(np.arange(size, dtype=float))
    return tuple(axes), extended


def _prepare_mixed_boundary_points(points):
    pts = points.copy()
    out_mask = np.zeros(len(pts), dtype=bool)
    for axis, periodic in enumerate(PERIODIC):
        if not periodic:
            out_mask |= (pts[:, axis] < 0.0) | (pts[:, axis] > GRID_SHAPE[axis] - 1)
    for axis, periodic in enumerate(PERIODIC):
        if periodic:
            pts[:, axis] = np.mod(pts[:, axis], GRID_SHAPE[axis])
        else:
            pts[:, axis] = np.clip(pts[:, axis], 0.0, GRID_SHAPE[axis] - 1)
    return pts, out_mask


def _benchmark_case(values, points):
    rgi = _rgi(values)
    # Warm up compiled paths and caches.
    rgi(points[:32])
    _numpy_trilinear(values, points[:32])
    _map_coordinates_linear(values, points[:32])

    t_rgi, out_rgi = _time_call(rgi, points)
    t_numpy, out_numpy = _time_call(_numpy_trilinear, values, points)
    t_map, out_map = _time_call(_map_coordinates_linear, values, points)

    np.testing.assert_allclose(out_numpy, out_rgi, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(out_map, out_rgi, rtol=2e-12, atol=2e-12)
    return t_rgi, t_numpy, t_map


def _benchmark_validity(values, mask, points):
    value_rgi = _rgi(values)
    mask_float = mask.astype(float)
    mask_rgi = _rgi(mask_float)

    def current_style():
        interpolated = value_rgi(points)
        validity = mask_rgi(points) >= 1.0 - 1e-9
        return interpolated, validity

    def compiled_proxy_style():
        interpolated = _map_coordinates_linear(values, points)
        # A production fused C kernel could reuse the already-computed cell
        # indices and trilinear weights here.  map_coordinates is used for the
        # benchmark proxy so the current >= 1 - 1e-9 semantics are reproduced
        # exactly, including vanishingly small support weights near cell faces.
        support = _map_coordinates_linear(mask_float, points)
        validity = support >= 1.0 - 1e-9
        return interpolated, validity

    current_style()
    compiled_proxy_style()
    t_current, current = _time_call(current_style)
    t_proxy, proxy = _time_call(compiled_proxy_style)
    np.testing.assert_allclose(proxy[0], current[0], rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(proxy[1], current[1])
    return t_current, t_proxy


def _benchmark_full_mixed_boundary(values, mask, count):
    axes, values_extended = _periodic_extended(values)
    _, mask_extended = _periodic_extended(mask.astype(float))
    value_rgi = RegularGridInterpolator(
        axes, values_extended, method="linear", bounds_error=True
    )
    mask_rgi = RegularGridInterpolator(
        axes, mask_extended, method="linear", bounds_error=True
    )

    # Include periodic wrapping and non-periodic clipping in the timed region.
    span = np.asarray(GRID_SHAPE, dtype=float)
    points = RNG.random((count, 3)) * (span + 4.0) - 2.0

    def current_no_mask():
        prepared, _ = _prepare_mixed_boundary_points(points)
        return value_rgi(prepared)

    def proxy_no_mask():
        prepared, _ = _prepare_mixed_boundary_points(points)
        return _map_coordinates_linear(values_extended, prepared)

    def current_with_mask():
        prepared, out_mask = _prepare_mixed_boundary_points(points)
        values_out = value_rgi(prepared)
        validity = (mask_rgi(prepared) >= 1.0 - 1e-9) & ~out_mask
        return values_out, validity

    def proxy_with_mask():
        prepared, out_mask = _prepare_mixed_boundary_points(points)
        values_out = _map_coordinates_linear(values_extended, prepared)
        validity = (
            _map_coordinates_linear(mask_extended, prepared) >= 1.0 - 1e-9
        ) & ~out_mask
        return values_out, validity

    current_no_mask()
    proxy_no_mask()
    t_current_no_mask, out_current = _time_call(current_no_mask)
    t_proxy_no_mask, out_proxy = _time_call(proxy_no_mask)
    np.testing.assert_allclose(out_proxy, out_current, rtol=2e-12, atol=2e-12)

    current_with_mask()
    proxy_with_mask()
    t_current_mask, out_current_mask = _time_call(current_with_mask)
    t_proxy_mask, out_proxy_mask = _time_call(proxy_with_mask)
    np.testing.assert_allclose(
        out_proxy_mask[0], out_current_mask[0], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_array_equal(out_proxy_mask[1], out_current_mask[1])
    return t_current_no_mask, t_proxy_no_mask, t_current_mask, t_proxy_mask


def test_uniform_trilinear_benchmark():
    scalar = RNG.normal(size=GRID_SHAPE)
    q5 = RNG.normal(size=GRID_SHAPE + (5,))
    mask = RNG.random(GRID_SHAPE) > 0.08

    rows = []
    validity_rows = []
    mixed_boundary_rows = []
    for count in QUERY_COUNTS:
        points = _make_points(count)
        for name, values in (("scalar", scalar), ("Q5", q5)):
            t_rgi, t_numpy, t_map = _benchmark_case(values, points)
            rows.append((count, name, t_rgi, t_numpy, t_map))

        t_current, t_proxy = _benchmark_validity(q5, mask, points)
        validity_rows.append((count, t_current, t_proxy))
        mixed_boundary_rows.append(
            (count, *_benchmark_full_mixed_boundary(q5, mask, count))
        )

    lines = [
        "# Uniform trilinear interpolation benchmark",
        "",
        f"Grid shape: `{GRID_SHAPE}`. Best of {REPEATS} runs.",
        "",
        "`map_coordinates(order=1)` is used only as a compiled proxy for a future specialized C kernel; it is not proposed as the production API.",
        "",
        "## Backend-only interpolation",
        "",
        "| queries | field | RegularGridInterpolator | NumPy trilinear | compiled proxy | proxy speedup vs RGI |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for count, name, t_rgi, t_numpy, t_map in rows:
        lines.append(
            f"| {count:,} | {name} | {t_rgi:.6f} s | {t_numpy:.6f} s | {t_map:.6f} s | {t_rgi / t_map:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Q5 value + strict validity",
            "",
            "Current-style baseline performs one RGI interpolation for Q5 values and a second RGI interpolation for the float mask. The proxy performs two compiled uniform-grid interpolation passes as well, so it reproduces the current >= 1 - 1e-9 validity semantics exactly. A real fused C kernel could reuse cell indices and weights across both outputs and should have additional headroom beyond this proxy.",
            "",
            "| queries | current-style | compiled proxy + direct validity | speedup |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for count, t_current, t_proxy in validity_rows:
        lines.append(
            f"| {count:,} | {t_current:.6f} s | {t_proxy:.6f} s | {t_current / t_proxy:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Full-ish Q5 path with mixed boundaries",
            "",
            f"Periodic axes: `{tuple(bool(v) for v in PERIODIC)}`. Timings include copying query points, out-of-domain detection, periodic modulo, non-periodic clipping, and interpolation. Periodic endpoint arrays are constructed outside the timed region, matching GridInterpolator construction-time behavior.",
            "",
            "| queries | current no mask | proxy no mask | speedup | current + validity | proxy + validity | speedup |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for count, tc0, tp0, tcm, tpm in mixed_boundary_rows:
        lines.append(
            f"| {count:,} | {tc0:.6f} s | {tp0:.6f} s | {tc0 / tp0:.2f}x | {tcm:.6f} s | {tpm:.6f} s | {tcm / tpm:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the compiled proxy is materially faster than RGI, a dedicated C kernel has plausible headroom because Nematics3D can additionally fuse coordinate preparation, periodic wrapping, clipping, component interpolation, and validity in one pass.",
            "- Note: the production validity rule is implemented as interpolated mask >= 1 - 1e-9. This is almost, but not exactly, equivalent to requiring all eight support voxels to be valid; an invalid corner with a sufficiently tiny interpolation weight can still pass. The first exploratory run found 1 such case among 500,000 random points.",
            "- If the direct NumPy trilinear path is slower, that does not argue against C; it mostly shows that eight fancy-index gathers and temporary arrays are expensive in NumPy.",
            "- This benchmark deliberately uses in-domain index-space points. Physical-coordinate transforms and periodic wrapping should be benchmarked separately before production work.",
            "",
        ]
    )
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
