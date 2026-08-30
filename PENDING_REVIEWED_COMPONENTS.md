# Pending Reviewed Components

This file is a lightweight staging list for functions that have already been cleaned up or reviewed during the current beta-preparation work, but have **not yet been added to** `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

It is intentionally less strict than the formal reviewed-components ledger. An item should stay here until the stronger archive requirements (tests or an explicit no-test decision, validation evidence, final source review, and exact reviewed commit) are satisfied and recorded.

## Pending archive

### `get_box_corners()`

- Source: `src/nematics3d/geometry/box.py`
- Status: implementation cleaned up; geometry module migrated into the `geometry/` package; active callers migrated away from the deleted `general.py`.
- Behavior reviewed: fixed eight-corner ordering, floating output, non-negative finite lengths, and zero-length degenerate dimensions.
- Focused tests: `tests/test_geometry_box.py`.
- Tutorial: `tutorials/geometry/box/get_box_corners.ipynb`.
- Archive note: re-run/record final validation before moving to the formal ledger.

### `sample_van_der_corput()`

- Source: `src/nematics3d/analysis/sampling/van_der_corput.py`
- Status: replaces the removed `sample_far()` helper with the standard base-2 van der Corput sequence; exported through `analysis.sampling`, `analysis`, and the top-level package API.
- Behavior reviewed: sequence begins `0, 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...`; output lies in `[0, 1)`; input uses the existing `as_number(..., is_integer=True)` validation path.
- Focused tests: intentionally not added because the implementation is very small and deterministic.
- Tutorial: `tutorials/analysis/sampling/sample_van_der_corput.ipynb`.
- Archive note: record a final direct smoke check and exact reviewed commit before moving to the formal ledger.

### `blue_red_in_white_bg()`

- Source: `src/nematics3d/classes/visual/color.py`
- Status: retained under its original public/internal name, but implementation rewritten to express the blue-to-green-to-red interpolation directly in normalized `[0, 1]` RGB coordinates instead of using 8-bit magic numbers.
- Behavior reviewed: 511 RGB colors; blue-to-green followed by green-to-red; shared green endpoint included once; each RGB vector is L2-normalized so mixed colors remain visually strong on a white background.
- Focused tests: none currently.
- Tutorial: none currently.
- Archive note: decide whether direct numerical smoke checks are sufficient before formal archival.

### `wrap_points_to_box()`

- Source: `src/nematics3d/grid/periodic.py`
- Status: retained as an active periodic-coordinate helper used by disclination-line section workflows; reviewed and cleaned up without changing the inverse-transform -> lattice-space wrap -> forward-transform algorithm.
- Behavior reviewed: single-point and multi-point inputs; mixed periodic/non-periodic axes; wrapping in lattice coordinates under non-identity grid transforms and offsets; input isolation; empty point collections; strict point-type validation.
- Fixes: preserved raw input types until `as_points()` validation so booleans and numeric strings are rejected instead of silently coerced; corrected empty one-dimensional input so `[]` returns shape `(0, 3)` instead of being misclassified as a single point and raising `IndexError`.
- Focused tests: `tests/test_grid_periodic.py` (new coverage for wrapping behavior, transform/offset semantics, empty input, input isolation, and invalid inputs).
- Tutorial: `tutorials/grid/periodic/wrap_points_to_box.ipynb`.
- Review commits: implementation `c1ba8a8fe963338c5154b6a8fa9a7b20c1818260`; focused tests `b7fdd04fae749d9894bd0bc727a2871a1c48ce5e`; tutorial `30e1cb3e50d0b21ee8187d537cae9699e4769064`.
- Archive note: focused tests have been added, but no GitHub Actions workflow ran for the branch push; record actual test/validation execution and final reviewed commit before moving to the formal ledger.

### `resolve_plane_physical_axes()`

- Source: `src/nematics3d/grid/plane.py`
- Status: retained as an active plane-basis helper used by `PlaneGrid` whenever its physical sampling basis is generated or updated; geometric strategy preserved while the public contract was clarified and validation tightened.
- Behavior reviewed: non-unit `normal` and `axis1` inputs are normalized internally; missing `axis1` generates a deterministic in-plane reference axis; non-perpendicular `axis1` is projected into the plane and renormalized; collinear `axis1` falls back to automatic axis generation; `axis2` is derived as `cross(normal, axis1)` to form an orthonormal right-handed plane basis.
- Fixes: corrected the docstring to match actual normalization behavior; added strict `as_bool()` validation for `is_warn`; made collinearity/perpendicularity tolerances explicit with `rtol=0.0`.
- Focused tests: `tests/test_grid_plane.py` (new coverage for valid normalized behavior, non-unit inputs, automatic axis generation, projection, collinear fallback, invalid `is_warn`, and invalid vectors).
- Tutorial: intentionally not added; the helper is simple and primarily supports `PlaneGrid`, and its direct public use is adequately described by the docstring/API reference.
- Review commits: implementation `eb0ed76442d791d875f628dc7a1ffcb5e5a767d9`; focused tests `e285f71f722bdacc576f0d785305d5ad04dd48f1`.
- Archive note: focused tests have been added, but actual test/validation execution has not yet been recorded; run focused validation and record the final reviewed commit before moving to the formal ledger.

### `SmoothedLine`

- Source: `src/nematics3d/classes/smoothed_line.py`
- Status: core smoothing pipeline and initialization path reviewed and cleaned up without changing the Savitzky-Golay plus FITPACK smoothing algorithm or the existing window-selection formula.
- Fixes and cleanup: `window_ratio` is now constrained to positive values; internal option normalization uses `OptsBase.act_internal_update()` with normal validated assignment; coupled `window_length`/`window_ratio` resolution is isolated in `_helper_resolve_window_opts()`; duplicate short-line fallback was removed; spline-position parameter validation/wrap handling is shared through `_helper_resolve_spline_u()`; `act_calc_tangent()` now validates `is_return_coord` with `as_bool()`; `raw_coords` and `state_is_window_warning` initialization now flows through `HostBase.__init__` and the normal HostBase raw/state validator path; redundant pre-smoothing bootstrap assignments for `calc_coords` and `calc_result` were removed because the first smoothing pass always resolves `calc_coords` and either success or fallback assigns `calc_result`.
- Behavior reviewed: explicit-window and ratio-derived smoothing; automatic odd-window normalization; short-line fallback and spline-cache clearing; periodic `wrap` seam behavior; position/tangent evaluation at the periodic 0/100 boundary; preservation of the original successful smoothing regression behavior; HostBase-managed validation and storage of initial `raw_coords` and `state_is_window_warning`; successful first-pass creation of `calc_coords`/`calc_result` without constructor bootstrap, including fallback.
- Focused tests: `tests/smooth/test_smoothed_line.py`, expanded from the baseline regression to four tests covering the cleanup cases above.
- Validation actually run: core cleanup GitHub Actions run `33328556266`, job `99302909548`, Python 3.12 on Ubuntu 24.04 reported `4 passed in 0.77s`; initialization refactor GitHub Actions run `33332340930`, job `99313078383`, reported `4 passed in 0.84s`; calc-bootstrap removal GitHub Actions run `33332489332`, job `99313485550`, reported `4 passed in 1.23s`; all three runs also passed `py_compile` and Black checks.
- Review commits: core implementation/tests `e3732be394fdb898b230eb38e89c5f5ee3a93b41`; HostBase initialization refactor `5d7d003`; redundant calc-bootstrap removal `f33a107`.
- Tutorial: no new standalone tutorial added during this cleanup; existing class-level documentation and tests cover the reviewed behavior for now.
- Deferred review: ndarray mutability/aliasing of raw and result arrays, NumPy 2.x `__array__` copy-protocol compatibility, and whether the remaining entity/impl/status bootstrap assignments can be reduced further are intentionally left for a separate API/architecture review.
- Archive note: smoothing behavior and the reviewed initialization simplifications are validated; resolve or explicitly accept the remaining deferred API/architecture questions before formal archival.
