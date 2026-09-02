# Pending Reviewed Components

This file is a lightweight staging list for functions that have already been cleaned up or reviewed during the current beta-preparation work, but have **not yet been added to** `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

It is intentionally less strict than the formal reviewed-components ledger. An item should stay here until the stronger archive requirements (tests or an explicit no-test decision, validation evidence, final source review, and exact reviewed commit) are satisfied and recorded.

## Pending archive

### `find_nearest_point()`

- Source: `src/nematics3d/geometry/nearest.py`.
- Status: reviewed and migrated from the temporary `general` namespace into the canonical `geometry` package; the old `nematics3d.general.find_nearest_point` access path is retained temporarily through lazy compatibility resolution so it does not reintroduce package-import cycles.
- Behavior reviewed: arbitrary-dimensional Euclidean nearest-point lookup; strict one-dimensional query and two-dimensional non-empty candidate shapes; matching coordinate dimensions; finite-value validation; strict boolean validation for `is_return_idx`; optional Python-integer row index; floating-point output; returned-point copy semantics; and deterministic first-row selection when several candidates have exactly equal minimum distance.
- Known limitation: exact ties are not represented as multiple solutions. The first tied row is returned, so changing the input order can change the selected point. This order dependence is documented prominently in the tutorial and explicitly regression-tested; a future API may expose tied nearest points more directly.
- Focused tests: `tests/geometry/test_nearest.py`, covering nearest-point selection, index return, exact ties, tie order dependence, arbitrary dimension, returned-point ownership, invalid shapes, empty candidates, non-finite values, and invalid return flags.
- Tutorial: `tutorials/geometry/find_nearest_point.ipynb`.
- Review commits: migration `2f92ff85ba440c4ce3b7bc63572dcfa1905ed706`; hardened implementation `dbf862d`; focused tests `62a7e17` and tie-order regression `1dd21cb`; circular-import compatibility fix `3afaba2`; tutorial `9fd6dc3` and tie-warning update `8c66860`.
- Archive note: implementation, API review, focused tests, and documentation are complete. Actual focused pytest/Black/compile/import validation has not yet been recorded, so this component remains reviewed but not formally archived.

### `closest_point_on_polyline()`

- Source: `src/nematics3d/geometry/nearest.py`.
- Status: reviewed and migrated from `general/misc.py` into the canonical nearest-point geometry module; production use in tube picking now uses the geometry path, while the old `nematics3d.general.closest_point_on_polyline` access path remains temporarily available through lazy compatibility resolution.
- Behavior reviewed: nearest-point projection onto the union of consecutive line segments; endpoint clamping; multiple-segment selection; repeated consecutive vertices and zero-length segments; one-point polylines; arbitrary spatial dimension; strict dimensional compatibility; finite real inputs; and returned-point copy semantics.
- Fixes and cleanup: replaced the previous `1e-30` denominator substitution with explicit zero-length-segment handling; centralized input normalization through `datatypes.as_points()`, `as_vector()`, and `as_bool()`-based shared nearest-point validation instead of duplicating real/finite/copy checks in the geometry implementation.
- Focused tests: `tests/geometry/test_nearest.py`, including interior projection, endpoint clamping, multiple segments, repeated vertices, one-point polylines, arbitrary dimension, empty input, invalid shapes, dimension mismatch, and non-finite values.
- Tutorial: intentionally not added; this is a low-level geometry primitive used primarily by higher-level picking behavior, and its direct API is sufficiently covered by the docstring and focused tests.
- Review commits: initial migration and hardening `82ccdca`; geometry export `b4c00a1`; removal from `general/misc.py` `867bdca`; lazy general compatibility `debfad3`; focused tests `db041b7`; datatype-validator refactor `c65455d`.
- Archive note: implementation, API review, focused tests, and documentation decision are complete. Actual focused pytest/Black/compile/import validation has not yet been recorded, so this component remains reviewed but not formally archived.

### `as_polydata_input()` and `copy_polydata_geometry()`

- Source: `src/nematics3d/geometry/polydata.py`.
- Status: reviewed geometry-boundary helpers, exported canonically through `nematics3d.geometry`; all active production callers use the geometry path, and the former `make_clean_polydata` name has been removed without a compatibility alias.
- Behavior reviewed: accepted PyVista/VTK dataset families and conversion fallbacks; existing `pyvista.PolyData` ownership semantics; keyword-only diagnostic naming; conversion failure chaining; independent geometry/topology deep copies; removal of point, cell, and field arrays; explicit rejection of non-PolyData copy inputs; and explicit distinction between geometry-only copying and `PolyData.clean()` topology cleanup.
- Focused tests: `tests/geometry/test_polydata.py`, with downstream coverage from `tests/sample/test_surface_sampling_package.py` and `tests/visual/test_plot_polydata.py`.
- Validation actually run: focused geometry, sampling, and PlotPolyData run reported `15 passed, 1 deselected`; the deselected test is the pre-existing `line_width` versus `edge_width` option-name mismatch and is unrelated to these helpers. Black and `git diff --check` passed for the modified files.
- Remaining limitation: geometry-only copying currently deep-copies attached arrays before removing them, which can create a temporary memory peak for data-heavy meshes; changing VTK topology ownership is deferred to a dedicated optimization.
- Archive note: implementation and focused review are complete; record the exact reviewed commit after these working-tree changes are committed, then move this entry to `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

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
- Status: smoothing behavior, initialization, spline-fitting/resampling allocation, NumPy array conversion, and result mutability have been reviewed without changing the Savitzky-Golay plus FITPACK algorithm, the window-selection formula, or public smoothing semantics.
- Fixes and cleanup: `window_ratio` is constrained to positive values; internal option normalization uses `OptsBase.act_internal_update()` with validated assignment; coupled `window_length`/`window_ratio` resolution is isolated in `_helper_resolve_window_opts()`; duplicate short-line fallback was removed; spline-position parameter handling is shared through `_helper_resolve_spline_u()`; `act_calc_tangent()` validates `is_return_coord`; `raw_coords` and `state_is_window_warning` initialization flows through `HostBase.__init__`; redundant constructor bootstrap assignments for `calc_coords`, `calc_result`, `entity_tck`, `entity_linefuncs`, `calc_is_smoothed`, and `calc_status` were removed, while `impl_linefunc_count` is initialized alongside the line-function registry after smoothing initialization; `__array__` implements the NumPy 2.x `(dtype=None, copy=None)` protocol.
- Performance/memory cleanup: final spline output is generated by `_helper_sample_spline_result()` into one preallocated `(M, D)` result array, evaluating one spline component at a time; `u_out` is created only after spline fitting; filtered input, spline-input, and `u_spline` temporaries are released before final-output allocation; `splprep()` receives the transposed view directly instead of an unnecessary full `.T.copy()`.
- Cached resampling: changing only `num_out_ratio` after successful smoothing reuses the existing `entity_tck` and resamples it directly, skipping both `savgol_filter()` and `splprep()`; changes to raw coordinates or spline-defining options still rebuild normally.
- Read-only result contract: all canonical `calc_result` assignments now flow through `_helper_set_result()`, which wraps the result with `as_readonly_array(..., dtype=None, copy=False)`. Successful smoothing and fast resampling therefore retain their existing result data without a second large copy while exposing a read-only canonical array. Fallback similarly returns a separate read-only view of `calc_coords`, preserving shared data but leaving `raw_coords`/`calc_coords` themselves writable.
- Behavior reviewed: explicit-window and ratio-derived smoothing; odd-window normalization; recoverable short-line fallback; periodic `wrap` seam behavior; position/tangent evaluation at the periodic 0/100 boundary; exact equality between the lower-memory component-wise sampler and the previous SciPy vector-output result; output-only resampling reuses the same spline and matches a fresh recomputation; NumPy 2.x no-copy, forced-copy, dtype-conversion, and impossible dtype-conversion-with-`copy=False` behavior; normal mutation attempts through `result`, `np.asarray(line, copy=False)`, and indexing fail with read-only-array errors; fallback result shares memory with `calc_coords` without making the raw/calc arrays read-only.
- Focused tests: `tests/smooth/test_smoothed_line.py`, now ten tests including old/new output equivalence, no-recompute fast-path coverage, NumPy 2.x protocol coverage, successful read-only result behavior, fallback shared-memory/read-only behavior, and fast-resample read-only preservation.
- Validation actually run: core cleanup run `33328556266`, job `99302909548`, reported `4 passed in 0.77s`; initialization refactor run `33332340930`, job `99313078383`, reported `4 passed in 0.84s`; calc-bootstrap removal run `33332489332`, job `99313485550`, reported `4 passed in 1.23s`; performance/reuse run `33334702185`, job `99319398377`, reported `6 passed in 0.76s`; transpose-view validation run `33334766970`, job `99319575944`, reported `6 passed in 0.73s`; NumPy 2.x protocol run `33335455742`, job `99321429836`, reported `7 passed in 0.86s`; read-only result run `33335969022`, job `99322802523`, reported `10 passed in 0.83s`. All recorded runs also passed `py_compile` and Black checks.
- FITPACK transpose benchmark: on 300,000 three-dimensional points in the performance/reuse CI environment, explicit `filtered.T.copy()` took `0.074831 s` with `28.61 MiB` traced Python-side peak allocation, while passing `filtered.T` directly took `0.068882 s` with `14.88 MiB`; the direct-view implementation was retained and separately regression-tested.
- Review commits: core implementation/tests `e3732be394fdb898b230eb38e89c5f5ee3a93b41`; HostBase initialization refactor `5d7d003`; redundant calc-bootstrap removal `f33a107`; spline reuse/lower-memory sampling and tests `acd8af67a6dd24bb0656f382ac3f8a01f6b1c91d`; transpose-copy removal `d4d7d64`; NumPy 2.x protocol support `baf304c`; read-only canonical result `72327a7`.
- Tutorial: no new standalone tutorial added during this cleanup; existing class documentation plus focused tests cover the reviewed behavior for now.
- Accepted input-ownership design: `as_points()` intentionally copies validated input coordinates so `SmoothedLine.raw_coords` is isolated from caller-side mutation. A zero-copy input mode is intentionally not supported because its aliasing risk outweighs the retained-input memory saving for the current API. Chunked spline output was intentionally not introduced in this optimization pass.
- Archive note: reviewed smoothing behavior, initialization, cached output-only resampling, current non-chunked memory optimizations, NumPy 2.x conversion semantics, result mutability protection, and input ownership are validated; remaining review work should focus on adjacent components rather than reopening zero-copy input semantics.

### `SmoothedLineFunc`

- Source: `src/nematics3d/classes/smoothed_line.py`.
- Status: review in progress; the pairwise-delta memory path and the raw sample-result protocol have now been cleaned up. `SmoothedLineFunc` still uses the same weighted local-polynomial smoothing and `interp1d` interpolation mathematics, but raw samples are now represented by explicit `ResultBase` objects rather than positional scalar/tuple return conventions.
- Performance/memory cleanup: the previous implementation materialized `_linefunc_sample_delta_matrix(u_samples, mode)`, an `N x N` float array, before iterating row by row. The smoothing loop now computes only the current length-`N` delta vector as `u_samples - u_center`; wrap mode applies the same `[-50, 50)` periodic remapping. The obsolete `_linefunc_sample_delta_matrix()` helper was removed. Peak delta-storage complexity therefore drops from `O(N^2)` to `O(N)` while arithmetic complexity remains `O(N^2)` at this stage.
- Result protocol: `raw_func(u_percent, **func_kwargs)` must now return a `ResultBase` instance at every sample. `result_value_attr` is exposed by both `SmoothedLine.act_create_linefunc()` and `SmoothedLineFunc`, defaults to `"value"`, and identifies the result attribute whose values are stacked, smoothed, and interpolated. The attribute name is validated as a non-empty string. A non-`ResultBase` sample raises `TypeError`; a sample lacking the configured attribute raises `AttributeError` with the sampled `u_percent` and concrete result type in the diagnostic.
- Raw-result retention: complete per-sample results are retained in `calc_results` as a tuple, so raw numerical values and any accompanying diagnostics remain accessible after smoothing. `calc_values` now unambiguously means the smoothed values extracted from `result_value_attr`. The old positional metadata containers `calc_metrics`, `calc_payload_samples`, and `calc_payload_shared` were removed from the class and from active tests.
- Beta integration migration: `DisclinationLineSmooth.act_add_beta_interpolator()` now uses `result_value_attr="beta"`; `_helper_sample_beta_from_smooth()` returns the complete `DefectSectionOmegaResult` directly rather than decomposing it into `(beta, metric, payload_sample, payload_shared)`. `DefectSectionOmegaResult` now also retains the sampled line `tangent`, so the information previously split among metric/sample/shared payloads is represented in one inspectable `ResultBase` object together with `omega`, `metric`, `layer`, `num_directors`, `R`, `opts`, `beta`, `u_percent`, and `position`.
- Behavior reviewed so far: streamed deltas are numerically identical to the former full-matrix deltas for both `interp` and periodic `wrap` modes; scalar and vector-valued smoothing retain the existing weighted local-polynomial behavior. The new result protocol preserves each complete raw sample, supports the default `value` attribute and a caller-selected alternative attribute, and rejects both raw non-`ResultBase` returns and missing selected attributes.
- Focused tests: `tests/smooth/test_smoothed_line_func.py` now contains six focused tests: two old-vs-streamed delta regressions plus four ResultBase-protocol tests covering complete-result retention/default `value`, custom `result_value_attr`, rejection of non-`ResultBase` returns, and rejection of missing selected attributes. `tests/smooth/test_smoothed_line_func_registry.py` was migrated from scalar-return helpers to `ResultBase` helpers, and `tests/classes/test_q_field_object_phase2.py` was migrated from the removed metric/payload fields to the new `calc_results` beta-result model.
- Validation actually run: delta optimization run `33337186451`, job `99326086983`, Ubuntu 24.04 / Python 3.12, reported `12 passed in 0.97s` across the ten existing `SmoothedLine` tests plus two delta-regression tests; `py_compile`, Black, and removal checks for `_linefunc_sample_delta_matrix` also passed. ResultBase protocol run `33341267062`, job `99337218436`, Ubuntu 24.04 / Python 3.12.14, reported `16 passed in 0.86s` across `tests/smooth/test_smoothed_line.py` and `tests/smooth/test_smoothed_line_func.py`; `py_compile` also passed for `smoothed_line.py`, `disclination_line.py`, `test_smoothed_line_func_registry.py`, and `test_q_field_object_phase2.py`; Black formatting/checks passed; final `git grep` found no remaining `calc_metrics`, `calc_payload_samples`, or `calc_payload_shared` references.
- Review commits: streamed-delta implementation/tests `74b23bb`; ResultBase sample protocol and beta-integration migration `9d75108`.
- Remaining review before archive: inspect constructor/state initialization, `act_update()`, scalar/vector output shape handling, interpolation behavior, registry interactions, and any remaining edge cases; then run the final focused suite and record the exact reviewed commit.
