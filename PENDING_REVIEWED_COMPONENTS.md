# Pending Reviewed Components

This file is a lightweight staging list for functions or classes that have already been cleaned up or reviewed during the current beta-preparation work, but have **not yet been added to** `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

It is intentionally less strict than the formal reviewed-components ledger. An item should stay here until the stronger archive requirements (tests or an explicit no-test decision, validation evidence, final source review, and exact reviewed commit) are satisfied and recorded.

## Pending archive

### `SmoothedLine`

- Source: `src/nematics3d/classes/smoothed_line.py`
- Status: smoothing behavior, initialization, spline-fitting/resampling allocation, NumPy array conversion, and result mutability have been reviewed without changing the Savitzky-Golay plus FITPACK algorithm, the window-selection formula, or public smoothing semantics.
- Fixes and cleanup: `window_ratio` is constrained to positive values; internal option normalization uses `OptsBase.act_internal_update()` with validated assignment; coupled `window_length`/`window_ratio` resolution is isolated in `_helper_resolve_window_opts()`; duplicate short-line fallback was removed; spline-position parameter handling is shared; initialization flows through `HostBase.__init__`; redundant constructor bootstrap assignments were removed; `__array__` implements the NumPy 2.x protocol.
- Performance/memory cleanup: final spline output uses one preallocated result array and evaluates one spline component at a time; avoidable large temporaries and a full transpose copy were removed.
- Cached resampling: changing only `num_out_ratio` after successful smoothing reuses the existing spline and skips filtering and fitting.
- Read-only result contract: canonical results are exposed as read-only arrays without an unnecessary second large copy.
- Focused tests: `tests/smooth/test_smoothed_line.py`, covering smoothing equivalence, fallback, cached resampling, NumPy conversion, and result mutability.
- Earlier validation: the recorded cleanup sequence culminated in ten passing focused tests plus syntax and Black checks.
- Remaining review before archive: the source has since undergone naming cleanup and currently needs formatting and a final source review against the latest object-model contracts before a new exact reviewed commit can be recorded.

### `SmoothedLineFunc`

- Source: `src/nematics3d/classes/smoothed_line.py`.
- Status: review in progress; pairwise-delta storage was reduced from `O(N^2)` to `O(N)`, and raw samples now use explicit `ResultBase` objects rather than positional scalar/tuple conventions.
- Result protocol: `raw_func(u_percent, **func_kwargs)` returns a `ResultBase`; `result_value_attr` selects the value to smooth; complete raw results are retained in `calc_results`.
- Beta integration migration: `DisclinationLineSmooth` consumes complete `DefectSectionOmegaResult` samples through `result_value_attr="beta"`.
- Focused tests: `tests/smooth/test_smoothed_line_func.py`, `tests/smooth/test_smoothed_line_func_registry.py`, and `tests/classes/test_q_field_object_phase2.py`.
- Earlier validation: focused delta and ResultBase protocol suites passed together with syntax and Black checks at their recorded review commits.
- Review commits: streamed-delta implementation/tests `74b23bb`; ResultBase sample protocol and beta-integration migration `9d75108`.
- Remaining review before archive: inspect constructor/state initialization, `act_update()`, scalar/vector output shape handling, interpolation behavior, registry interactions, and any remaining edge cases; then run the final focused suite and record the exact reviewed commit.

### `find_plane_normal()`

- Source: `src/nematics3d/geometry/plane.py`.
- Status: reviewed and moved out of `geometry/misc.py` into a dedicated plane-fitting module. The canonical call returns `PlaneNormalResult(ResultBase)`, with the fitted normal and all fixed diagnostics exposed as typed fields; `metric` remains a shallow convenience view for downstream code.
- Behavior reviewed: finite 3D point collections with at least three samples; least-squares plane through the centroid; smallest-eigenvalue eigenvector as the plane normal; explicit acknowledgement that normal sign is intrinsically ambiguous; non-negative clipping of tiny numerical eigenvalue noise; planarity score clamped to `[0, 1]`; RMS normal thickness; and a scale-independent normal-degeneracy diagnostic `linearity_risk = lambda_0 / lambda_1`, with the exactly one-dimensional case mapped to `1`.
- API cleanup: input validation uses `as_points(..., d=3, min_num=3)` rather than ad hoc length checking. The old implementation in `geometry/misc.py`, the temporary `_LegacyPlaneNormalResult`, and the `is_return_metric` compatibility argument were removed completely. `PlaneNormalResult` plus `find_plane_normal` are exported canonically through `nematics3d.geometry`.
- Downstream migration: `DisclinationLine.act_calc_norm()` now consumes `PlaneNormalResult` directly, uses typed diagnostic fields for warning logic, caches `result.normal` and the convenience `result.metric` view, and returns the complete structured result instead of only the normal vector.
- Focused tests: `tests/geometry/test_plane.py`, covering the structured `ResultBase` return, exact-plane normal up to sign, centroid and metric consistency, exact-line degeneracy, minimum sample count, 3D shape validation, non-finite rejection, and explicit rejection of the removed `is_return_metric` API.
- Tutorial: `tutorials/geometry/find_plane_normal.ipynb`, following the repository tutorial guide and the `q_diagonalize()` reference structure; it covers an exact plane, the structured result, the least-squares/eigenvalue derivation, a noisy plane, the distinction between planarity and normal-direction degeneracy, the exact-line failure mode, downstream `DisclinationLine` use, and limitations. Notebook JSON and `nbformat` schema validation were checked before commit; execution has not yet been recorded.
- Review commits: structured implementation `460dd7d`; canonical geometry export `5fd81d7`; removal from `geometry/misc.py` `f56fc3d`; focused tests `688fce4`; numerical tolerance cleanup `af93c71`; direct `DisclinationLine` migration `5a81619`; removal of the private compatibility adapter and metric flag `15fbd26`; legacy-API regression test update `57c7142`; tutorial `adc25f3`.
- Archive note: implementation, structured output, caller migration, API cleanup, focused tests, and tutorial are complete. Keep this component pending until focused/downstream pytest plus Black/compile/import validation and complete tutorial execution are actually run and an exact final reviewed commit is recorded.
