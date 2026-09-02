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

### `find_rotation_axis()`

- Source: `src/nematics3d/geometry/rotation.py`.
- Status: reviewed and moved out of the catch-all geometry module into a dedicated rotation-axis module; `QPlanePolar.act_calc_omega()` now calls the canonical geometry API directly and consumes `RotationAxisResult`; the old `general` compatibility adapter and duplicate implementation in `geometry/misc.py` were removed.
- Behavior reviewed: finite ordered 3D directors; at least two samples; explicit unit-vector validation; smallest-eigenvalue second-moment fit; orientation by net adjacent cross-product rotation; and structured diagnostics through `RotationAxisResult`.
- Focused tests: `tests/geometry/test_rotation.py`, covering result type, positive and negative ordered rotation, metric/result consistency, minimum sample count, unit-vector validation, and non-finite input rejection.
- Downstream integration: `tests/classes/test_q_plane.py` directly exercises the real `find_rotation_axis()` result through `QPlanePolar.act_calc_omega()` and confirms that `axis`, every fit diagnostic, defect-domain flags, ring metadata, and the copied opts snapshot are represented correctly in `OmegaResult`.
- Tutorial: `tutorials/geometry/rotation/find_rotation_axis.ipynb`, covering orientation, the structured result fields and metric view, reversed ordering, input requirements, and the zero-net-rotation sign limitation.
- Validation actually run: `python -m pytest tests/geometry/test_rotation.py tests/classes/test_q_plane.py tests/classes/test_plane_grid_polar.py tests/classes/test_q_field_object_phase2.py -q` reported `25 passed`; Black formatting and `black --check` passed for the implementation, downstream, and focused tests; in-memory syntax compile passed; notebook JSON and nbformat schema validation passed; the complete tutorial executed successfully; `git diff --check` passed.
- Review commits: structured implementation `18fbeca`; direct QPlane migration `a92fd41`; removal of the `general` adapter `e4ae2a0`; removal of the obsolete `geometry/misc.py` implementation `bda5046`; initial formatting cleanup `ceb2022`; final downstream review, integration test, formatting, and tutorial `b1ffd66d7ab290aa1f23d0b6cfa3207d24e1ca51`.
- Archive note: implementation, structured output, caller migration, focused and downstream integration tests, API cleanup, tutorial, source formatting, and final validation are complete. This component is ready to move to the formal reviewed-components ledger using reviewed commit `b1ffd66d7ab290aa1f23d0b6cfa3207d24e1ca51`.

### `find_plane_normal()`

- Source: `src/nematics3d/geometry/plane.py`.
- Status: reviewed and moved out of `geometry/misc.py` into a dedicated plane-fitting module. The canonical call returns `PlaneNormalResult(ResultBase)`, with the fitted normal and all fixed diagnostics exposed as typed fields; `metric` remains a shallow convenience view for downstream code.
- Behavior reviewed: finite 3D point collections with at least three samples; least-squares plane through the centroid; smallest-eigenvalue eigenvector as the plane normal; explicit acknowledgement that normal sign is intrinsically ambiguous; non-negative clipping of tiny numerical eigenvalue noise; planarity score clamped to `[0, 1]`; RMS normal thickness; and a scale-independent normal-degeneracy diagnostic `linearity_risk = lambda_0 / lambda_1`, with the exactly one-dimensional case mapped to `1`.
- API cleanup: input validation now uses `as_points(..., d=3, min_num=3)` rather than ad hoc length checking. The old implementation in `geometry/misc.py` was removed, and `PlaneNormalResult` plus `find_plane_normal` are exported canonically through `nematics3d.geometry`.
- Focused tests: `tests/geometry/test_plane.py`, covering the structured `ResultBase` return, exact-plane normal up to sign, centroid and metric consistency, exact-line degeneracy, minimum sample count, 3D shape validation, non-finite rejection, and the current internal compatibility path.
- Current downstream compatibility: `DisclinationLine.act_calc_norm()` is the remaining production caller and still requests the historical tuple-unpack form. To avoid an unsafe unrelated rewrite of the 1800-line `disclination_line.py`, `geometry/plane.py` temporarily uses a private `_LegacyPlaneNormalResult`, which still subclasses `PlaneNormalResult`/`ResultBase` and only changes iteration for that one old caller. New code should call `find_plane_normal(points)` without `is_return_metric` and consume the structured result directly.
- Review commits: structured implementation `460dd7d`; canonical geometry export `5fd81d7`; removal from `geometry/misc.py` `f56fc3d`; focused tests `688fce4`, compatibility coverage `e65b98a`, and numerical tolerance cleanup `af93c71`; temporary private compatibility adapter `a8ad4cd`.
- Archive note: the core implementation and structured public result are reviewed, but this component is not ready for formal archive until `DisclinationLine.act_calc_norm()` is migrated to consume `PlaneNormalResult` directly, the private compatibility adapter and `is_return_metric` argument are deleted, focused/downstream pytest and Black/compile/import validation are run, and the exact final reviewed commit is recorded.
