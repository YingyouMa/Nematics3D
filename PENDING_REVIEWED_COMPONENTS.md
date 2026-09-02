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
- PEP 8 review: long source lines in the result-field documentation and RMS expression were reformatted; naming, imports, spacing, and layout are otherwise consistent with the repository style. Formatting cleanup commit: `ceb2022`.
- Review commits: structured implementation `18fbeca`; direct QPlane migration `a92fd41`; removal of the `general` adapter `e4ae2a0`; removal of the obsolete `geometry/misc.py` implementation `bda5046`; unrelated QPlane description restoration `a2c0b99`; final formatting cleanup `ceb2022`.
- Archive note: implementation, caller migration, focused tests, API cleanup, and source-style review are complete. Actual focused pytest/Black/compile/import validation has not yet been recorded, so keep this component pending rather than formally archived.
