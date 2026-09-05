# Beta Release Reviewed Components

This file records functions, classes, modules, and other Python components that
have been fully reviewed and verified during the current public-beta cleanup.
It is an evidence log, not a list of planned work.

Planned or partially reviewed components belong in
`BETA_RELEASE_CHECKLIST.md` or an issue, and must not be added here until their
review is complete.

## Meaning of "reviewed"

A component may be recorded here only after all applicable work below is
complete:

- its implementation and public behavior have been inspected;
- discovered correctness, API, documentation, and style problems have been
  resolved or explicitly accepted as documented limitations;
- relevant existing tests have been identified and run;
- missing focused tests have been added when practical;
- numerical edge cases and failure behavior have been considered;
- the modified Python files pass the repository formatter;
- focused validation passes in the `Nematics3D` conda environment;
- any remaining limitation is documented in the record; and
- the reviewed source has not changed since the recorded commit.

A passing test alone does not mean that a component has been fully reviewed.
Similarly, formatting-only work is not enough to add a component to this file.

## Recording rules

- Record the smallest meaningful component: prefer a function or class over an
  entire module when only that function or class was reviewed.
- Record a whole Python file only when all relevant contents of that file were
  reviewed.
- Use repository-relative paths so links continue to work outside a local
  checkout.
- List every focused test file used as evidence. Write `None` only when no test
  exists and explain why that is acceptable in the notes.
- Record the exact validation commands, not only "tests passed".
- Record the reviewed commit. A later source change makes the entry stale until
  it is reviewed again.
- Do not silently delete stale entries. Move them to the stale section with an
  explanation so the review history remains visible.

## Confirmed reviewed components

### `nematics3d.core.RegistryBase`

| Field | Evidence |
| --- | --- |
| Kind | Public ordered registry for named repository objects |
| Source | [`src/nematics3d/core/registry_base.py`](../../src/nematics3d/core/registry_base.py), with the registry-aware rename hook in [`src/nematics3d/core/class_base.py`](../../src/nematics3d/core/class_base.py) |
| Tests | [`tests/classes/test_registry_base.py`](../../tests/classes/test_registry_base.py), plus `tests/classes/test_class_base.py`, `tests/smooth/test_smoothed_line_func_registry.py`, `tests/core/test_core_package.py`, `tests/visual/test_figure_manager.py`, and `tests/visual/test_glyph_scalar_bar.py` as downstream compatibility checks |
| Tutorial | [`tutorials/core/RegistryBase.ipynb`](../../tutorials/core/RegistryBase.ipynb) |
| Review scope | Construction and managed state; ordered storage and immutable public tuple view; registration return value; duplicate-object handling; duplicate-name uniquification; registry-aware object renaming without self-conflict; moving objects between registries; weak `registry` relation binding/unbinding; lookup by name, integer index, and `None` passthrough; containment, iteration, length, clearing, and representation helpers. Existing `is_contain_ok` / `is_missing_ok` recovery semantics are intentionally retained. |
| Validation | `python -m pytest tests/classes/test_registry_base.py tests/classes/test_class_base.py tests/smooth/test_smoothed_line_func_registry.py tests/core/test_core_package.py tests/visual/test_figure_manager.py tests/visual/test_glyph_scalar_bar.py` passed twice with no failures (41 passed, then 29 passed under the environment's optional-test collection state); Black passed on `src/nematics3d/core/registry_base.py`, `src/nematics3d/core/class_base.py`, and `tests/classes/test_registry_base.py`; Ruff passed on the same Python files; all code cells in `tutorials/core/RegistryBase.ipynb` were executed sequentially in a temporary focused pytest harness (1 passed) and that harness was removed afterward |
| Reviewed commit | `ac78f14dddec8ade4a801cb8826234afe4eb3e20` |
| Reviewed date | 2026-09-05 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | Name lookup and duplicate-name detection remain linear in registry size by design; current registries are small, and adding a synchronized name index would increase mutation complexity. `registry[None]` intentionally returns `None` for convenience. Duplicate registration and missing-unregister cases keep the repository's existing logger recovery behavior rather than raising through to callers when their corresponding `*_ok` flag is false. |

Summary of changes and evidence:

- Fixed registry-aware renaming so a registered object does not conflict with
  its own current name while still avoiding names used by other objects.
- Standardized `act_register()` to return the registered object on successful
  and already-contained paths.
- Expanded focused coverage across registration, lookup, ordering, rename,
  move, relation cleanup, clear, and invalid-key behavior, and removed the
  obsolete import workaround from the focused test file.
- Rewrote and executed the dedicated tutorial according to
  `tutorials/for_developers/tutorial_writing_guide.md`, using real public
  `Nematics3D` objects rather than a developer-only demo subclass.

### `nematics3d.classes.OptsSmoothedLine` and `SmoothedLine`

| Field | Evidence |
| --- | --- |
| Kind | Public managed polyline-smoothing object and paired options container |
| Source | [`src/nematics3d/classes/smoothed_line.py`](../../src/nematics3d/classes/smoothed_line.py) |
| Tests | [`tests/smooth/test_smoothed_line.py`](../../tests/smooth/test_smoothed_line.py), plus `tests/smooth/test_smoothed_line_func_registry.py` as a downstream registry/integration check |
| Tutorial | [`tutorials/smooth/SmoothedLine.ipynb`](../../tutorials/smooth/SmoothedLine.ipynb) |
| Review scope | `OptsSmoothedLine`, `LineSmoothingConfigError`, and `SmoothedLine` only; construction, HostBase commit/reapply behavior, window normalization, Savitzky-Golay filtering, FITPACK spline creation, output resampling, cached resampling, fallback/recovery, raw-coordinate replacement, position/tangent queries, NumPy array protocol, and read-only result behavior. `SmoothedLineFunc` and its helper functions are explicitly outside this review scope. |
| Validation | `python -m pytest tests/smooth/test_smoothed_line.py` (13 passed); `python -m pytest tests/smooth/test_smoothed_line.py tests/smooth/test_smooth.py tests/smooth/test_smoothed_line_func_registry.py` (19 passed); Black applied to source and focused tests; Ruff passed on source and focused tests; all code cells in `tutorials/smooth/SmoothedLine.ipynb` were executed sequentially in a temporary focused pytest harness (1 passed) and that harness was removed afterward |
| Reviewed commit | `ac62b2c595f203aaf2509abdc536b474858d2dd2` |
| Reviewed date | 2026-09-05 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | The smoothing algorithm intentionally remains Savitzky-Golay filtering followed by a FITPACK spline rather than an arc-length-preserving or exact geometric smoother. `mode="wrap"` assumes periodic geometry. Recognized smoothing/configuration failures fall back to processed raw coordinates and clear the spline cache. The source file also contains unreviewed `SmoothedLineFunc` code, so this record applies only to the explicitly listed classes and methods. |

Summary of changes and evidence:

- Preserved the established smoothing algorithm while cleaning initialization,
  option normalization, spline sampling allocation, and NumPy 2.x conversion.
- Output-only `num_out_ratio` changes reuse the cached spline and avoid
  refiltering/refitting.
- Canonical outputs are read-only without an unnecessary second large copy.
- Added regression coverage for raw-coordinate commits, complete fallback-to-
  success recovery, and the minimum one-sample output-density boundary.
- Removed the obsolete focused-test import workaround now that normal package
  imports work in the project test environment.
- Added and executed a dedicated `SmoothedLine` tutorial; `SmoothedLineFunc` is
  deliberately deferred to its own review.

### `nematics3d.geometry.find_plane_normal`

| Field | Evidence |
| --- | --- |
| Kind | Public least-squares 3D plane-fitting utility with structured diagnostics |
| Source | [`src/nematics3d/geometry/plane.py`](../../src/nematics3d/geometry/plane.py) |
| Tests | [`tests/geometry/test_plane.py`](../../tests/geometry/test_plane.py) plus downstream geometry/class validation |
| Tutorial | [`tutorials/geometry/find_plane_normal.ipynb`](../../tutorials/geometry/find_plane_normal.ipynb) |
| Review scope | `PlaneNormalResult`, finite 3D point validation, least-squares normal fit, centroid, eigenvalue diagnostics, exact-plane numerical roundoff handling, planarity, RMS thickness, linearity risk, canonical export, and `DisclinationLine.act_calc_norm()` migration |
| Validation | `python -m pytest tests/geometry/test_plane.py tests/geometry/test_rotation.py tests/classes/test_q_plane.py tests/classes/test_plane_grid_polar.py tests/classes/test_q_field_object_phase2.py` (39 passed); Black applied to source and focused tests; Ruff passed on source and focused tests; tutorial content and notebook structure were reviewed during the component cleanup |
| Reviewed commit | `eef44b2caf70b7d1df4ec2113196e415c36dc12f` |
| Reviewed date | 2026-09-05 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | The fitted normal has the unavoidable sign ambiguity of an unoriented plane. Exactly or nearly one-dimensional point sets do not determine a stable plane normal; `linearity_risk` exposes that degeneracy. Eigenvalues at the matrix roundoff scale are treated as numerical zero so mathematically exact planes report zero RMS thickness. No logger is used in this low-level deterministic geometry helper. |

Summary of changes and evidence:

- Consolidated the plane fit into the dedicated geometry module and removed the
  obsolete compatibility implementation and return-mode flag.
- The public call returns `PlaneNormalResult`, including the normal, centroid,
  eigenvalues, planarity, thickness, and linearity-risk diagnostics.
- Exact rank-2 point clouds now treat smallest eigenvalues at the matrix
  roundoff scale as zero rather than reporting artificial finite thickness.
- `DisclinationLine.act_calc_norm()` consumes the structured result directly.
- Explicit logging decision: no logger is appropriate for this deterministic
  numerical helper.

### `nematics3d.geometry.find_rotation_axis` and `rotation_matrix_from_vectors`

| Field | Evidence |
| --- | --- |
| Kind | Public 3D rotation-axis fitting and direction-to-direction rotation utilities |
| Source | [`src/nematics3d/geometry/rotation.py`](../../src/nematics3d/geometry/rotation.py) |
| Tests | [`tests/geometry/test_rotation.py`](../../tests/geometry/test_rotation.py) plus downstream `tests/classes/test_q_plane.py`, `tests/classes/test_plane_grid_polar.py`, `tests/classes/test_q_field_object_phase2.py`, and `tests/geometry/test_angles.py` |
| Tutorials | [`tutorials/geometry/rotation/find_rotation_axis.ipynb`](../../tutorials/geometry/rotation/find_rotation_axis.ipynb) and [`tutorials/geometry/rotation_matrix_from_vectors.ipynb`](../../tutorials/geometry/rotation_matrix_from_vectors.ipynb) |
| Review scope | Structured rotation-axis fitting for ordered unit directors; fit diagnostics and orientation convention; direction-to-direction minimal rotation; parallel and antiparallel cases; finite/non-zero input validation; internal normalization of rotation-matrix inputs; public exports and active downstream callers |
| Validation | `python -m pytest tests/geometry/test_rotation.py tests/classes/test_q_plane.py tests/classes/test_plane_grid_polar.py tests/classes/test_q_field_object_phase2.py tests/geometry/test_angles.py` (53 passed); Black applied to source and focused tests; Ruff passed on source and focused tests; both tutorials were previously reviewed for executable examples, and the rotation-matrix tutorial contract text was synchronized with the tested internal-normalization behavior |
| Reviewed commit | `d6275a5a6657858e7838c69ad786076ce6624d20` |
| Reviewed date | 2026-09-05 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | `find_rotation_axis()` requires pre-normalized directors and its fitted-axis sign is intrinsically weak when the ordered sequence has zero net rotation. For exactly antiparallel source/target directions, `rotation_matrix_from_vectors()` chooses one deterministic perpendicular axis because the minimal 180-degree rotation is not unique. No logger is used because these are deterministic low-level geometry helpers. |

Summary of changes and evidence:

- `find_rotation_axis()` returns a typed `RotationAxisResult`, validates ordered
  unit directors, and is exercised through its principal `QPlanePolar`
  downstream integration.
- `rotation_matrix_from_vectors()` accepts arbitrary finite non-zero vectors,
  ignores their magnitudes through internal normalization, and is tested for
  generic, parallel, antiparallel, zero, non-finite, and wrong-shape inputs.
- The rotation-matrix tutorial now documents the same normalization contract as
  the implementation and tests.
- Explicit logging decision: no logger is appropriate for these small,
  deterministic numerical helpers.

### `nematics3d.geometry` angle utilities

| Field | Evidence |
| --- | --- |
| Kind | Public geometry angle-conversion and wrapping utilities |
| Source | [`src/nematics3d/geometry/angles.py`](../../src/nematics3d/geometry/angles.py) |
| Tests | [`tests/geometry/test_angles.py`](../../tests/geometry/test_angles.py) and downstream [`tests/classes/test_q_plane.py`](../../tests/classes/test_q_plane.py) |
| Tutorial | None; these are compact mathematical helpers whose conventions and contracts are documented in their public docstrings |
| Review scope | `vector_from_spherical_angles()`, `azimuth_from_vector()`, `polar_angle_from_vector()`, `plane_azimuth_from_direction()`, `wrap_angle_to_pi()`, private validation helpers, public geometry exports, and active repository callers |
| Validation | `python -m pytest tests/geometry/test_angles.py tests/classes/test_q_plane.py` (23 passed); `python -m pytest tests/geometry` (91 passed, 2 unrelated existing failures in plane/rotation tests); full `python -m pytest` blocked during collection by duplicate `test_class_base.py` module names; Black passed on source and focused tests; `ruff check src/nematics3d/geometry/angles.py tests/geometry/test_angles.py` passed; active caller/export audit completed |
| Reviewed commit | `4c69a9f1bb536efc59c7e982cb6976092c65c66c` |
| Reviewed date | 2026-09-05 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | Azimuth at either spherical pole is defined as zero by convention. Vector rejection uses the existing absolute norm threshold of `1e-12`. `plane_azimuth_from_direction()` inherits the rotation module's local-frame convention. No logger is used because these are small deterministic mathematical helpers called in interactive and numerical paths. The broader geometry failures and full-suite collection blocker are outside this review. |

Summary of changes and evidence:

- Consolidated the angle helpers in the dedicated geometry module and confirmed
  that active downstream callers use the public geometry API.
- Made scalar output behavior consistent: single-vector angle extraction and
  scalar wrapping return Python `float`, while batched operations return NumPy
  arrays with preserved leading shape.
- Documented radians, coordinate conventions, broadcasting, output shape,
  failure behavior, and non-mutation guarantees.
- Added focused tests for round trips, poles, scalar and batched return types,
  shape preservation, non-mutation, invalid input, plane projection, and angle
  wrapping boundaries.
- Explicit logging decision: no logger is appropriate for these low-level,
  deterministic helpers.

### `nematics3d.datatypes.Vect` and `as_vector`

| Field | Evidence |
| --- | --- |
| Kind | Public semantic annotation and vector-input validator |
| Source | [`src/nematics3d/datatypes/vector.py`](../../src/nematics3d/datatypes/vector.py) |
| Tests | None; the validator is intentionally covered by direct contract checks and existing downstream tests rather than a dedicated test file |
| Tutorial | None; these are compact developer-facing input helpers |
| Review scope | Dimension semantics, real finite values, exact shape, arbitrary positive dimensions, zero-vector policy, optional normalization, validated replacement recovery, logging, naming, public exports, and all active repository callers |
| Validation | Direct `as_vector()` smoke checks for normalized 3-vectors, arbitrary 5-vectors, and replacement recovery; in-memory compile of all 77 source files; `python -m pytest tests/test_datatypes_check_sn.py tests/core/test_datatypes_qfield.py -q` (16 passed, 23 subtests passed); `python -m pytest tests/classes/test_q_plane.py -q` (1 passed); `black --check` on all modified Python files; active-source stale-name search; `git diff --check` |
| Reviewed commit | `bdb7e25` |
| Reviewed date | 2026-08-24 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | `Vect(d)` records dimension only as reader-facing semantic metadata and does not provide static shape checking. Near-zero rejection uses the repository's existing absolute norm threshold of `1e-12`. Generated build artifacts are not source-of-truth callers and were not rewritten. |

Summary of changes and evidence:

- Retained the compact `Vect(d)` annotation so vector dimensions remain visible
  in function signatures while making its reader-facing role explicit.
- Replaced the mixed-case `as_Vect()` API with `as_vector()` and migrated every
  active source and development caller without a compatibility alias.
- Unified exact-shape, real-number, finite-value, zero-vector, normalization,
  and replacement validation for any positive vector dimension.
- Confirmed that replacement values follow the same contract and cannot return
  an invalid fallback silently.

### `nematics3d.datatypes.Tensor` and `as_tensor`

| Field | Evidence |
| --- | --- |
| Kind | Public semantic annotation and tensor-input validator |
| Source | [`src/nematics3d/datatypes/tensor.py`](../../src/nematics3d/datatypes/tensor.py) |
| Tests | None; the validator is intentionally covered by direct contract checks and existing downstream tests rather than a dedicated test file |
| Tutorial | None; these are compact developer-facing input helpers |
| Review scope | Shape semantics, arbitrary positive-rank shapes, real finite values, exact shape, validated replacement recovery, logging, naming, public exports, `as_axes()` integration, grid transforms, plot extents, and all active repository callers |
| Validation | Direct `as_tensor()` smoke checks for a 2x2 matrix, a rank-three tensor, and replacement recovery; in-memory compile of all 77 source files; `python -m pytest tests/classes/test_q_plane.py tests/core/test_datatypes_qfield.py -q` (16 passed, 23 subtests passed); `black --check` on all modified Python files; active-source stale-name search; `git diff --check` |
| Reviewed commit | `bdb7e25` |
| Reviewed date | 2026-08-24 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | `Tensor(shape)` records shape only as reader-facing semantic metadata and does not provide static shape checking. Scalar shape `()` and dimensions of size zero are intentionally unsupported. Generated build artifacts are not source-of-truth callers and were not rewritten. |

Summary of changes and evidence:

- Retained the compact `Tensor(shape)` annotation so exact shapes remain
  visible in function signatures while making its reader-facing role explicit.
- Replaced the mixed-case `as_Tensor()` API with `as_tensor()` and migrated
  every active caller without a compatibility alias.
- Unified shape-definition, exact-shape, real-number, finite-value, and
  replacement validation for matrices and higher-rank tensors.
- Updated `as_axes()`, grid-transform validation, and plot-extent validation to
  use the normalized interface.

### `nematics3d.datatypes.as_bool`

| Field | Evidence |
| --- | --- |
| Kind | Public scalar boolean-input validator and normalizer |
| Source | [`src/nematics3d/datatypes/bool.py`](../../src/nematics3d/datatypes/bool.py) |
| Tests | [`tests/test_datatypes_bool.py`](../../tests/test_datatypes_bool.py) and downstream datatype, defect, diagonalization, and visual tests |
| Tutorial | None; this is a compact scalar input helper used throughout public APIs and option validators |
| Review scope | Python and NumPy booleans, Python and NumPy real zero/one values, rejection of other real values and non-scalar/non-real inputs, Python `bool` output, validated replacement recovery, logging, PEP 8 naming, dedicated module, public exports, deletion of name-driven `check_bool_flags()`, and active callers |
| Validation | `python -m pytest tests/test_datatypes_bool.py tests/test_datatypes_number.py tests/test_datatypes_dimension_info.py tests/test_disclination_defect_detect.py tests/core/test_q_diagonalization.py tests/visual/test_glyph_empty_coords.py tests/visual/test_glyph_resolver_source.py tests/visual/test_glyph_scalar_bar.py tests/visual/test_plot_vector.py tests/classes/test_grid_offset_none.py -q` (119 passed, 10 subtests passed); Black and `black --check` on all 12 modified Python files; in-memory compile of 140 Python files; reviewed-component manual-boolean-check audit; `git diff --check` |
| Reviewed commit | `5089673` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Strings such as `"true"` and `"false"`, zero-dimensional arrays, and general truthy/falsy objects are intentionally rejected. Integer APIs retain explicit bool rejection because that protects an integer contract rather than normalizing a boolean option. The visual regression run emitted pre-existing VTK cleanup errors and three unrelated warnings. |

Summary of changes and evidence:

- Extracted `as_bool()` from `misc.py` into a dedicated datatype module and
  guaranteed a Python `bool` result for every accepted input.
- Distinguished wrong values from wrong types and revalidated configured
  replacements instead of returning them unchecked.
- Removed `check_bool_flags()` and replaced its name-driven inspection with
  explicit normalization of each public boolean option.
- Migrated reviewed boolean checks in number, DimensionInfo, defect detection,
  and Q diagonalization while retaining deliberate bool rejection in integer
  validators.

### `nematics3d.datatypes.Number` and `as_number`

| Field | Evidence |
| --- | --- |
| Kind | Public real-number semantic alias and scalar-input validator |
| Source | [`src/nematics3d/datatypes/number.py`](../../src/nematics3d/datatypes/number.py) |
| Tests | [`tests/test_datatypes_number.py`](../../tests/test_datatypes_number.py) and downstream datatype and option-validation tests |
| Tutorial | None; this is a compact input helper used by public and internal APIs |
| Review scope | Python and NumPy real scalars, explicit boolean rejection, finite values by default, opt-in NaN and infinity, integer-valued mode, Python scalar return types, inclusive ranges, optional clipping, validated replacement recovery, option validation, logging, PEP 8 naming, public exports, and active callers |
| Validation | `python -m pytest tests/test_datatypes_number.py -q` (25 passed); combined number, Q-field, defect-index, and line-classification run (63 passed, 23 subtests passed); 65 downstream option and visual tests passed with 2 subtests; Black and `black --check` on all 27 modified Python files; in-memory compile of 136 Python files; active-source stale-name search; `git diff --check` |
| Reviewed commit | `c7443c2343dc31c700db9257f3f4125a517e4533` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | `Number` is a reader-facing `numbers.Real` alias, so runtime boolean rejection remains the responsibility of `as_number()`. Integer mode intentionally accepts integer-valued real inputs such as `3.0`; callers whose contract requires an integral input type must retain that stricter boundary check. Ruff was unavailable in the project environment during final validation. |

Summary of changes and evidence:

- Removed the broad `NumericInput` alias and replaced the mixed-case
  `as_Number()` API without retaining a compatibility alias.
- Made finite real scalars the safe default and required explicit opt-in for
  NaN or infinity.
- Ensured ordinary and integer modes return Python `float` and `int` values,
  respectively, while rejecting Python and NumPy booleans.
- Centralized range validation, clipping, and replacement recovery, including
  revalidation of replacement values.
- Migrated active callers and reused the normalized finite non-negative scalar
  contract for Q-field tolerances and defect-index tolerance.

### `nematics3d.datatypes.DimensionInfo` and `as_dimension_info`

| Field | Evidence |
| --- | --- |
| Kind | Public reader-facing per-axis input alias and runtime broadcaster |
| Source | [`src/nematics3d/datatypes/dimension_info.py`](../../src/nematics3d/datatypes/dimension_info.py) |
| Tests | [`tests/test_datatypes_dimension_info.py`](../../tests/test_datatypes_dimension_info.py) and downstream defect, grid, bounds, and plane tests |
| Tutorial | None; this is a compact input helper whose scalar and xyz forms are documented in its public docstring and callers |
| Review scope | One value shared by x, y, and z; three values assigned to the axes in order; Python and NumPy real scalars; exact `(3,)` shape; independent output storage; real-value validation; optional strict boolean/0/1 mode; boolean output dtype; parameter-name errors; removal of redundant `DimensionInfoInput`, `DimensionFlag`, and `DimensionFlagInput`; public exports; and all active callers |
| Validation | `python -m pytest tests/test_datatypes_dimension_info.py tests/test_disclination_defect_detect.py tests/test_disclination_line_classification.py tests/classes/test_grid_offset_none.py tests/classes/test_q_plane.py tests/classes/test_plane_grid.py tests/classes/test_plane_grid_polar.py tests/classes/test_bounds_obb.py -q` (79 passed); post-extraction focused run (56 passed); Black and `black --check`; in-memory compile of 137 Python files; stale-name and call-site audit; direct-module and public import identity smoke test; `git diff --check` |
| Reviewed commit | `5089673` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The base converter intentionally validates structure rather than domain-specific ranges, so spacing, radii, and minimum lengths retain their caller-level positivity checks. `DimensionPeriodic` remains a temporary specialization pending the separate periodic-box-size review. The wider validation emitted two pre-existing `DisclinationLine` warnings from remainder operations involving infinite non-periodic sizes. |

Summary of changes and evidence:

- Made `DimensionInfo` itself express the flexible public input contract: one
  shared value or three values corresponding to x, y, and z.
- Removed redundant input and flag aliases while retaining strict flag behavior
  through `is_bool=True`, which accepts booleans and numeric zero/one only.
- Migrated flag callers to strict boolean mode and added meaningful parameter
  names to ordinary numerical callers for clearer validation errors.
- Kept physical constraints at their owning APIs rather than turning the
  structural broadcaster into a collection of unrelated domain rules.

### `nematics3d.datatypes.BoxSizePeriodic` and `as_box_size_periodic`

| Field | Evidence |
| --- | --- |
| Kind | Public periodic-box semantic alias and three-axis input validator |
| Source | [`src/nematics3d/datatypes/box_size_periodic.py`](../../src/nematics3d/datatypes/box_size_periodic.py) |
| Tests | [`tests/test_datatypes_box_size_periodic.py`](../../tests/test_datatypes_box_size_periodic.py), [`tests/test_disclination_line_classification.py`](../../tests/test_disclination_line_classification.py), and downstream grid and line tests |
| Tutorial | None; this compact datatype helper is documented by its public docstring and the scientific functions that consume periodic boxes |
| Review scope | One shared or xyz-specific box size, positive finite periods, positive infinity as the non-periodic marker, mixed periodic axes, floating output dtype, independent storage, boolean/NaN/negative-infinity/zero/negative/complex/string/shape rejection, public exports, removal of `DimensionPeriodic` aliases and the redundant periodic-flag helper, grid and disclination caller migration, and lattice-index integer-period specialization |
| Validation | `python -m pytest tests/test_datatypes_box_size_periodic.py tests/test_disclination_line_classification.py tests/test_disclination_defect_detect.py tests/classes/test_grid_offset_none.py tests/classes/test_q_plane.py tests/classes/test_plane_grid.py tests/classes/test_plane_grid_polar.py tests/classes/test_bounds_obb.py tests/smooth/test_smooth.py tests/smooth/test_smoothed_line_func_registry.py -q` (74 passed); broader migration run (177 passed, 5 known unrelated `GridFieldDataset` failures); Black and `black --check` on all 10 modified Python files; in-memory compile of 142 Python files; stale-name search; `git diff --check` |
| Reviewed commit | `746bc12` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The generic datatype intentionally allows arbitrary positive floating periods. Defect classification separately requires integer-valued finite periods because its coordinates live in lattice-index space. Periodic trajectory unwrapping and related grid algorithms retain their existing numerical behavior and are scheduled for separate review. The broader run retained pre-existing missing-interpolator, Gaussian trailing-axis, and reserved-mask test failures. |

Summary of changes and evidence:

- Replaced the generic `DimensionInfo` boundary with an explicit periodic-box
  contract throughout grid and disclination APIs.
- Removed `DimensionPeriodic`, `DimensionPeriodicInput`, and
  `boundary_periodic_size_to_flag()` without compatibility aliases.
- Replaced the old flag helper with direct `numpy.isfinite()` masks after box
  validation.
- Corrected the default disclination-line box from boolean `False` to the
  explicit fully non-periodic value `numpy.inf`.
- Kept the generic physical-coordinate datatype permissive while making the
  stricter integer lattice-period requirement visible at defect classification.

### `nematics3d.datatypes.as_director_field` and `as_scalar_field`

| Field | Evidence |
| --- | --- |
| Kind | Public field-input validators with domain-specific `nField` and `SField` aliases |
| Source | [`src/nematics3d/datatypes/director_field.py`](../../src/nematics3d/datatypes/director_field.py) and [`scalar_field.py`](../../src/nematics3d/datatypes/scalar_field.py) |
| Tests | [`tests/test_datatypes_director_field.py`](../../tests/test_datatypes_director_field.py) and downstream Q-field tests |
| Tutorial | None; these are compact input helpers used by public scientific functions |
| Review scope | Arbitrary leading dimensions, optional strict 3D spatial shapes, real finite values, dtype-level fast paths, floating-dtype preservation, object-array fallback, avoidable norm and copy elimination, per-point director normalization, allowed and rejected zero directors, generic `ScalarField` output, domain-specific `SField`, validated replacement recovery, logging, naming, deletion of the mixed-purpose `check_Sn()`, and active callers |
| Validation | `python -m pytest tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py tests/classes/test_q_plane.py -q` (30 passed); `black --check src/nematics3d/datatypes.py src/nematics3d/disclination.py src/nematics3d/classes/q_field_object.py src/nematics3d/classes/q_plane.py tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py`; `ruff check tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py`; in-memory syntax compile; executed defect tutorial validation; `git diff --check` |
| Reviewed commit | `bdb7e25` |
| Reviewed date | 2026-08-25 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | `nField` and `SField` intentionally remain domain-style naming exceptions. Allowed director norms at or below `1e-12` are represented as zero during normalization. Physical value restrictions on scalar order are intentionally delegated to calling scientific functions. The wider `QFieldObject` test retains a pre-existing failure because `FieldData` has no `interpolator` attribute. |

Summary of changes and evidence:

- Split the mixed string-dispatched `check_Sn()` helper into explicit director
  and scalar-field validators and removed the old API without an alias.
- Preserved strict spatial-grid validation where existing callers required it
  while keeping field utilities compatible with arbitrary leading dimensions.
- Added generic `ScalarField` output semantics and retained `SField` only as a
  liquid-crystal scalar-order alias.
- Migrated `QFieldObject`, defect analysis, field construction, director
  alignment, and color mapping to the explicit validators.
- Replaced per-component Python type checks for ordinary `NumPy` arrays with
  dtype-level validation, preserved existing floating dtypes, and skipped norm
  calculation when neither normalization nor zero rejection requires it.

### `nematics3d.classes.result_base.ResultBase`

| Field | Evidence |
| --- | --- |
| Kind | Internal base class |
| Source | [`src/nematics3d/classes/result_base.py`](../../src/nematics3d/classes/result_base.py) |
| Tests | [`tests/core/test_q_diagonalization.py`](../../tests/core/test_q_diagonalization.py) |
| Tutorial | [`tutorials/classes/ResultBase/ResultBase.ipynb`](../../tutorials/classes/ResultBase/ResultBase.ipynb) |
| Review scope | Dataclass field discovery, attribute and key access, dictionary-like inspection, shallow dictionary conversion, field descriptions, representation, error behavior exercised by the concrete diagonalization result, documentation, formatting, and logging |
| Validation | `conda run -n Nematics3D pytest -q tests/core/test_q_diagonalization.py` (10 passed, 8 subtests passed); `black --check src/nematics3d/classes/result_base.py tests/core/test_q_diagonalization.py`; `ruff check --select E,F,W,N,I src/nematics3d/classes/result_base.py tests/core/test_q_diagonalization.py`; in-memory syntax compile; notebook validation; `git diff --check` |
| Reviewed commit | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` |
| Reviewed date | 2026-08-24 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Validation intentionally uses `QDiagonalizationResult` as the sole concrete result subclass. Other result subclasses and their scientific functions are outside this review. The existing `get()` behavior was retained by explicit decision. |

Summary of changes and evidence:

- Confirmed that `ResultBase` is an internal base class; concrete result
  subclasses are the user-facing interfaces.
- Reviewed every inherited inspection and representation method without
  changing the accepted `get()` behavior.
- Verified inherited behavior through the real `QDiagonalizationResult`
  returned by `q_diagonalize()`, rather than maintaining a separate synthetic
  result test file.
- Documented the user interface, repository-developer subclassing conventions,
  direct construction relationships, logging decision, and the restriction
  against field names that hide inherited interface methods.
- The focused diagonalization test file passes all ten tests and eight
  parameterized subtests, and the reviewed source and test file pass Black,
  Ruff, syntax, notebook, and whitespace validation.

### `nematics3d.analysis.q_diagonalization.q_diagonalize`

| Field | Evidence |
| --- | --- |
| Kind | Public scientific function with private Python and compiled C backends |
| Source | [`src/nematics3d/analysis/q_diagonalization/`](../../src/nematics3d/analysis/q_diagonalization/) |
| Build configuration | [`setup.py`](../../setup.py), [`pyproject.toml`](../../pyproject.toml), and [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) |
| Tests | [`tests/core/test_q_diagonalization.py`](../../tests/core/test_q_diagonalization.py) and [`tests/core/test_datatypes_qfield.py`](../../tests/core/test_datatypes_qfield.py) |
| Tutorial | [`tutorials/analysis/q_diagonalization/q_diagonalize.ipynb`](../../tutorials/analysis/q_diagonalization/q_diagonalize.ipynb) |
| Review scope | Public Q5/Q9 contract, named result, principal-only and complete eigensystems, C and `NumExpr` backend selection, Python worker threading, isotropic classification, near-degenerate orthonormality, optional right-handed frames, validation and errors, logging, performance documentation, packaging, exports, and direct callers |
| Validation | `python -m pytest tests/core -q` (25 passed, 31 subtests passed); focused `tests/core/test_q_diagonalization.py` run (10 passed, 8 subtests passed); `black --check setup.py src/nematics3d/analysis/q_diagonalization src/nematics3d/classes/result_base.py tests/core`; `ruff check --select E,F,W,N,I setup.py src/nematics3d/analysis/q_diagonalization src/nematics3d/classes/result_base.py tests/core`; in-memory syntax compile; notebook JSON, code-cell, local-link, and stale-term validation; `git diff --check`; isolated `python -m build`; wheel installation and public-API smoke test outside the repository |
| Reviewed commit | `5089673` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The public scalar order is defined as $S=3\lambda_{\max}/2$, so negative-$S$ conventions for oblate or anti-nematic systems are unsupported. Isotropic directors are deterministic placeholders. Individual eigenvectors remain physically non-unique in degenerate subspaces, and director sign is intentionally unspecified. A dedicated `get_q()` round trip was added subsequently; additional downstream boundary tests remain deferred by maintainer decision. The wider non-visual suite remains blocked during collection by pre-existing `ClassBase`/`HostBase` test incompatibilities; Windows VTK cleanup also emits unrelated OpenGL errors. The clean local wheel build and smoke test covered Windows CPython 3.12; Linux compilation is configured in CI, while a macOS wheel was not built locally. |

Summary of changes and evidence:

- Replaced the earlier monolithic implementation with a focused analysis
  package containing a robust traceless symmetric eigensolver, an optional C
  extension, and a small backend/threading adapter.
- Confirmed C and `NumExpr` agreement, worker-count consistency, descending
  eigenvalue ordering, reconstruction, right-handed frames, isotropic
  conventions, empty-input rejection, and near-degenerate orthonormality
  against independent `numpy.linalg.eigh()` results.
- Kept the default low-memory path limited to the dominant eigenpair and
  documented the complete-eigensystem path, numerical design, sign and
  degeneracy conventions, backend selection, staged logging, and measured
  performance.
- Built both sdist and Windows CPython 3.12 wheel in an isolated build
  environment. Installed the wheel into a temporary environment outside the
  repository and successfully forced the compiled C principal and complete
  paths, right-handed conversion, and the `NumExpr` fallback.
- Updated public exports and direct callers, removed the obsolete compatibility
  module, and added Linux/Windows core CI plus an installed-wheel smoke job.

### `nematics3d.analysis.disclination.defect_detect`

| Field | Evidence |
| --- | --- |
| Kind | Public scientific function with a private multithreaded `NumExpr` plaquette kernel |
| Source | [`src/nematics3d/analysis/disclination/detection.py`](../../src/nematics3d/analysis/disclination/detection.py) |
| Legacy backup | [`dev/backup/defect_detection_legacy.py`](../backup/defect_detection_legacy.py) |
| Tests | [`tests/test_disclination_defect_detect.py`](../../tests/test_disclination_defect_detect.py) and [`tests/test_datatypes_director_field.py`](../../tests/test_datatypes_director_field.py) |
| Tutorial | [`tutorials/analysis/disclination/defect_detect.ipynb`](../../tutorials/analysis/disclination/defect_detect.ipynb) |
| Review scope | Three plaquette-normal directions, nematic sign-aligned closure criterion, non-periodic and periodic boundaries on all spatial axes, coordinate conventions, selected planes, empty output, `NumExpr` worker control, trusted-input bypass, director-field validation integration, public callers, logging decision, performance, documentation, and legacy equivalence |
| Validation | `python -m pytest tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py tests/classes/test_q_plane.py -q` (30 passed); focused defect file (19 passed); `black --check src/nematics3d/datatypes.py src/nematics3d/analysis/disclination/detection.py src/nematics3d/classes/q_field_object.py src/nematics3d/classes/q_plane.py tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py`; `ruff check tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py`; Ruff E/W/import validation for the reviewed detection implementation; in-memory syntax compile; executed notebook schema and code-cell validation; `git diff --check`; coordinate-set comparison with the archived implementation on `example/data/Q_example_workflow.npy` |
| Reviewed commit | `5089673` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Nonzero `threshold` behavior is retained for developers but intentionally lacks dedicated tests because current user workflows use the default zero criterion. `is_input_validated=True` deliberately trusts the caller. Explicit `worker_count` temporarily changes a process-wide `NumExpr` setting and should not be varied by concurrent calls. Periodic detection currently copies the extended field, adding about 49 MiB on the bundled example. Output is grouped by plaquette-normal axis rather than globally sorted. The wider suite remains blocked during collection by pre-existing `ClassBase`/`HostBase` test incompatibilities. |

Summary of changes and evidence:

- Replaced the stacked, repeatedly validated sign-alignment implementation
  with a fused `NumExpr` closure predicate and one shared periodic extension.
- Removed repeated normalization, per-plane validation, incremental `vstack`,
  and the final global `numpy.unique()` while preserving the legacy coordinate
  set across randomized plane and periodic-boundary combinations.
- Added explicit trusted-input and worker-count controls. `QFieldObject` and
  `QPlane` use the trusted path because their directors are already prepared.
- Added artificial single-defect and uniform-field tests, explicit defects
  crossing x, y, and z periodic boundaries, periodic-normal duplicate checks,
  invalid-input tests, and legacy equivalence tests.
- On the bundled two-million-point example, the archived algorithm with the
  optimized validator took about 1.30 s, the new validated path about 0.12 s
  with one worker, and the trusted path about 0.11 s. With the current default
  16-thread `NumExpr` setting, the trusted path measured about 0.020 s.
- Intentionally omitted the logging decorator: the function has no useful
  progress messages, and higher-level workflow owners provide user-facing
  logging where needed.
- Documented the algorithm, lattice-coordinate convention, plane selection,
  periodic boundaries, trusted path, worker behavior, and a complete
  $Q$-tensor-to-defect example in an executed tutorial.

### `nematics3d.datatypes.as_qfield5` and `as_qfield9`

| Field | Evidence |
| --- | --- |
| Kind | Public Q-field representation validators and converters |
| Source | [`src/nematics3d/datatypes/q_field.py`](../../src/nematics3d/datatypes/q_field.py) |
| Tests | [`tests/core/test_datatypes_qfield.py`](../../tests/core/test_datatypes_qfield.py) |
| Tutorial | [`tutorials/analysis/q_diagonalization/q_diagonalize.ipynb`](../../tutorials/analysis/q_diagonalization/q_diagonalize.ipynb) |
| Review scope | Compact five-component and full symmetric-traceless 3x3 representations, strict 3D and relaxed leading dimensions, dtype and finite-value validation, symmetry and trace tolerances, empty relaxed inputs, conversion behavior, zero-copy same-representation returns, public exports, and diagonalization integration |
| Validation | `python -m pytest tests/core/test_datatypes_qfield.py -q` (15 passed, 23 subtests passed); focused datatypes and disclination regression run (68 passed, 23 subtests passed); Black; in-memory syntax and import checks; `git diff --check` |
| Reviewed commit | `35af036` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Semantic aliases remain reader-facing NumPy aliases rather than statically shape-aware types. Full QField9 validation scans the complete input for finite values, symmetry, and trace; callers with an already validated internal tensor may explicitly skip numerical validation. |

### `nematics3d.datatypes.DefectIndex` and `as_defect_index`

| Field | Evidence |
| --- | --- |
| Kind | Public defect-index semantic alias and strict canonicalizer |
| Source | [`src/nematics3d/datatypes/defect_index.py`](../../src/nematics3d/datatypes/defect_index.py) |
| Tests | [`tests/test_datatypes_defect_index.py`](../../tests/test_datatypes_defect_index.py) and downstream line-classification tests |
| Tutorial | None; the coordinate convention is documented by the defect-analysis tutorials and public scientific functions |
| Review scope | Shape `(N, 3)`, empty collections, real numeric dtype, finite values, integer/half-integer lattice structure, configurable non-negative finite tolerance, canonical half-grid snapping, error row reporting, PEP 8 naming, public exports, and classification integration |
| Validation | `python -m pytest tests/test_datatypes_defect_index.py -q` (12 passed); `python -m pytest tests/test_disclination_line_classification.py -q` (11 passed); combined datatypes and disclination regression run (68 passed, 23 subtests passed); Black; in-memory syntax and import checks; `git diff --check` |
| Reviewed commit | `35af036` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | `DefectIndex` intentionally denotes the complete `(N, 3)` collection rather than one `(3,)` row. Periodic wrapping and the doubled-integer graph encoding are algorithm-specific and remain outside this converter. |

### `nematics3d.analysis.disclination.defect_classify_into_lines`

| Field | Evidence |
| --- | --- |
| Kind | Public defect-line classification function with private vectorized graph helpers |
| Source | [`src/nematics3d/analysis/disclination/classification.py`](../../src/nematics3d/analysis/disclination/classification.py) |
| Legacy backup | [`dev/backup/defect_line_classification_legacy.py`](../backup/defect_line_classification_legacy.py) |
| Tests | [`tests/test_disclination_line_classification.py`](../../tests/test_disclination_line_classification.py) and [`tests/test_datatypes_defect_index.py`](../../tests/test_datatypes_defect_index.py) |
| Tutorial | [`tutorials/analysis/disclination/defect_classify_into_lines.ipynb`](../../tutorials/analysis/disclination/defect_classify_into_lines.ipynb) |
| Review scope | Half-grid canonicalization, periodic coordinate wrapping, duplicate rejection, vectorized neighbor-edge construction, the exact ten legal dual-lattice continuations for each defect plaquette, adjacency and Euler-trail extraction, open and closed lines, branched graphs, periodic-boundary lines, grid transforms and offsets, deterministic line construction, public callers, legacy equivalence, performance, logging, and reader-facing documentation |
| Validation | `python -m pytest tests/test_disclination_defect_detect.py tests/test_disclination_line_classification.py tests/test_datatypes_defect_index.py -q` (43 passed); bundled-example comparison against the archived classifier (1270 defects and 8 equivalent lines); complete execution of the classification tutorial; JSON and local-link validation of six affected tutorials (no broken links); Black; in-memory syntax and import checks; `git diff --check` |
| Reviewed commit | `b40d6a5` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | This is intentionally not a general point-cloud clustering algorithm. It accepts the canonical half-grid defect-index geometry and connects only dual-lattice links sharing an endpoint. Branched graphs are represented as deterministic maximal trails rather than as a single simple line. |

### `nematics3d.datatypes.ColorRGB`, `as_ColorRGB`, and `as_ColorRGB_array`

| Field | Evidence |
| --- | --- |
| Kind | Public RGB semantic annotation and scalar/array color validators |
| Source | [`src/nematics3d/datatypes/color_rgb.py`](../../src/nematics3d/datatypes/color_rgb.py) |
| Tests | [`tests/test_datatypes_color_rgb.py`](../../tests/test_datatypes_color_rgb.py) |
| Tutorial | None; these are compact input helpers used by visualization APIs |
| Review scope | Scalar `(3,)` and array `(N, 3)` shapes, real numeric and finite values, inclusive `[0, 1]` range, copy behavior, scalar and broadcast replacement recovery, replacement revalidation, historical sum-of-powers normalization, near-zero normalization, and public exports |
| Validation | Focused contract review and direct smoke checks during the datatype cleanup; dedicated pytest coverage added in `tests/test_datatypes_color_rgb.py` for valid inputs, shape errors, range/finite/complex failures, normalization, zero normalization, scalar replacement, array replacement, and structural-error behavior |
| Reviewed commit | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | Normalization intentionally preserves the package's historical sum-of-powers rule rather than conventional Lp normalization. Array replacement intentionally recovers invalid values only; invalid outer structure or shape remains an immediate error. |

Summary of changes and evidence:

- Extracted RGB handling from `misc.py` into a dedicated datatype module while
  preserving the public names and compatibility re-export.
- Fixed the previous control-flow bug where valid scalar RGB input skipped
  normalization even when `is_norm=True`.
- Unified real/finite/range validation and replacement revalidation between
  scalar and array forms.

### `nematics3d.datatypes.as_str`

| Field | Evidence |
| --- | --- |
| Kind | Public scalar string validator |
| Source | [`src/nematics3d/datatypes/string.py`](../../src/nematics3d/datatypes/string.py) |
| Tests | None; the helper is intentionally treated as a compact direct-contract validator |
| Tutorial | None |
| Review scope | String type validation, optional membership pool, replacement recovery, replacement revalidation against the same type and pool contract, error messages, and public exports |
| Validation | Direct contract review of ordinary strings, pool-constrained strings, invalid input recovery, non-string replacement rejection, and out-of-pool replacement rejection |
| Reviewed commit | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | No dedicated pytest file was added because the implementation is small and deterministic; downstream option/name validators provide additional coverage. |

Summary of changes and evidence:

- Extracted `as_str()` from `misc.py` into a dedicated module.
- Made replacement values pass the same string and optional-pool validation as
  ordinary input instead of allowing an unchecked fallback.

### `nematics3d.datatypes.as_axes`

| Field | Evidence |
| --- | --- |
| Kind | Public 3D orthonormal-frame validator and right-handed normalizer |
| Source | [`src/nematics3d/datatypes/axes.py`](../../src/nematics3d/datatypes/axes.py) |
| Tests | [`tests/test_datatypes_axes.py`](../../tests/test_datatypes_axes.py) |
| Tutorial | None; this is a compact geometric input helper |
| Review scope | Exact `(3, 3)` shape, real finite values, orthonormal columns, absolute-tolerance semantics, right- and left-handed frames, optional right-handed conversion, copy behavior, complex rejection, and parameter validation |
| Validation | Focused contract review and dedicated pytest coverage in `tests/test_datatypes_axes.py` for identity/general frames, copy semantics, left-handed handling, tolerance behavior, invalid shape/type/complex/finite/orthogonality inputs, and invalid option parameters |
| Reviewed commit | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | The helper validates an orthonormal frame stored by columns but does not encode shape or orthogonality statically in its type annotation. |

Summary of changes and evidence:

- Extracted `as_axes()` from `misc.py` and removed the low-value `Axes` and
  `AxesInput` aliases.
- Rejected complex and non-finite inputs explicitly, made `atol` a finite
  non-negative real contract, and made `rtol=0` so `atol` alone controls the
  orthogonality tolerance.
- Retained optional conversion to a right-handed frame by flipping the final
  axis when needed.

### `nematics3d.datatypes.as_list`

| Field | Evidence |
| --- | --- |
| Kind | Public single-or-multiple list normalizer |
| Source | [`src/nematics3d/datatypes/list.py`](../../src/nematics3d/datatypes/list.py) |
| Tests | [`tests/test_datatypes_list.py`](../../tests/test_datatypes_list.py) |
| Tutorial | None |
| Review scope | Existing-list identity, tuple/set expansion, scalar wrapping, and deliberate treatment of strings, ranges, generators, NumPy arrays, and `None` as single objects |
| Validation | Direct contract review plus focused tests in `tests/test_datatypes_list.py` for list identity, tuple/set expansion, scalar-like iterable handling, generators, and `None` |
| Reviewed commit | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and ChatGPT |
| Remaining limitations | Sets are accepted and therefore lose deterministic ordering. General iterables are intentionally not expanded; only tuples and sets are treated as multi-item inputs. |

Summary of changes and evidence:

- Extracted `as_list()` from `misc.py` into a dedicated module.
- Simplified it from an exception/replacement/logging wrapper into a small
  deterministic normalization helper matching its actual repository use.

### `nematics3d.datatypes.as_points`

| Field | Evidence |
| --- | --- |
| Kind | Public point-collection validator and normalizer |
| Source | [`src/nematics3d/datatypes/points.py`](../../src/nematics3d/datatypes/points.py) |
| Tests | [`tests/test_datatypes_points.py`](../../tests/test_datatypes_points.py), [`tests/test_geometry_obb.py`](../../tests/test_geometry_obb.py), and downstream bounds, smoothing, interpolation, grid, and geometry tests |
| Tutorial | None; this is a compact geometric input helper |
| Review scope | Single-point promotion, `(N, d)` shape semantics, arbitrary dimensions through `d=None`, empty-input normalization and policy, real finite coordinates by default, optional non-finite values, duplicate removal, minimum point count after deduplication, independent floating output, boolean option validation, PEP 8 parameter naming, dedicated module, public exports, and active callers |
| Validation | `python -m pytest tests/test_datatypes_points.py tests/test_datatypes_number.py -q` (38 passed); combined datatype and downstream run (67 passed, with one unrelated `as_list(name=...)` failure and two unrelated zero-tolerance OBB assertions); Black on all modified Python files; in-memory compile of 99 source files; active-source `dim=` caller search; `git diff --check` |
| Reviewed commit | `c7443c2343dc31c700db9257f3f4125a517e4533` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | With `d=None`, dimensionless empty input is normalized to shape `(0, 0)` because no point dimension can be inferred. `is_unique=True` uses `numpy.unique()` and therefore returns points in lexicographic rather than original order. |

Summary of changes and evidence:

- Extracted `as_points()` from `misc.py` into a dedicated datatype module and
  migrated every active caller from `dim=` to the repository-standard `d=`.
- Distinguished structural `ValueError` failures from coordinate-type
  `TypeError` failures instead of wrapping every invalid input as `TypeError`.
- Added explicit finite, empty, uniqueness, and minimum-count policies, with
  the minimum count evaluated after optional deduplication.
- Added focused tests for single points, arbitrary dimensions, empty inputs,
  independent output, non-finite opt-in, invalid coordinate types, dimension
  validation, and deduplicated minimum counts.

### `nematics3d.datatypes.as_value_range`

| Field | Evidence |
| --- | --- |
| Kind | Public inclusive numeric-interval validator |
| Source | [`src/nematics3d/datatypes/number.py`](../../src/nematics3d/datatypes/number.py) |
| Tests | [`tests/test_datatypes_number.py`](../../tests/test_datatypes_number.py) |
| Tutorial | None; this is a compact scalar-range helper used by `as_number()` |
| Review scope | Exact two-value shape, real numeric dtype, complex and boolean rejection, NaN rejection, strictly increasing bounds, infinite open-ended bounds, Python-float tuple output, parameter-name errors, public export, and `as_number()` range and clipping integration |
| Validation | `python -m pytest tests/test_datatypes_points.py tests/test_datatypes_number.py -q` (38 passed); direct inspection of `as_number()` inclusive-range, clipping, integer-boundary, and replacement integration; in-memory compile of 99 source files; `git diff --check` |
| Reviewed commit | `c7443c2343dc31c700db9257f3f4125a517e4533` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Bounds are strictly increasing, so a zero-width interval such as `(1, 1)` is intentionally invalid. Positive and negative infinity are allowed when their ordering defines a meaningful open-ended interval. |

Summary of changes and evidence:

- Recorded `as_value_range()` separately from `as_number()` so the public
  helper's own interval contract is visible in the reviewed-component ledger.
- Confirmed exact shape, real dtype, NaN, ordering, and conversion behavior,
  together with its inclusive-range and clipping integration in `as_number()`.

### `nematics3d.grid.shift_to_box`

| Field | Evidence |
| --- | --- |
| Kind | Public whole-trajectory periodic-box translation utility |
| Source | [`src/nematics3d/grid/periodic.py`](../../src/nematics3d/grid/periodic.py) |
| Tests | [`tests/test_grid_periodic.py`](../../tests/test_grid_periodic.py), with datatype validation covered by [`tests/test_datatypes_box_size_periodic.py`](../../tests/test_datatypes_box_size_periodic.py) and [`tests/test_datatypes_points.py`](../../tests/test_datatypes_points.py) |
| Tutorial | None; the helper is documented by its public docstring and is used internally by the [`unwrap_trajectory()` tutorial](../../tutorials/grid/periodic/unwrap_trajectory.ipynb) |
| Review scope | Whole-trajectory translation by integer box periods, mixed periodic and non-periodic axes, selectable positive or negative reference index, copy-by-default behavior, explicit in-place mutation, strict writable floating-array requirements for in-place operation, empty and malformed input rejection, boolean option validation, trusted `is_validate=False` fast path, and public exports |
| Validation | `python -m pytest tests/test_grid_periodic.py tests/test_datatypes_box_size_periodic.py tests/test_datatypes_points.py -q` (39 passed); Black check of the implementation and focused tests; execution and link validation of the periodic-trajectory tutorial; `git diff --check` |
| Reviewed commit | `c7443c2` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The function translates the complete trajectory by whole periods; it does not repair discontinuities between successive points. The `is_validate=False` path deliberately gives undefined behavior for malformed arrays and is intended only for validated internal callers. |

### `nematics3d.grid.unwrap_trajectory`

| Field | Evidence |
| --- | --- |
| Kind | Public minimum-image periodic-trajectory reconstruction utility |
| Source | [`src/nematics3d/grid/periodic.py`](../../src/nematics3d/grid/periodic.py) |
| Tests | [`tests/test_grid_periodic.py`](../../tests/test_grid_periodic.py), with datatype validation covered by [`tests/test_datatypes_box_size_periodic.py`](../../tests/test_datatypes_box_size_periodic.py) and [`tests/test_datatypes_points.py`](../../tests/test_datatypes_points.py) |
| Tutorial | [`tutorials/grid/periodic/unwrap_trajectory.ipynb`](../../tutorials/grid/periodic/unwrap_trajectory.ipynb) |
| Review scope | Minimum-image correction of consecutive displacements, mixed periodic axes, scalar and xyz box sizes, input isolation, empty and single-point trajectories, reverse anchoring, optional translation of a selected reference point into the principal box, positive and negative reference indices, validation failures, internal use of the trusted in-place `shift_to_box()` path, debug logging, public exports, and disclination-line classification integration |
| Validation | `python -m pytest tests/test_grid_periodic.py tests/test_datatypes_box_size_periodic.py tests/test_datatypes_points.py -q` (39 passed); Black check of the implementation and focused tests; complete execution of the 17-cell tutorial and local-link validation (no broken links); `git diff --check` |
| Reviewed commit | `c7443c2` |
| Reviewed date | 2026-08-26 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The minimum-image convention assumes every true step is shorter than half the corresponding periodic length. An exact half-period displacement is direction-ambiguous and follows `numpy.round()` tie-to-even behavior. Unwrapping preserves continuity and may intentionally return coordinates outside the principal box. |

### `nematics3d.grid.generate_coordinate_grid` and `generate_fixed_step_grid`

| Field | Evidence |
| --- | --- |
| Kind | Public dense coordinate-grid generators |
| Source | [`src/nematics3d/grid/coordinate.py`](../../src/nematics3d/grid/coordinate.py) |
| Tests | [`tests/test_grid_coordinate.py`](../../tests/test_grid_coordinate.py), with direct caller coverage in [`tests/classes/test_grid_field_dataset.py`](../../tests/classes/test_grid_field_dataset.py), [`tests/classes/test_plane_grid.py`](../../tests/classes/test_plane_grid.py), and [`tests/classes/test_plane_grid_polar.py`](../../tests/classes/test_plane_grid_polar.py) |
| Tutorial | [`tutorials/grid/coordinate/generate_coordinate_grid.ipynb`](../../tutorials/grid/coordinate/generate_coordinate_grid.ipynb) and [`tutorials/grid/coordinate/generate_fixed_step_grid.ipynb`](../../tutorials/grid/coordinate/generate_fixed_step_grid.ipynb) |
| Review scope | N-dimensional source-index resampling coordinates, identity grids, endpoint preservation, one-sample target axes, two-dimensional fixed-step grids, bottom-left and center alignment, integer topology, effective extents, zero sizes, Python and NumPy scalar inputs, invalid inputs, decimal step-boundary rounding, dense-memory behavior, public exports, active callers, documentation, and the explicit decision that these lightweight deterministic functions need no logger |
| Validation | `python -m pytest tests/test_grid_coordinate.py -q` (31 passed); focused caller run covering `GridFieldDataset`, `PlaneGrid`, and `PlaneGridPolar` (38 passed); broader six-file caller run (121 passed, with 6 unrelated pre-existing failures in interpolator relations, Gaussian smoothing of trailing component axes, reserved-mask expectations, and legacy Q-field interpolation); Black and `black --check` on the implementation and affected tests; complete in-memory execution of both tutorials; notebook JSON validation; `git diff --check` |
| Reviewed commit | `2b54edaa83e084f999fb415dddff6914dd256dfb` |
| Reviewed date | 2026-08-27 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Both functions intentionally allocate dense coordinate arrays and can exhaust memory for impractically large requested grids. Their repository validators `as_grid_shape()`, `as_number()`, and `as_str()` are separately confirmed. |

Summary of changes and evidence:

- Confirmed the source-index coordinate convention, dimensionality, shapes,
  dtypes, endpoint behavior, fixed-step alignment, and effective-size contracts.
- Corrected fixed-step snapping at decimal boundaries such as `0.3 / 0.1`,
  where raw binary floating-point division can fall one representable value
  below an exact integer before `floor()` is applied.
- Removed a stale downstream test assumption from the former tuple-return API
  and confirmed the active `GridFieldDataset` and `PlaneGrid` callers.
- Split and expanded the public tutorials, including memory, topology,
  alignment, implementation, and common-misuse guidance.
- Kept both functions undecorated because their operations are lightweight and
  deterministic, their validation errors are already explicit, and logging is
  more useful at the higher-level workflow boundary.

### `nematics3d.datatypes.as_grid_shape`

| Field | Evidence |
| --- | --- |
| Kind | Public ordered grid-shape validator and normalizer |
| Source | [`src/nematics3d/datatypes/grid_shape.py`](../../src/nematics3d/datatypes/grid_shape.py) |
| Tests | [`tests/test_datatypes_grid_shape.py`](../../tests/test_datatypes_grid_shape.py), with coordinate-grid and `InputGridField` caller coverage |
| Tutorial | None; this is a compact input validator whose complete contract is documented in its docstring and focused tests |
| Review scope | Ordered iterable inputs, tuple/list/NumPy-array/generator support, Python and NumPy integers, Python-int tuple output, arbitrary positive dimensionality, strict three-dimensional mode, empty and non-positive shapes, boolean/float/complex rejection, explicit mapping/set/string/bytes rejection, parameter-specific errors, public exports, internal imports, active callers, and logging decision |
| Validation | `python -m pytest tests/test_datatypes_grid_shape.py tests/test_grid_coordinate.py -q` (55 passed); focused `InputGridField`, coordinate, plane-grid, and contour-surface caller run (73 passed); broader related run (184 passed, with 6 unrelated pre-existing failures in interpolator relations, Gaussian smoothing of trailing component axes, reserved-mask expectations, and legacy Q-field interpolation); Black and `black --check` on the implementation, focused tests, and affected callers; in-memory syntax compile; active-source import audit; `git diff --check` |
| Reviewed commit | `fa2239e7681b262cea48d01fa580452f2e5d851e` |
| Reviewed date | 2026-08-27 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The validator intentionally does not impose an upper bound on dimensions; downstream array constructors remain responsible for resource limits. Generator inputs are consumed once during normalization. |

Summary of changes and evidence:

- Defined the accepted input as an ordered iterable of positive integers and
  explicitly rejected mappings and sets, whose iteration order is not a valid
  grid-axis contract.
- Preserved tuple, list, NumPy integer array, NumPy integer scalar, and generator
  support while guaranteeing a tuple of Python integers.
- Added focused tests for main paths, strict three-dimensional mode, invalid
  containers, empty and non-positive shapes, invalid dimension types, and the
  validated strictness flag.
- Updated active internal callers to import the validator directly from
  `nematics3d.datatypes` while preserving the existing
  `nematics3d.classes.grid_field` re-export.
- Kept the function undecorated because this lightweight deterministic
  validator has no useful intermediate state or workflow boundary to log. Its
  only repository-function dependency is the already confirmed `as_bool()`.

### `nematics3d.field.get_q`

| Field | Evidence |
| --- | --- |
| Kind | Public uniaxial and biaxial Q-tensor field constructor |
| Source | [`src/nematics3d/field.py`](../../src/nematics3d/field.py) |
| Tests | [`tests/core/test_get_q.py`](../../tests/core/test_get_q.py), with downstream initialization coverage in [`tests/classes/test_q_plane.py`](../../tests/classes/test_q_plane.py) and [`tests/classes/test_q_field_object_phase2.py`](../../tests/classes/test_q_field_object_phase2.py) |
| Tutorial | None; the physical convention, parameter pairing, broadcasting, and failure behavior are documented in the function docstring and focused scientific tests |
| Review scope | Uniaxial convention $Q=S(nn-I/3)$, optional signed biaxial contribution $P(mm-ll)$ with $l=n\times m$, director normalization, default unit scalar order, scalar and field broadcasting, symmetric-traceless invariants, director-sign invariance, positive and negative biaxial order, orthogonality tolerance, paired `m`/`P` inputs, zero directors, incompatible shapes, input isolation, floating output, top-level public export, `q_diagonalize()` round trip, active callers, and logging decision |
| Validation | `python -m pytest tests/core/test_get_q.py tests/core/test_q_diagonalization.py tests/classes/test_q_plane.py -q` (23 passed, 8 subtests passed); `tests/classes/test_q_field_object_phase2.py` downstream run (10 passed, with 1 unrelated pre-existing `FieldData.interpolator` relation failure); Black and `black --check` on the implementation and focused tests; in-memory syntax compile; active-source caller and export audit; `git diff --check` |
| Reviewed commit | `70871e394effca0755983efe6888a377180871ea` |
| Reviewed date | 2026-08-28 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Biaxial directors must already be orthogonal within an absolute dot-product tolerance of `1e-8`; the function rejects non-orthogonal pairs rather than orthogonalizing them. Director and scalar validation intentionally normalizes inputs to the repository's standard floating representation. |

Summary of changes and evidence:

- Confirmed the uniaxial and signed biaxial physical conventions through exact
  diagonal examples, symmetric-traceless invariants, and director-sign
  equivalence.
- Confirmed broadcasting for uniaxial and biaxial fields, preservation of input
  arrays, floating output, zero-director rejection, and paired `m`/`P` failure
  behavior.
- Reconstructed a randomized biaxial tensor from the complete
  `q_diagonalize()` result and recovered the original tensor within numerical
  tolerance.
- Confirmed both `nematics3d.field.get_q` and the top-level `nematics3d.get_q`
  public surface, together with the `QFieldObject` and principal-plane callers.
- Kept the function undecorated because it is a deterministic vectorized
  tensor construction with no useful workflow event or recovery path to log.
  Its repository validators `as_director_field()` and `as_scalar_field()` are
  already confirmed.

### `defect_validity_from_mask()`

| Field | Evidence |
| --- | --- |
| Kind | Public defect-plaquette validity filter |
| Source | [`src/nematics3d/analysis/disclination/misc.py`](../../src/nematics3d/analysis/disclination/misc.py) |
| Tests | [`tests/test_disclination_defect_validity.py`](../../tests/test_disclination_defect_validity.py), with initialization integration coverage in [`tests/classes/test_q_field_object_phase2.py`](../../tests/classes/test_q_field_object_phase2.py) |
| Tutorial | None; the plaquette geometry, mask convention, periodic wrapping, ordering, and failure behavior are documented in the function docstring and focused tests |
| Review scope | All three plaquette orientations, all four supporting corners, unrelated invalid voxels, mixed-result input ordering, boolean result shape and dtype, input isolation, periodic wrapping across two axes, non-periodic bounds diagnostics, empty input, public export, `QFieldObject` initialization mask filtering, active callers, and logging decision |
| Validation | Focused validity and `QFieldObject` mask-integration run (8 passed); defect detection/classification regression run (38 passed); full phase-2 Q-field run (18 passed, with 1 unrelated pre-existing `FieldData.interpolator` relation failure); Black on the implementation and new focused test, in-memory syntax compile, active-caller/export/dependency audit, and `git diff --check` |
| Reviewed commit | `7e03526ffe198d1970543661f698770942dc6e9e` |
| Reviewed date | 2026-08-28 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The mask must be a three-dimensional lattice mask with the same lattice shape used to produce the defect indices; shape agreement is enforced indirectly by corner bounds rather than by a separate grid-shape parameter. Its repository dependencies `as_lattice_mask()`, `as_defect_index()`, and `as_dimension_info()` are separately confirmed. |

Summary of changes and evidence:

- Confirmed that a defect is valid exactly when all four vertices of its
  supporting plaquette are valid, for plaquettes normal to each lattice axis.
- Confirmed periodic corner wrapping, empty input, stable input order, boolean
  output, and preservation of caller-owned arrays.
- Improved non-periodic bounds errors to identify every offending axis and its
  corner range.
- Confirmed the initialization path moves mask-supported invalid detections to
  `calc_defect_indices_masked` while retaining valid detections for downstream
  analysis.
- Kept the function undecorated because it is a deterministic vectorized
  boolean filter; useful defect counts and mask-filtering events are already
  logged by its workflow-level `QFieldObject.act_defect_detect()` caller.

### `nematics3d.datatypes.GeneralField` and `as_real_lattice_field()`

| Field | Evidence |
| --- | --- |
| Kind | Public semantic alias and runtime converter for real-valued lattice fields |
| Source | [`src/nematics3d/datatypes/lattice_field.py`](../../src/nematics3d/datatypes/lattice_field.py) |
| Tests | [`tests/test_datatypes_lattice_field.py`](../../tests/test_datatypes_lattice_field.py), with Fourier, contour-surface, grid-dataset, mask, and defect downstream coverage |
| Tutorial | None; lattice-axis semantics, component-axis rules, finite-value behavior, range handling, and copy behavior are documented in the function docstring and focused tests |
| Review scope | Minimum three lattice axes, exact trailing-component-axis count, standard complete-shape validation, nonempty axes, real numeric input, integer-to-float normalization, compatible floating-input reuse, finite-value enforcement, optional NaN/infinity preservation, inclusive finite-value ranges, rejection versus clipping, non-finite preservation during clipping, caller-input isolation when clipping, bool-like options, public datatype export, active physical-field callers, and logging decision |
| Validation | Complete lattice-field datatype run (23 passed); contour and defect-mask downstream run (18 passed); Fourier transform run excluding unrelated correlation/distance failures (20 passed, 7 deselected); full Fourier/contour/defect run (40 passed, with 5 unrelated pre-existing correlation shape failures); full grid-dataset run (102 passed, with 5 unrelated pre-existing interpolator/smoothing/dynamic-mask failures); Black and `black --check`; in-memory syntax compile; active-caller/export/dependency audit; `git diff --check` |
| Reviewed commit | `63cb2c3b47ae847480a89f921b93da7eb1b41879` |
| Reviewed date | 2026-08-28 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | This converter validates generic real lattice data, not domain-specific physics such as director normalization, Q-tensor symmetry/tracelessness, or binary mask values. Compatible floating input may share memory with the caller by design; clipping returns an independent result. Complete `shape` validation includes component axes as well as the first three lattice axes. |

Summary of changes and evidence:

- Replaced the private exact-shape validator with the confirmed public
  `as_grid_shape()` contract while preserving complete-array shape checks.
- Fixed range handling so `is_finite=False` genuinely preserves NaN and
  positive/negative infinity, including when finite values are clipped.
- Confirmed scalar, vector, and tensor lattice layouts through exact
  component-axis counts without conflating this converter with the arbitrary
  rank accepted by `as_scalar_field()`.
- Confirmed integer normalization, zero-copy compatible floating input,
  independent clipped output, finite-value errors, and bool-like option rules.
- Kept the converter undecorated because it is a deterministic array validator;
  successful conversions are too frequent and lightweight to be useful log
  events, while validation exceptions already identify the failed contract.

### `nematics3d.datatypes.MaskField` and `as_lattice_mask()`

| Field | Evidence |
| --- | --- |
| Kind | Public semantic alias and runtime converter for lattice validity masks |
| Source | [`src/nematics3d/datatypes/lattice_field.py`](../../src/nematics3d/datatypes/lattice_field.py) |
| Tests | [`tests/test_datatypes_lattice_field.py`](../../tests/test_datatypes_lattice_field.py), with defect-mask coverage in [`tests/test_disclination_defect_validity.py`](../../tests/test_disclination_defect_validity.py) and Q-field initialization coverage in [`tests/classes/test_q_field_object_phase2.py`](../../tests/classes/test_q_field_object_phase2.py) |
| Tutorial | None; the physical validity convention and normalization behavior are documented in the function docstring and focused tests |
| Review scope | `True`/`False` physical convention, boolean and exact numeric 0/1 input, rejection of fractional, non-finite, complex, and non-numeric values, exact three-axis rank, nonempty axes, standard strict grid-shape validation, shape mismatch, independent boolean output, public datatype export, dataset construction, Q-field initialization, defect filtering, active callers, and logging decision |
| Validation | Mask-focused datatype run (12 passed); defect validity and Q-field mask-initialization run (8 passed); grid-dataset and defect downstream run (133 passed, with 5 unrelated pre-existing interpolator/smoothing/dynamic-mask failures); Black and `black --check`; in-memory syntax compile; active-caller/export/dependency audit; `git diff --check` |
| Reviewed commit | `d28dfc2529c64fbd26bf56ba86adc1b84cb00400` |
| Reviewed date | 2026-08-28 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Masks are deliberately binary validity fields, not continuous confidence weights. Its numeric-field dependency `as_real_lattice_field()` and the dependencies `as_bool()`, `as_value_range()`, and `as_grid_shape()` are separately confirmed. |

Summary of changes and evidence:

- Made the physical convention explicit: `True` means usable physical field
  data and `False` means downstream analysis must exclude that voxel.
- Unified boolean and numeric inputs on one rank, nonempty-axis, and optional
  shape-validation path.
- Replaced the mask-specific private shape rules with the confirmed strict
  three-dimensional `as_grid_shape()` contract.
- Confirmed exact 0/1 normalization, rejection of uncertain weights, output
  isolation, dataset shape agreement, Q-field initialization, and downstream
  defect filtering.
- Kept the converter undecorated because validation failures already contain
  actionable context and successful per-array conversion is not a useful
  workflow event to log.

### `nematics3d.grid` transform utilities

| Field | Evidence |
| --- | --- |
| Kind | Public grid-transform annotation, identity sentinel, validators, and forward/inverse coordinate mapping |
| Source | [`src/nematics3d/grid/transform.py`](../../src/nematics3d/grid/transform.py) |
| Tests | [`tests/test_grid_transform.py`](../../tests/test_grid_transform.py), with shared-grid geometry coverage in [`tests/classes/test_grid_field_dataset.py`](../../tests/classes/test_grid_field_dataset.py) and downstream periodic, defect-line, contour, bounds, plane-grid, and Q-field tests |
| Tutorial | [`tutorials/grid/transform/apply_linear_transform.ipynb`](../../tutorials/grid/transform/apply_linear_transform.ipynb) |
| Review scope | `GridTransform`, `GRID_TRANSFORM_IDENTITY`, identity canonicalization, three-dimensional offsets, read-only snapshots, the row-vector convention $x_{physical}=x_{index}T+o$, transform rows as scaled lattice-basis vectors, arbitrary leading point shapes, empty points, forward and inverse maps, anisotropic rotation, row orthogonality, right-handedness, reflection/shear/degeneracy rejection, physical-scale-stable direction checks, grid-spacing extraction, input validation, public exports, active callers, and logging decision |
| Validation | Focused transform and shared-grid geometry run (19 passed); broader periodic, defect classification, contour, bounds, plane-grid, and Q-field caller run (63 passed, with 1 unrelated pre-existing `FieldData.interpolator` relation failure); Black and `black --check` on the implementation and affected tests; complete in-memory execution of the transform tutorial; notebook JSON validation; in-memory syntax compile; active-source row/column convention audit; `git diff --check` |
| Reviewed commit | `7b3ec3a91ec30e7d7003c4570c000650ce55be12` |
| Reviewed date | 2026-08-28 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Grid transforms intentionally support only finite right-handed orthogonal bases with independent row scales. General shear, reflection, and degenerate affine maps are rejected. A row length at or below the absolute `1e-12` degeneracy threshold is unsupported. |

Summary of changes and evidence:

- Unified validation, application, spacing, tests, and tutorial documentation on
  the repository row-vector convention: row `i` of `T` is the physical basis
  vector reached by one unit step along lattice-index axis `i`.
- Added a direct rotated-basis test so a forward/inverse round trip can no
  longer conceal a transposed convention, and confirmed the shared-grid
  coordinates and spacing use the same basis rows.
- Made orthogonality and handedness checks operate on normalized row directions,
  avoiding scale-dependent acceptance and rejection for otherwise equivalent
  physical transforms.
- Confirmed identity and offset-only paths, read-only storage, arbitrary leading
  point shapes, invalid inputs, right-handedness, shear/reflection rejection,
  and forward/inverse recovery.
- Kept the utilities undecorated because they are deterministic validation and
  vectorized mapping primitives; workflow-level callers own useful logging.
  Their repository dependencies `as_bool()`, `as_points()`, `as_tensor()`, and
  `as_vector()` are separately confirmed.

### `nematics3d.geometry.triangulate_surface_points`

| Field | Evidence |
| --- | --- |
| Kind | Public surface-triangulation function |
| Source | [`src/nematics3d/geometry/triangulation.py`](../../src/nematics3d/geometry/triangulation.py) |
| Tests | [`tests/geometry/test_triangulation.py`](../../tests/geometry/test_triangulation.py) |
| Tutorial | [`tutorials/geometry/triangulation/triangulate_surface_points.ipynb`](../../tutorials/geometry/triangulation/triangulate_surface_points.ipynb) |
| Review scope | Centroid-based radial projection, spherical convex-hull connectivity, exact input-vertex preservation, triangular PyVista face construction, translation invariance, minimum point validation, centroid collision, degenerate projected point sets, public export, active `QSurface` caller, documentation, and known geometric limitations |
| Validation | `python -m pytest tests/geometry/test_triangulation.py -q` (5 passed); `black src/nematics3d/geometry/triangulation.py tests/geometry/test_triangulation.py`; `black --check src/nematics3d/geometry/triangulation.py tests/geometry/test_triangulation.py`; in-memory syntax compile of the implementation and focused test; complete execution of the tutorial with `jupyter nbconvert --to notebook --execute tutorials/geometry/triangulation/triangulate_surface_points.ipynb --stdout`; `git diff --check` |
| Reviewed commit | `f8638a98f3c03cbe3bd54fb32cc593d55ed5ff00` |
| Reviewed date | 2026-09-01 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Reconstruction assumes one closed surface that is approximately star-shaped with respect to the point-cloud centroid. Strong concavity, tori, disconnected surfaces, centroid-colliding samples, and other degenerate projected point sets are not repaired automatically. The latter two cases raise explicit `ValueError` exceptions. |

Summary of changes and evidence:

- Moved the implementation into the dedicated geometry triangulation module and
  retained its public geometry and top-level API exports.
- Documented the spherical-projection algorithm, exact vertex-preservation
  contract, star-shape assumption, and unsupported degenerate cases.
- Confirmed tetrahedral and sampled-sphere reconstruction, triangle counts,
  exact input vertices, translation-invariant connectivity, centroid-collision
  diagnostics, and conversion of Qhull failures to an actionable `ValueError`.
- Added and completely executed a focused tutorial covering the algorithm,
  inputs, output mesh, visualization, assumptions, and limitations.
- Kept PyVista as a deferred local import so this heavier optional visualization
  dependency is loaded only when a mesh is constructed.

### `nematics3d.geometry.find_nearest_point` and `closest_point_on_polyline`

| Field | Evidence |
| --- | --- |
| Kind | Public nearest-point geometry functions |
| Source | [`src/nematics3d/geometry/nearest.py`](../../src/nematics3d/geometry/nearest.py) |
| Tests | [`tests/geometry/test_nearest.py`](../../tests/geometry/test_nearest.py) |
| Tutorial | [`tutorials/geometry/find_nearest_point.ipynb`](../../tutorials/geometry/find_nearest_point.ipynb); no separate polyline tutorial is needed for the lower-level picking primitive |
| Review scope | Arbitrary-dimensional finite inputs, shape and dimension validation, nearest-row selection, optional index return, copy ownership, deterministic first-row tie behavior, segment projection and endpoint clamping, repeated vertices, zero-length segments, and one-point polylines |
| Validation | Combined pending-function run `python -m pytest tests/geometry/test_nearest.py tests/geometry/test_polydata.py tests/test_geometry_box.py tests/test_van_der_corput.py tests/test_grid_periodic.py tests/test_grid_plane.py tests/geometry/test_points.py -q` (105 passed); Black; canonical import; in-memory syntax compile; complete in-memory execution of the nearest-point tutorial; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Exact nearest-point ties return the first matching row and therefore remain input-order dependent. |

### `nematics3d.geometry.as_polydata_input` and `copy_polydata_geometry`

| Field | Evidence |
| --- | --- |
| Kind | Public PyVista/VTK boundary and geometry-copy functions |
| Source | [`src/nematics3d/geometry/polydata.py`](../../src/nematics3d/geometry/polydata.py) |
| Tests | [`tests/geometry/test_polydata.py`](../../tests/geometry/test_polydata.py), with downstream sampling and PlotPolyData coverage |
| Tutorial | None; these are low-level boundary helpers covered by their API documentation and focused tests |
| Review scope | Accepted dataset families, conversion fallbacks and failure chaining, existing-PolyData ownership, keyword-only diagnostics, independent geometry/topology copying, attached-array removal, and non-PolyData rejection |
| Validation | Included in the 105-test combined pending-function run; earlier focused downstream run (15 passed, 1 unrelated deselection); Black; canonical import; in-memory syntax compile; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Geometry-only copying deep-copies attached arrays before removing them, which can temporarily increase memory use for data-heavy meshes. |

### `nematics3d.geometry.get_box_corners` and `select_points_in_box`

| Field | Evidence |
| --- | --- |
| Kind | Public rectangular-box construction and point-selection functions |
| Source | [`src/nematics3d/geometry/box.py`](../../src/nematics3d/geometry/box.py) |
| Tests | [`tests/test_geometry_box.py`](../../tests/test_geometry_box.py) |
| Tutorial | [`tutorials/geometry/box/get_box_corners.ipynb`](../../tutorials/geometry/box/get_box_corners.ipynb) and [`tutorials/geometry/select_points_in_box.ipynb`](../../tutorials/geometry/select_points_in_box.ipynb) |
| Review scope | Fixed corner ordering, finite non-negative lengths, degenerate dimensions, empty points, all-point selection, translated and rotated rectangular boxes, edge validation, inclusive boundaries, tolerance, mask output, caller migration, and removal of the legacy selector name |
| Validation | Included in the 105-test combined pending-function run; Black; canonical import; in-memory syntax compile; complete in-memory execution of both tutorials; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | `select_points_in_box()` uses the first four corners and supports oriented rectangular boxes, not general skew parallelepipeds. |

### `nematics3d.analysis.sampling.sample_van_der_corput`

| Field | Evidence |
| --- | --- |
| Kind | Public low-discrepancy sequence function |
| Source | [`src/nematics3d/analysis/sampling/van_der_corput.py`](../../src/nematics3d/analysis/sampling/van_der_corput.py) |
| Tests | [`tests/test_van_der_corput.py`](../../tests/test_van_der_corput.py) |
| Tutorial | [`tutorials/analysis/sampling/sample_van_der_corput.ipynb`](../../tutorials/analysis/sampling/sample_van_der_corput.ipynb) |
| Review scope | Standard base-2 sequence, optional insertion of one, lengths zero through two, long-sequence output dtype and interval, endpoint count, non-negative integer validation, and boolean-option validation |
| Validation | Included in the 105-test combined pending-function run; Black; canonical API use; in-memory syntax compile; complete in-memory execution of the tutorial; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | None identified for the documented base-2 sequence contract. |

### `nematics3d.classes.visual.color.blue_red_in_white_bg`

| Field | Evidence |
| --- | --- |
| Kind | Public/internal white-background RGB palette function |
| Source | [`src/nematics3d/classes/visual/color.py`](../../src/nematics3d/classes/visual/color.py) |
| Tests | None; direct deterministic numerical validation is sufficient for this zero-input helper |
| Tutorial | None; the returned palette is an implementation-level visualization preset |
| Review scope | 511 RGB rows, blue-green-red endpoint and midpoint ordering, single shared green row, normalized coordinates, and unit L2 norm for every color |
| Validation | Direct numerical smoke validation of shape, endpoints, midpoint, and all row norms; Black; in-memory syntax compile; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The palette is a fixed deterministic preset and does not expose resolution or interpolation-space options. |

### `nematics3d.grid.wrap_points_to_box`

| Field | Evidence |
| --- | --- |
| Kind | Public periodic-coordinate wrapping function |
| Source | [`src/nematics3d/grid/periodic.py`](../../src/nematics3d/grid/periodic.py) |
| Tests | [`tests/test_grid_periodic.py`](../../tests/test_grid_periodic.py) |
| Tutorial | [`tutorials/grid/periodic/wrap_points_to_box.ipynb`](../../tutorials/grid/periodic/wrap_points_to_box.ipynb) |
| Review scope | Single and multiple points, mixed periodic axes, transformed and offset lattice coordinates, input isolation, empty collections, strict point validation, and active callers |
| Validation | Included in the 105-test combined pending-function run; Black; canonical import; in-memory syntax compile; complete in-memory execution of the tutorial; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Wrapping follows the configured per-axis periodicity and the grid transform contract; it does not infer periodic axes. |

### `nematics3d.grid.resolve_plane_physical_axes`

| Field | Evidence |
| --- | --- |
| Kind | Public plane-basis resolution function |
| Source | [`src/nematics3d/grid/plane.py`](../../src/nematics3d/grid/plane.py) |
| Tests | [`tests/test_grid_plane.py`](../../tests/test_grid_plane.py) |
| Tutorial | None; the helper primarily supports `PlaneGrid` and is documented directly in its API |
| Review scope | Input normalization, deterministic automatic axes, in-plane projection, collinear fallback, right-handed orthonormal basis construction, warning validation, and invalid vectors |
| Validation | Included in the 105-test combined pending-function run; Black; canonical import; in-memory syntax compile; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | A collinear requested axis falls back to the deterministic automatic-axis policy rather than preserving a user-selected orientation. |

### `nematics3d.geometry.points_membership_mask`

| Field | Evidence |
| --- | --- |
| Kind | Public exact row-membership function |
| Source | [`src/nematics3d/geometry/points.py`](../../src/nematics3d/geometry/points.py) |
| Tests | [`tests/geometry/test_points.py`](../../tests/geometry/test_points.py) |
| Tutorial | [`tutorials/geometry/points_membership_mask.ipynb`](../../tutorials/geometry/points_membership_mask.ipynb) |
| Review scope | Exact row membership, arbitrary coordinate dimension, finite values, empty inputs, duplicate candidates, common-dtype promotion, signed-zero equality, shape errors, caller migration, and removal of the legacy helper name |
| Validation | Included in the 105-test combined pending-function run; Black; canonical import; in-memory syntax compile; complete in-memory execution of the tutorial; `git diff --check` |
| Reviewed commit | `1af0fe771da3611e02b5e973fe2ec21a7270bd03` |
| Reviewed date | 2026-09-02 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | Floating-point membership is intentionally exact and does not apply a tolerance. |

## Stale review records

Move an entry here when its reviewed source or relevant behavior changes after
the recorded commit. Include the reason it became stale and the commit or change
that invalidated the earlier review.

No stale review records currently exist.

## Review history

Use this table as a compact chronological index after adding a detailed record
above.

| Date | Component | Source | Tests | Commit | Status |
| --- | --- | --- | --- | --- | --- |
| 2026-09-05 | `OptsSmoothedLine` and `SmoothedLine` | `src/nematics3d/classes/smoothed_line.py` | `tests/smooth/test_smoothed_line.py` and downstream registry check | `ac62b2c595f203aaf2509abdc536b474858d2dd2` | Confirmed |
| 2026-09-05 | `find_plane_normal()` | `src/nematics3d/geometry/plane.py` | `tests/geometry/test_plane.py` and downstream geometry/class tests | `eef44b2caf70b7d1df4ec2113196e415c36dc12f` | Confirmed |
| 2026-09-05 | `find_rotation_axis()` and `rotation_matrix_from_vectors()` | `src/nematics3d/geometry/rotation.py` | `tests/geometry/test_rotation.py` and downstream geometry/class tests | `d6275a5a6657858e7838c69ad786076ce6624d20` | Confirmed |
| 2026-08-24 | `Vect(d)` and `as_vector()` | `src/nematics3d/datatypes/vector.py` | Direct contract checks and downstream tests | `bdb7e25` | Confirmed |
| 2026-08-24 | `Tensor(shape)` and `as_tensor()` | `src/nematics3d/datatypes/tensor.py` | Direct contract checks and downstream tests | `bdb7e25` | Confirmed |
| 2026-08-24 | `ResultBase` | `src/nematics3d/classes/result_base.py` | `tests/core/test_q_diagonalization.py` | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` | Confirmed |
| 2026-08-26 | `q_diagonalize()` | `src/nematics3d/analysis/q_diagonalization/` | `tests/core/test_q_diagonalization.py`, `tests/core/test_datatypes_qfield.py` | `5089673` | Confirmed |
| 2026-08-25 | `as_director_field()` and `as_scalar_field()` | `src/nematics3d/datatypes/director_field.py`; `src/nematics3d/datatypes/scalar_field.py` | `tests/test_datatypes_director_field.py` and downstream tests | `bdb7e25` | Confirmed |
| 2026-08-26 | `as_bool()` | `src/nematics3d/datatypes/bool.py` | `tests/test_datatypes_bool.py` and downstream tests | `5089673` | Confirmed |
| 2026-08-26 | `defect_detect()` | `src/nematics3d/analysis/disclination/detection.py` | `tests/test_disclination_defect_detect.py`, `tests/test_datatypes_director_field.py` | `5089673` | Confirmed |
| 2026-08-26 | `DimensionInfo` and `as_dimension_info()` | `src/nematics3d/datatypes/dimension_info.py` | `tests/test_datatypes_dimension_info.py` and downstream tests | `5089673` | Confirmed |
| 2026-08-26 | `BoxSizePeriodic` and `as_box_size_periodic()` | `src/nematics3d/datatypes/box_size_periodic.py` | `tests/test_datatypes_box_size_periodic.py` and downstream tests | `746bc12` | Confirmed |
| 2026-08-26 | `defect_classify_into_lines()` | `src/nematics3d/analysis/disclination/classification.py` | `tests/test_disclination_line_classification.py` | `746bc12` | Confirmed |
| 2026-08-26 | `Number` and `as_number()` | `src/nematics3d/datatypes/number.py` | `tests/test_datatypes_number.py` and downstream tests | `c7443c2343dc31c700db9257f3f4125a517e4533` | Confirmed |
| 2026-08-26 | `as_qfield5()` and `as_qfield9()` | `src/nematics3d/datatypes/q_field.py` | `tests/core/test_datatypes_qfield.py` | `35af036` | Confirmed |
| 2026-08-26 | `DefectIndex` and `as_defect_index()` | `src/nematics3d/datatypes/defect_index.py` | `tests/test_datatypes_defect_index.py` | `35af036` | Confirmed |
| 2026-08-26 | `ColorRGB`, `as_ColorRGB()`, and `as_ColorRGB_array()` | `src/nematics3d/datatypes/color_rgb.py` | `tests/test_datatypes_color_rgb.py` | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` | Confirmed |
| 2026-08-26 | `as_str()` | `src/nematics3d/datatypes/string.py` | Direct contract review and downstream tests | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` | Confirmed |
| 2026-08-26 | `as_axes()` | `src/nematics3d/datatypes/axes.py` | `tests/test_datatypes_axes.py` | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` | Confirmed |
| 2026-08-26 | `as_list()` | `src/nematics3d/datatypes/list.py` | `tests/test_datatypes_list.py` | `60c63dd8c1bc497e6b9ec3de076aa6e2076b3dae` | Confirmed |
| 2026-08-26 | `as_points()` | `src/nematics3d/datatypes/points.py` | `tests/test_datatypes_points.py` and downstream tests | `c7443c2343dc31c700db9257f3f4125a517e4533` | Confirmed |
| 2026-08-26 | `as_value_range()` | `src/nematics3d/datatypes/number.py` | `tests/test_datatypes_number.py` | `c7443c2343dc31c700db9257f3f4125a517e4533` | Confirmed |
| 2026-08-26 | `shift_to_box()` | `src/nematics3d/grid/periodic.py` | `tests/test_grid_periodic.py` and datatype validation tests | `c7443c2` | Confirmed |
| 2026-08-26 | `unwrap_trajectory()` | `src/nematics3d/grid/periodic.py` | `tests/test_grid_periodic.py` and datatype validation tests | `c7443c2` | Confirmed |
| 2026-08-27 | `generate_coordinate_grid()` and `generate_fixed_step_grid()` | `src/nematics3d/grid/coordinate.py` | `tests/test_grid_coordinate.py` and direct caller tests | `2b54edaa83e084f999fb415dddff6914dd256dfb` | Confirmed |
| 2026-08-27 | `as_grid_shape()` | `src/nematics3d/datatypes/grid_shape.py` | `tests/test_datatypes_grid_shape.py` and downstream tests | `fa2239e7681b262cea48d01fa580452f2e5d851e` | Confirmed |
| 2026-08-28 | `get_q()` | `src/nematics3d/field.py` | `tests/core/test_get_q.py` and downstream tests | `70871e394effca0755983efe6888a377180871ea` | Confirmed |
| 2026-08-28 | Grid transform utilities | `src/nematics3d/grid/transform.py` | `tests/test_grid_transform.py` and downstream geometry tests | `7b3ec3a91ec30e7d7003c4570c000650ce55be12` | Confirmed |
| 2026-08-28 | `defect_validity_from_mask()` | `src/nematics3d/analysis/disclination/misc.py` | `tests/test_disclination_defect_validity.py` and Q-field initialization integration | `7e03526ffe198d1970543661f698770942dc6e9e` | Confirmed |
| 2026-08-28 | `MaskField` and `as_lattice_mask()` | `src/nematics3d/datatypes/lattice_field.py` | `tests/test_datatypes_lattice_field.py` and downstream mask integration | `d28dfc2529c64fbd26bf56ba86adc1b84cb00400` | Confirmed |
| 2026-08-28 | `GeneralField` and `as_real_lattice_field()` | `src/nematics3d/datatypes/lattice_field.py` | `tests/test_datatypes_lattice_field.py` and downstream physical-field callers | `63cb2c3b47ae847480a89f921b93da7eb1b41879` | Confirmed |
