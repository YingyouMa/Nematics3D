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

### `nematics3d.datatypes.Number` and `as_number`

| Field | Evidence |
| --- | --- |
| Kind | Public real-number semantic alias and scalar-input validator |
| Source | [`src/nematics3d/datatypes/number.py`](../../src/nematics3d/datatypes/number.py) |
| Tests | [`tests/test_datatypes_number.py`](../../tests/test_datatypes_number.py) and downstream datatype and option-validation tests |
| Tutorial | None; this is a compact input helper used by public and internal APIs |
| Review scope | Python and NumPy real scalars, explicit boolean rejection, finite values by default, opt-in NaN and infinity, integer-valued mode, Python scalar return types, inclusive ranges, optional clipping, validated replacement recovery, option validation, logging, PEP 8 naming, public exports, and active callers |
| Validation | `python -m pytest tests/test_datatypes_number.py -q` (25 passed); combined number, Q-field, defect-index, and line-classification run (63 passed, 23 subtests passed); 65 downstream option and visual tests passed with 2 subtests; Black and `black --check` on all 27 modified Python files; in-memory compile of 136 Python files; active-source stale-name search; `git diff --check` |
| Reviewed commit | `35af036` |
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
| Tutorial | [`tutorials/analysis/q_diagonalize.ipynb`](../../tutorials/analysis/q_diagonalize.ipynb) |
| Review scope | Public Q5/Q9 contract, named result, principal-only and complete eigensystems, C and `NumExpr` backend selection, Python worker threading, isotropic classification, near-degenerate orthonormality, optional right-handed frames, validation and errors, logging, performance documentation, packaging, exports, and direct callers |
| Validation | `python -m pytest tests/core -q` (25 passed, 31 subtests passed); focused `tests/core/test_q_diagonalization.py` run (10 passed, 8 subtests passed); `black --check setup.py src/nematics3d/analysis/q_diagonalization src/nematics3d/classes/result_base.py tests/core`; `ruff check --select E,F,W,N,I setup.py src/nematics3d/analysis/q_diagonalization src/nematics3d/classes/result_base.py tests/core`; in-memory syntax compile; notebook JSON, code-cell, local-link, and stale-term validation; `git diff --check`; isolated `python -m build`; wheel installation and public-API smoke test outside the repository |
| Reviewed commit | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` |
| Reviewed date | 2026-08-24 |
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
| Tutorial | [`tutorials/analysis/defect_detect.ipynb`](../../tutorials/analysis/defect_detect.ipynb) |
| Review scope | Three plaquette-normal directions, nematic sign-aligned closure criterion, non-periodic and periodic boundaries on all spatial axes, coordinate conventions, selected planes, empty output, `NumExpr` worker control, trusted-input bypass, director-field validation integration, public callers, logging decision, performance, documentation, and legacy equivalence |
| Validation | `python -m pytest tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py tests/classes/test_q_plane.py -q` (30 passed); focused defect file (19 passed); `black --check src/nematics3d/datatypes.py src/nematics3d/analysis/disclination/detection.py src/nematics3d/classes/q_field_object.py src/nematics3d/classes/q_plane.py tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py`; `ruff check tests/test_datatypes_director_field.py tests/test_disclination_defect_detect.py`; Ruff E/W/import validation for the reviewed detection implementation; in-memory syntax compile; executed notebook schema and code-cell validation; `git diff --check`; coordinate-set comparison with the archived implementation on `example/data/Q_example_workflow.npy` |
| Reviewed commit | `bdb7e25` |
| Reviewed date | 2026-08-25 |
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
| Tutorial | [`tutorials/analysis/q_diagonalize.ipynb`](../../tutorials/analysis/q_diagonalize.ipynb) |
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
| Tutorial | None yet |
| Review scope | Half-grid canonicalization, periodic coordinate wrapping, duplicate rejection, vectorized neighbor-edge construction, adjacency and Euler-trail extraction, open and closed lines, periodic-boundary lines, grid transforms and offsets, deterministic line construction, public callers, legacy equivalence, and performance |
| Validation | `python -m pytest tests/test_disclination_line_classification.py -q` (11 passed); `python -m pytest tests/test_datatypes_defect_index.py -q` (12 passed); bundled-example comparison against the archived classifier (1270 defects and 8 equivalent lines); Black; in-memory syntax and import checks; `git diff --check` |
| Reviewed commit | `bdb7e25` |
| Reviewed date | 2026-08-25 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The public `box_size_periodic` contract still relies on the overly general `as_dimension_info()` validator and is scheduled for separate normalization. No dedicated classification tutorial exists yet. |

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
| 2026-08-24 | `Vect(d)` and `as_vector()` | `src/nematics3d/datatypes/vector.py` | Direct contract checks and downstream tests | `bdb7e25` | Confirmed |
| 2026-08-24 | `Tensor(shape)` and `as_tensor()` | `src/nematics3d/datatypes/tensor.py` | Direct contract checks and downstream tests | `bdb7e25` | Confirmed |
| 2026-08-24 | `ResultBase` | `src/nematics3d/classes/result_base.py` | `tests/core/test_q_diagonalization.py` | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` | Confirmed |
| 2026-08-24 | `q_diagonalize()` | `src/nematics3d/analysis/q_diagonalization/` | `tests/core/test_q_diagonalization.py`, `tests/core/test_datatypes_qfield.py` | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` | Confirmed |
| 2026-08-25 | `as_director_field()` and `as_scalar_field()` | `src/nematics3d/datatypes/director_field.py`; `src/nematics3d/datatypes/scalar_field.py` | `tests/test_datatypes_director_field.py` and downstream tests | `bdb7e25` | Confirmed |
| 2026-08-25 | `defect_detect()` | `src/nematics3d/analysis/disclination/detection.py` | `tests/test_disclination_defect_detect.py`, `tests/test_datatypes_director_field.py` | `bdb7e25` | Confirmed |
| 2026-08-25 | `defect_classify_into_lines()` | `src/nematics3d/analysis/disclination/classification.py` | `tests/test_disclination_line_classification.py` | `bdb7e25` | Confirmed |
| 2026-08-26 | `Number` and `as_number()` | `src/nematics3d/datatypes/number.py` | `tests/test_datatypes_number.py` and downstream tests | `35af036` | Confirmed |
| 2026-08-26 | `as_qfield5()` and `as_qfield9()` | `src/nematics3d/datatypes/q_field.py` | `tests/core/test_datatypes_qfield.py` | `35af036` | Confirmed |
| 2026-08-26 | `DefectIndex` and `as_defect_index()` | `src/nematics3d/datatypes/defect_index.py` | `tests/test_datatypes_defect_index.py` | `35af036` | Confirmed |
