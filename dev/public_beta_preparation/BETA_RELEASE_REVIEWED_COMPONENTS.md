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
| Source | [`src/nematics3d/datatypes.py`](../../src/nematics3d/datatypes.py) |
| Tests | None; the validator is intentionally covered by direct contract checks and existing downstream tests rather than a dedicated test file |
| Tutorial | None; these are compact developer-facing input helpers |
| Review scope | Dimension semantics, real finite values, exact shape, arbitrary positive dimensions, zero-vector policy, optional normalization, validated replacement recovery, logging, naming, public exports, and all active repository callers |
| Validation | Direct `as_vector()` smoke checks for normalized 3-vectors, arbitrary 5-vectors, and replacement recovery; in-memory compile of all 77 source files; `python -m pytest tests/test_datatypes_check_sn.py tests/core/test_datatypes_qfield.py -q` (16 passed, 23 subtests passed); `python -m pytest tests/classes/test_q_plane.py -q` (1 passed); `black --check` on all modified Python files; active-source stale-name search; `git diff --check` |
| Reviewed commit | `ea653a2e3ba7a1d847055339bef2b429fe1143d5` |
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
| Source | [`src/nematics3d/datatypes.py`](../../src/nematics3d/datatypes.py) |
| Tests | None; the validator is intentionally covered by direct contract checks and existing downstream tests rather than a dedicated test file |
| Tutorial | None; these are compact developer-facing input helpers |
| Review scope | Shape semantics, arbitrary positive-rank shapes, real finite values, exact shape, validated replacement recovery, logging, naming, public exports, `as_axes()` integration, grid transforms, plot extents, and all active repository callers |
| Validation | Direct `as_tensor()` smoke checks for a 2x2 matrix, a rank-three tensor, and replacement recovery; in-memory compile of all 77 source files; `python -m pytest tests/classes/test_q_plane.py tests/core/test_datatypes_qfield.py -q` (16 passed, 23 subtests passed); `black --check` on all modified Python files; active-source stale-name search; `git diff --check` |
| Reviewed commit | `ea653a2e3ba7a1d847055339bef2b429fe1143d5` |
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
| Tutorial | [`tutorials/field/q_diagonalize.ipynb`](../../tutorials/field/q_diagonalize.ipynb) |
| Review scope | Public Q5/Q9 contract, named result, principal-only and complete eigensystems, C and `NumExpr` backend selection, Python worker threading, isotropic classification, near-degenerate orthonormality, optional right-handed frames, validation and errors, logging, performance documentation, packaging, exports, and direct callers |
| Validation | `python -m pytest tests/core -q` (25 passed, 31 subtests passed); focused `tests/core/test_q_diagonalization.py` run (10 passed, 8 subtests passed); `black --check setup.py src/nematics3d/analysis/q_diagonalization src/nematics3d/classes/result_base.py tests/core`; `ruff check --select E,F,W,N,I setup.py src/nematics3d/analysis/q_diagonalization src/nematics3d/classes/result_base.py tests/core`; in-memory syntax compile; notebook JSON, code-cell, local-link, and stale-term validation; `git diff --check`; isolated `python -m build`; wheel installation and public-API smoke test outside the repository |
| Reviewed commit | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` |
| Reviewed date | 2026-08-24 |
| Reviewer | Yingyou Ma and Codex |
| Remaining limitations | The public scalar order is defined as $S=3\lambda_{\max}/2$, so negative-$S$ conventions for oblate or anti-nematic systems are unsupported. Isotropic directors are deterministic placeholders. Individual eigenvectors remain physically non-unique in degenerate subspaces, and director sign is intentionally unspecified. A dedicated `getQ` round trip and additional downstream boundary tests were deferred by maintainer decision. The wider non-visual suite remains blocked during collection by pre-existing `ClassBase`/`HostBase` test incompatibilities; Windows VTK cleanup also emits unrelated OpenGL errors. The clean local wheel build and smoke test covered Windows CPython 3.12; Linux compilation is configured in CI, while a macOS wheel was not built locally. |

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
| 2026-08-24 | `Vect(d)` and `as_vector()` | `src/nematics3d/datatypes.py` | Direct contract checks and downstream tests | `ea653a2e3ba7a1d847055339bef2b429fe1143d5` | Confirmed |
| 2026-08-24 | `Tensor(shape)` and `as_tensor()` | `src/nematics3d/datatypes.py` | Direct contract checks and downstream tests | `ea653a2e3ba7a1d847055339bef2b429fe1143d5` | Confirmed |
| 2026-08-24 | `ResultBase` | `src/nematics3d/classes/result_base.py` | `tests/core/test_q_diagonalization.py` | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` | Confirmed |
| 2026-08-24 | `q_diagonalize()` | `src/nematics3d/analysis/q_diagonalization/` | `tests/core/test_q_diagonalization.py`, `tests/core/test_datatypes_qfield.py` | `faa6259b6dc48d2296a7d60aa2958613b0f26bf8` | Confirmed |
