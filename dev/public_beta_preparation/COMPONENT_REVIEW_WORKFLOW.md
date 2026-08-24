# Component Review Workflow

This document defines the standard workflow for reviewing and normalizing a
function, class, module, or Python file in Nematics3D before the first public
beta and during later maintenance.

The workflow is intended to make the phrase "reviewed and clean" precise and
repeatable. A component is not complete merely because it is formatted or its
existing tests pass. Its contract, scientific behavior, failure modes,
documentation, implementation, and validation evidence must all be addressed.

Related documents:

- [`BETA_RELEASE_CHECKLIST.md`](BETA_RELEASE_CHECKLIST.md) tracks project-level
  beta release readiness.
- [`BETA_RELEASE_REVIEWED_COMPONENTS.md`](BETA_RELEASE_REVIEWED_COMPONENTS.md)
  records components that have completed this workflow.
- [`DISCOVERED_FEATURES.md`](DISCOVERED_FEATURES.md) records new functionality
  discovered as necessary during component reviews.
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) describes repository-wide
  contribution conventions.

## Core principle

Define the behavior that the project intends to support, express that behavior
in tests, correct the implementation, and only then perform final naming,
documentation, and formatting cleanup.

Do not let an accidental behavior become a permanent API contract solely
because the current implementation happens to produce it.

## Review state model

Each review proceeds through the following states:

```text
Selected
  -> Scope defined
  -> Contract defined
  -> Characterization tests added
  -> Edge tests added
  -> Implementation reviewed
  -> Input and error behavior reviewed
  -> Documentation reviewed
  -> Naming and structure reviewed
  -> Formatting and static checks passed
  -> Logging decision reviewed
  -> Focused tests passed
  -> Full suite passed
  -> Confirmed
```

Only a `Confirmed` component may be added to
`BETA_RELEASE_REVIEWED_COMPONENTS.md`.

## 0. Select the component and define the scope

Before editing, write down the review boundary.

- [ ] Identify the qualified component name.
- [ ] Decide whether the review covers a function, class, functional section,
      module, or complete Python file.
- [ ] Prefer the smallest meaningful component.
- [ ] For a large class, divide the work by method or functional section.
- [ ] Identify whether the component is public API, internal API, or private
      implementation.
- [ ] Find direct callers, exports, related documentation, and existing tests.
- [ ] Identify downstream behavior that may be affected.
- [ ] State which adjacent components are explicitly outside this review.
- [ ] Define the evidence required to declare the component complete.

Do not claim that an entire file or class is reviewed when only one function or
method was inspected.

Suggested scope record:

```markdown
Component: `nematics3d.module.component`
Kind: function / class / module / Python file
Source: `src/nematics3d/path.py`
Public API: yes / no / undecided
Existing tests: `tests/path/test_component.py`
In scope: ...
Out of scope: ...
Completion evidence: ...
```

## 1. Define the intended contract

Read the implementation, call sites, exports, tests, and relevant scientific
documentation before changing behavior. Explicitly resolve ambiguity where
possible.

### Inputs

- [ ] Accepted Python and NumPy types.
- [ ] Required shapes and permitted leading dimensions.
- [ ] Axis ordering and coordinate conventions.
- [ ] Accepted dtypes and conversion rules.
- [ ] Physical units or dimensionless assumptions.
- [ ] Broadcasting behavior.
- [ ] Optional/default argument behavior.
- [ ] Whether inputs are copied, viewed, cached, or modified in place.
- [ ] Rules for masks, NaN, Inf, empty arrays, and missing data.

### Outputs

- [ ] Return type, shape, dtype, and ordering.
- [ ] Physical or mathematical meaning.
- [ ] Ownership and mutability of returned data.
- [ ] Determinism and ordering guarantees.
- [ ] Numerical precision and tolerance expectations.
- [ ] Equivalences such as director sign, periodic coordinates, or unordered
      collections.

### Failure behavior

- [ ] Invalid input conditions.
- [ ] Exception types.
- [ ] Error-message content.
- [ ] Warnings and logging behavior.
- [ ] Unsupported but currently accepted cases.
- [ ] Known scientific or numerical limitations.

Classify observed behavior as one of:

- intended and supported;
- intended but insufficiently documented;
- accidental and safe to change;
- legacy behavior requiring deprecation; or
- undecided and requiring an explicit project decision.

## 2. Add characterization and main-path tests

Before changing the implementation, protect behavior that has been confirmed as
correct.

- [ ] Add the simplest valid example.
- [ ] Add at least one representative scientific use case.
- [ ] Assert return type, shape, dtype, and important invariants.
- [ ] Check whether the input is modified unexpectedly.
- [ ] Compare against an independent reference implementation where possible.
- [ ] Test intended public imports rather than relying only on internal module
      imports.
- [ ] Use deterministic data and fixed random seeds.

When the current implementation contradicts the intended contract, write a test
for the intended behavior and confirm that it fails for the expected reason
before fixing the implementation.

Avoid tests that merely check that no exception was raised. Verify meaningful
results.

## 3. Add boundary, invalid-input, and numerical tests

Cover all applicable categories.

### Boundary cases

- [ ] Smallest valid input.
- [ ] Scalar, single-point, or one-element input.
- [ ] Empty input or zero-length dimensions.
- [ ] Zero values and near-zero values.
- [ ] Minimum and maximum valid parameter values.
- [ ] Periodic boundaries and domain edges.
- [ ] Cropped, offset, transformed, or non-unit-spacing grids.

### Numerical cases

- [ ] Very small and very large scales.
- [ ] Degenerate and nearly degenerate states.
- [ ] Floating-point cancellation or unstable formulas.
- [ ] NaN and Inf behavior.
- [ ] Multiple supported floating-point dtypes.
- [ ] Comparison against a slower or simpler trusted reference.
- [ ] Explicit tolerances with a documented rationale.

### Invalid inputs

- [ ] Incorrect types and object arrays.
- [ ] Incorrect shapes or dimension counts.
- [ ] Incompatible argument combinations.
- [ ] Non-finite values where they are forbidden.
- [ ] Physically invalid tensors, vectors, coordinates, or options.
- [ ] Expected exception type and useful message.

For mathematical equivalences, test the equivalence rather than a particular
representation. For example, compare nematic directors using the absolute dot
product instead of requiring identical signs.

## 4. Organize the tests

Keep normal, reference, edge, and invalid-input tests in one focused test file
when the file remains easy to navigate. Group them by test class or clearly
named test functions.

Suggested initial structure:

```text
tests/test_component.py
  - normal behavior
  - scientific/reference comparison
  - boundary and numerical stability
  - invalid inputs and failure behavior
```

Split tests only when there is a concrete reason, such as:

- the focused test file has become difficult to navigate;
- reference tests require large or specialized datasets;
- tests are slow and need a separate CI marker;
- GUI, off-screen, cluster, or platform-specific setup is required; or
- integration behavior is meaningfully distinct from unit behavior.

Possible split:

```text
tests/test_component.py
tests/test_component_edge_cases.py
tests/test_component_reference.py
```

Test data should be small, deterministic, legally distributable, and documented.
Do not use private or unnecessarily large research data when a reduced fixture
can verify the same behavior.

## 5. Review and correct the implementation

After the intended behavior is protected by tests:

- [ ] Fix correctness problems.
- [ ] Remove duplicate, dead, or unreachable logic.
- [ ] Simplify control flow where doing so improves verifiability.
- [ ] Extract helpers only when they have a coherent responsibility.
- [ ] Preserve performance-critical paths unless a measured tradeoff is
      justified.
- [ ] Retain a simple reference implementation or test oracle for optimized
      scientific algorithms when practical.
- [ ] Avoid unrelated refactors.
- [ ] Run focused tests after each small logical change.
- [ ] Confirm that changed behavior matches the written contract.

Do not weaken validation or broaden tolerances merely to make a failing test
pass. Determine whether the failure comes from the environment, test, contract,
or implementation first.

## 6. Review input validation and error behavior

Perform a separate validation pass even when the main algorithm is correct.

- [ ] Invalid inputs fail close to the public boundary.
- [ ] Exceptions use stable and appropriate types.
- [ ] Error messages identify the parameter and actual problem.
- [ ] Dangerous broadcasting and silent conversions are prevented.
- [ ] NumPy arrays are not used in ambiguous boolean contexts.
- [ ] Array-like comparisons use `nematics3d.format.is_equal()` where
      appropriate.
- [ ] Object dtype is rejected when it would create unsafe or surprising
      behavior.
- [ ] Symmetry, tracelessness, normalization, or other scientific constraints
      are checked when required by the contract.
- [ ] Validation is not needlessly repeated in performance-sensitive loops.
- [ ] Warnings distinguish recoverable unusual cases from normal operation.

## 7. Review documentation and public API exposure

Update documentation only after the behavior is settled and tested.

- [ ] Docstrings describe actual behavior rather than intended future behavior.
- [ ] Parameters include types, shapes, units, and conventions.
- [ ] Returns include types, shapes, ownership, and scientific meaning.
- [ ] Raised exceptions are documented when useful to callers.
- [ ] Notes explain numerical algorithms, equivalences, and limitations.
- [ ] Examples are minimal and executable.
- [ ] Type annotations agree with runtime behavior.
- [ ] README, tutorials, and examples use the current API.
- [ ] Top-level exports expose only intentional public objects.
- [ ] Public renames or removals follow the beta compatibility/deprecation
      policy.

Do not publish implementation details as stable API by accident.

## 8. Review names, structure, and comments

- [ ] Function, class, method, and variable names express domain meaning.
- [ ] Boolean names begin with `is_` when they represent boolean state.
- [ ] Private implementation helpers use appropriate private naming.
- [ ] Class fields follow the repository `raw_`, `state_`, `calc_`, `entity_`,
      and `impl_` conventions where applicable.
- [ ] Methods in a substantially revised class are grouped into functional
      sections.
- [ ] Intentional overrides include the repository override comment block.
- [ ] Imports remain at module scope unless a documented circular or optional
      dependency requires a local import.
- [ ] Comments explain rationale, assumptions, or non-obvious mathematics rather
      than restating code.
- [ ] Long text and comments are wrapped near Black's normal line width where
      practical.
- [ ] Debug prints and temporary diagnostics are removed.

Do not rename public objects solely for style without considering compatibility
and migration cost.

## 9. Run formatting and static checks

Formatting is a final mechanical pass, not a substitute for semantic review.

- [ ] Run Black on every modified Python file.
- [ ] Run `black --check` on the reviewed files.
- [ ] Run the repository's configured Ruff or lint checks when available.
- [ ] Run import checks when available.
- [ ] Run a syntax-only compile check.
- [ ] Run `git diff --check`.
- [ ] Run type checking for the reviewed public interface when configured.

On Windows, use the project conda environment:

```powershell
C:\Users\myy23\anaconda3\Scripts\conda.exe run -n Nematics3D `
    black .\src\nematics3d\path.py .\tests\test_path.py
```

If `py_compile` cannot write `__pycache__`, perform an in-memory compile so a
cache permission problem is not misclassified as a syntax failure:

```powershell
C:\Users\myy23\anaconda3\Scripts\conda.exe run -n Nematics3D `
    python -c "from pathlib import Path; compile(Path('src/nematics3d/path.py').read_text(encoding='utf-8'), 'src/nematics3d/path.py', 'exec')"
```

## 10. Review the logging decision and message design

Logging is reviewed after behavior, documentation, naming, and formatting have
settled. At this point the reviewer can judge which events are genuinely useful
to users or developers without using temporary diagnostics as permanent logs.

Every reviewed function or method must receive an explicit decision: add or
keep structured logging, revise its logging, or intentionally use no logger.

### Decide whether the component needs a logger

- [ ] Add or keep `@logging_and_warning_decorator(...)` only when the function
      body uses the injected `logger`, or when start/finish timing and nested
      call tracing are intentionally valuable.
- [ ] Do not add the decorator merely for consistency with nearby functions.
- [ ] Remove an existing decorator if it provides no useful body messages and
      its start/finish trace has no diagnostic or performance value.
- [ ] Consider call frequency: tiny helpers, property accessors, and functions
      inside tight loops normally should not create logging noise or overhead.
- [ ] Consider ownership: prefer logging at the public workflow boundary rather
      than repeating the same message in every internal helper.
- [ ] Keep `logger=None` only where it is required by the decorator-injected
      calling convention.
- [ ] Decide whether default `DEBUG` start/finish messages are appropriate or
      whether `start_finish_level` should be changed deliberately.

### Repository log-level standard

Nematics3D defines eight levels in `src/nematics3d/logging_decorator.py`. Use
the lowest level that accurately describes the event:

| Value | Level | Use |
| ---: | --- | --- |
| 5 | `DETAIL` | Extremely fine-grained tracing, per-iteration state, and deep internal diagnostics. |
| 10 | `DEBUG` | Important internal state and intermediate values useful to developers diagnosing a problem. |
| 15 | `PROGRESS` | High-level procedural updates describing what the current workflow is doing. |
| 20 | `INFO` | Key results, concise outcome or performance summaries, and important defaults implicitly applied because the user did not provide them. |
| 30 | `WARNING` | A potentially incorrect or risky condition; execution continues, but the user should inspect input or configuration. |
| 35 | `RECOVERY` | An error occurred and the system automatically corrected or compensated for it, so the result differs from the originally intended operation. |
| 40 | `ERROR` | The current operation failed; it may or may not be possible for a higher-level caller to continue. |
| 50 | `CRITICAL` | A severe unrecoverable condition requiring immediate termination, such as corrupted state, divergence, or unrecoverable resource failure. |

The global default threshold is `PROGRESS(15)`. Consequently, ordinary users
see `PROGRESS` and higher messages by default, while `DEBUG` and `DETAIL` remain
hidden. `RECOVERY` is always shown by the current logging implementation because
the corrected result differs from the requested operation.

Use the corresponding logger methods:

```python
logger.detail(...)
logger.debug(...)
logger.progress(...)
logger.info(...)
logger.warning(...)
logger.recovery(...)
logger.error(...)
logger.critical(...)
```

Use `logger.exception(...)` while handling an active exception when the
traceback is part of the required diagnostic record.

### Decide what each message contains

- [ ] State the event and its consequence, not merely that a code line was
      reached.
- [ ] Include identifiers, counts, shapes, thresholds, elapsed time, or selected
      fallback behavior when they help the reader act on the message.
- [ ] Avoid dumping full arrays or large objects; report compact summaries.
- [ ] Do not include secrets, private data, unnecessary local paths, or complete
      user datasets.
- [ ] Do not repeat function names unnecessarily; the decorator already adds
      contextual function or method ownership.
- [ ] Keep messages stable enough for diagnosis, but do not make tests depend on
      incidental wording unless the text is part of the supported user
      experience.
- [ ] Write `PROGRESS` messages around meaningful workflow phases, not every
      implementation step.
- [ ] Write `INFO` messages for results or implicitly selected defaults that the
      user needs to understand or reproduce the operation.
- [ ] Use `WARNING` only when user attention is warranted; normal numerical
      branches and expected fallbacks are not automatically warnings.
- [ ] Use `RECOVERY` when an actual requested operation failed and the library
      substituted a different safe behavior.
- [ ] Log an `ERROR` only for a failed operation. If the function raises an
      exception, remember that the decorator already logs the exception and
      traceback; avoid duplicate error messages.
- [ ] Reserve `CRITICAL` for truly unrecoverable program state.

### Validate logging behavior

- [ ] Test significant warning, recovery, and user-visible default messages.
- [ ] Confirm that normal calls do not emit unnecessary warning or recovery
      noise.
- [ ] Confirm that `log_mode="none"` suppresses ordinary output as intended.
- [ ] Confirm that nested decorated calls preserve useful context and do not
      duplicate high-level progress messages.
- [ ] Confirm that logging does not change return values, exceptions, numerical
      results, or mutable state.
- [ ] Record the final logging decision in the component review summary,
      including an explicit `no logger needed` decision when applicable.

## 11. Perform layered validation

Run validation from narrowest to broadest so failures are easy to classify.

1. The newly added or modified individual test.
2. The focused component test file.
3. Tests for the containing module or functional area.
4. The complete test suite.
5. A clean build of the source distribution and wheel when relevant.
6. Installation of the wheel in a clean environment.
7. A smoke test outside the repository source directory.

Typical focused command:

```powershell
C:\Users\myy23\anaconda3\Scripts\conda.exe run -n Nematics3D `
    python -m pytest .\tests\test_component.py -q
```

For optimized scientific functions, also run:

- [ ] randomized comparisons against an independent reference;
- [ ] multiple numerical scales and fixed random seeds;
- [ ] invariants appropriate to the physical quantity;
- [ ] a focused performance comparison when regression risk is material.

Record the exact commands and results for the final review entry.

## 12. Inspect the final diff manually

Passing tests do not replace a final human review.

- [ ] Only intended files and behavior changed.
- [ ] No unrelated user changes were overwritten.
- [ ] No accidental public API change was introduced.
- [ ] No local absolute paths, credentials, caches, debug output, or private
      data were added.
- [ ] Tests assert scientifically meaningful outcomes.
- [ ] Tests do not reproduce the implementation in a way that duplicates the
      same potential mistake.
- [ ] Documentation, implementation, and tests agree.
- [ ] Tolerances are neither unexplained nor excessively loose.
- [ ] Generated or formatted changes do not obscure semantic changes.
- [ ] Remaining limitations are explicit.

## 13. Commit and record the confirmed review

After all applicable checks pass:

- [ ] Commit the reviewed source, tests, and documentation together or in a
      clearly traceable series.
- [ ] Record the full commit SHA.
- [ ] Add a detailed entry to
      `BETA_RELEASE_REVIEWED_COMPONENTS.md`.
- [ ] List every source and test file used as evidence.
- [ ] Include the exact validation commands.
- [ ] Record the review date and remaining limitations.
- [ ] Add the component to the compact review-history table.

If the reviewed implementation or relevant contract changes later, move the
entry to the stale section. It may return to the confirmed section only after
the affected workflow steps are repeated.

## Completion gate

A component is `Confirmed` only when all of the following are true:

- [ ] The scope is precise.
- [ ] The intended contract is explicit.
- [ ] Main behavior has meaningful tests.
- [ ] Applicable boundary and invalid-input cases are tested.
- [ ] Scientific results are checked against invariants or an independent
      reference.
- [ ] The implementation has been reviewed, not merely formatted.
- [ ] Input validation and failure behavior are deliberate.
- [ ] Public documentation matches tested behavior.
- [ ] Names, structure, and comments follow repository conventions.
- [ ] The logger/decorator decision is explicit, message levels follow the
      repository standard, and user-visible logging behavior has been tested
      where material.
- [ ] Formatting and static checks pass.
- [ ] Focused tests pass in the `Nematics3D` conda environment.
- [ ] The relevant broader test suite passes.
- [ ] The final diff has been inspected manually.
- [ ] Validation evidence and the reviewed commit are recorded.

## Completed local review: `as_qfield9`

Review date: 2026-08-22

Scope:

- `src/nematics3d/datatypes.py::as_qfield9()` and its private validation helpers.
- Focused tests in `tests/core/test_datatypes_qfield.py`.
- Conversion and validation only; projection, eigensolver behavior, and a
  user-facing tutorial are outside this component's scope.

Recorded evidence:

- The final dtype, shape, empty-field, finite-value, symmetry, trace, tolerance,
  bypass, and zero-copy contracts are documented in the function docstring.
- Symmetry validation compares three independent off-diagonal pairs and reuses
  two leading-shape work arrays.
- Black and the 11 focused tests pass in the `Nematics3D` conda environment.
- The 16 relevant `q_diagonalize()` tests also pass; a dedicated downstream
  invalid-input boundary test was explicitly deferred by maintainer decision.
- `example/data/Q_example_workflow.npy` remains accepted. Its 2,000,000 compact
  `float32` tensors converted in 0.116 s. The resulting 68.7 MiB full field
  validated in 0.120 s with 76.3 MiB peak extra Python allocation and a
  zero-copy return.
- The broader non-visual/non-slow pytest attempt is currently blocked during
  collection by pre-existing `ClassBase` and `HostBase` test incompatibilities;
  unmarked VTK tests also emit Windows OpenGL cleanup errors.
- The final local diff was inspected and recorded in implementation commit
  `f757ffb`.

Disposition: implementation and local review complete; suitable for internal
use without a dedicated tutorial.

## First application: `q_diagonalize`

The first planned use of this workflow is
`nematics3d.analysis.q_diagonalization.q_diagonalize` in
`src/nematics3d/analysis/q_diagonalization/`.

Its focused review should include:

1. Define the Q5/Q9 input contract, supported leading dimensions, dtype rules,
   and invalid-input behavior.
2. Preserve and improve the existing isotropic and analytic-director fallback
   tests.
3. Add `getQ` to `q_diagonalize` round-trip tests.
4. Compare randomized symmetric traceless tensors with `np.linalg.eigh`.
5. Verify eigenvalue, eigenvector, normalization, and sign-equivalence
   invariants.
6. Test zero, near-zero, degenerate, near-degenerate, large-scale, and
   small-scale tensors.
7. Decide and document NaN, Inf, asymmetric, non-traceless, and negative-S
   behavior.
8. Review fallback tolerances, performance, warning behavior, and docstrings.
9. Decide whether the logging decorator remains justified, classify every
   retained message by the repository level standard, and test material
   user-visible messages.
10. Run formatting, focused tests, the relevant broader suite, and a clean
   package smoke test.
11. Record the component only after the reviewed commit exists.
