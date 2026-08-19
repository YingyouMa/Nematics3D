# Features Discovered During Public Beta Preparation

This file records new functionality that is discovered as necessary while
reviewing Nematics3D for its first public beta. These entries are proposals and
design requirements, not confirmed completed work.

When a feature is implemented and verified, link its tests and reviewed commit
here, then update the relevant release checklist and component review record.

## Validate full Q tensors in `as_qfield9()`

| Field | Value |
| --- | --- |
| Status | Proposed |
| Discovered while reviewing | `nematics3d.q_diagonalize()` |
| Primary implementation | `src/nematics3d/datatypes.py::as_qfield9()` |
| Downstream beneficiary | Every function that consumes the validated full Q-tensor representation |
| Discovered date | 2026-08-18 |

### Motivation

`q_diagonalize()` mathematically assumes that each full input matrix is a
finite, symmetric, traceless Q tensor. Its current call to `as_qfield9()` checks
the floating dtype and trailing shape, but a full `(..., 3, 3)` input currently
passes through without checking finiteness, symmetry, or trace.

This validation belongs primarily at the `as_qfield9()` conversion boundary so
that downstream algorithms receive the same structural guarantee and do not
implement inconsistent duplicate checks.

### Required behavior

- Continue accepting the compact `(..., 5)` and full `(..., 3, 3)`
  representations.
- Require floating-point input under the current dtype contract.
- Reject NaN and positive or negative infinity.
- For a full representation, verify symmetry within a documented numerical
  tolerance.
- For a full representation, verify zero trace within a documented numerical
  tolerance.
- Preserve arbitrary leading dimensions when strict 3D-field validation is
  disabled.
- Preserve the input values and avoid silently modifying scientific data.
- Raise a clear `ValueError` when symmetry or tracelessness fails.
- Report the violated condition, invalid-tensor count, maximum residual, and
  first invalid leading index when practical.
- Keep valid full tensors zero-copy when the existing conversion path can safely
  return the original array.

The compact five-component representation already reconstructs symmetric
entries and sets

\[
Q_{zz} = -Q_{xx} - Q_{yy},
\]

so symmetry and tracelessness are guaranteed by its conversion. Compact input
still requires a finite-value check.

### Tolerance design

Do not use exact equality. The default tolerance should account for both dtype
precision and tensor magnitude, conceptually using a scale-aware form such as

\[
\text{tolerance}
= \text{atol} + \text{rtol}\,\max(1, \lVert Q \rVert_{\infty}).
\]

The actual default factors must be selected from tests using `float32`,
`float64`, transformed or interpolated tensors, and representative repository
data. Do not rely on `numpy.allclose()` defaults without documenting why those
defaults are appropriate for this scientific contract.

To reduce temporary memory for large fields, symmetry can be checked through
the three independent off-diagonal residuals instead of materializing a full
`Q - swapaxes(Q)` array. Trace can be checked from the three diagonal
components.

### Proposed API direction

A possible initial interface is:

```python
def as_qfield9(
    qtensor,
    name="QField",
    is_strict_3d_field=True,
    *,
    is_validate=True,
    validation_atol=None,
    validation_rtol=None,
):
    ...
```

The exact API remains undecided. If a validation bypass is provided, it should
be explicit and intended only for performance-sensitive paths whose input has
already been validated.

### Non-goals

- Do not silently symmetrize a full input tensor.
- Do not silently subtract one third of the trace.
- Do not treat a non-symmetric or non-traceless matrix as valid merely because
  an eigensolver can process it.
- Do not add positivity checks; a nematic Q tensor is not required to be
  positive semidefinite.
- Do not decide negative-order or eigenvalue-degeneracy behavior here; those
  belong to the diagonalization contract.

If projection onto the symmetric traceless subspace is later needed, provide an
explicit transformation function rather than hiding that data change inside
the default validator.

### Required tests

- Valid compact and full representations.
- Single tensors and arbitrary leading dimensions.
- `float32` and `float64` inputs.
- Compact conversion is finite, symmetric, and traceless.
- Full input within and outside the symmetry tolerance.
- Full input within and outside the trace tolerance.
- NaN and infinity rejection.
- Incorrect dtype and trailing shape.
- Input arrays are not modified.
- Valid full input preserves existing zero-copy behavior where intended.
- Any explicit validation bypass behaves exactly as documented.
- Error messages identify the failed property and useful residual context.

### Completion evidence

- [ ] Final API and tolerance policy documented.
- [ ] Implementation completed.
- [ ] Focused validator tests added and passing.
- [ ] `q_diagonalize()` tests confirm that invalid full tensors fail at the
      conversion boundary.
- [ ] Representative existing Q datasets remain accepted.
- [ ] Performance and temporary-memory impact measured on a large field.
- [ ] Component review and commit recorded.
