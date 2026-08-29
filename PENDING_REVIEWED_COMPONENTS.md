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
