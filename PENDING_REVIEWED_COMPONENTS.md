# Pending Reviewed Components

This file is a lightweight staging list for functions or classes that have already been cleaned up or reviewed during the current beta-preparation work, but have **not yet been added to** `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

It is intentionally less strict than the formal reviewed-components ledger. An item should stay here until the stronger archive requirements (tests or an explicit no-test decision, validation evidence, final source review, and exact reviewed commit) are satisfied and recorded.

## Pending archive

### `HostBase` / `OptsBase`

- Source: `src/nematics3d/core/host_base.py`.
- Status: black-box behavior has been substantially reviewed and hardened without attempting an architectural refactor of the implementation.
- Covered contract: opts lifecycle and validation, host/opts commit routing, raw/state updates and opts reapplication, writable properties, extra attrs, protection/wrapping, wrapper forwarding, sync/enrichment callbacks, snapshots, JSON persistence, and HostBase inspection surfaces.
- Focused tests: `tests/classes/test_host_base.py` (59 passed at the current review point).
- Broader validation at the behavior-hardening commit covered representative smoothing, geometry, and visual HostBase descendants (95 passed), with Black and Ruff clean for the touched HostBase/test files.
- Review commit: black-box coverage and the minimal writable-property commit fix `4126db2`; reference tutorial added in `0c48290` at `tutorials/reference/core/HostBase.ipynb`.
- Remaining review before archive: exercise the reference tutorial itself, inspect representative real HostBase subclasses for integration-specific contracts, and record final archive validation/evidence. `PlotGlyph` is the next subclass being inspected.

### `SmoothedLineFunc`

- Source: `src/nematics3d/classes/smoothed_line.py`.
- Status: review in progress; pairwise-delta storage was reduced from `O(N^2)` to `O(N)`, and raw samples now use explicit `ResultBase` objects rather than positional scalar/tuple conventions.
- Result protocol: `raw_func(u_percent, **func_kwargs)` returns a `ResultBase`; `result_value_attr` selects the value to smooth; complete raw results are retained in `calc_results`.
- Beta integration migration: `DisclinationLineSmooth` consumes complete `DefectSectionOmegaResult` samples through `result_value_attr="beta"`.
- Focused tests: `tests/smooth/test_smoothed_line_func.py`, `tests/smooth/test_smoothed_line_func_registry.py`, and `tests/classes/test_q_field_object_phase2.py`.
- Earlier validation: focused delta and ResultBase protocol suites passed together with syntax and Black checks at their recorded review commits.
- Review commits: streamed-delta implementation/tests `74b23bb`; ResultBase sample protocol and beta-integration migration `9d75108`.
- Remaining review before archive: inspect constructor/state initialization, `act_update()`, scalar/vector output shape handling, interpolation behavior, registry interactions, and any remaining edge cases; then run the final focused suite and record the exact reviewed commit.

