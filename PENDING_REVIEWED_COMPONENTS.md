# Pending Reviewed Components

This file is a lightweight staging list for functions or classes that have already been cleaned up or reviewed during the current beta-preparation work, but have **not yet been added to** `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

It is intentionally less strict than the formal reviewed-components ledger. An item should stay here until the stronger archive requirements (tests or an explicit no-test decision, validation evidence, final source review, and exact reviewed commit) are satisfied and recorded.

## Pending archive

### Processed Plot interaction consoles

- Canonical Qt panels for the already-migrated ordinary Plot classes now live under `src/nematics3d/visual/qt/`: `InteractSphere`, `InteractRod`, `InteractTube`, `InteractVector`, `InteractDelaunay`, and `InteractPolyData`. `PlotExtent` continues to reuse the Tube interaction path, so it needs no separate panel. Their former `classes.visual.qt` modules are compatibility shims.
- `InteractGlyphBase` now generates a consistent default panel title from the concrete host class (`<Kind> Controls of '<name>'`). This fixes the previously generic Sphere/Tube titles while automatically giving PlotExtent an Extent-specific title when it reuses InteractTube.
- Resolver scaling cleanup: scalar/array/callable scaling and host-side rescale-baseline reset are centralized in `InteractGlyphBase`. `InteractRod` and `InteractVector` no longer duplicate the three resolver-form branches or their reset bookkeeping. Radius rescaling in the base panel uses the same helper.
- PolyData edge-control host→GUI synchronization remains part of the canonical `InteractPolyData` implementation from its preceding cleanup. Delaunay/PolyData continue to disable radius/sides/extra geometry controls because those options are not meaningful for their mesh surfaces.
- Canonical package exports: `nematics3d.visual.qt` lazily exports all of these processed Plot panels, including `InteractTube`.
- Validation: `tests/visual/test_interact_plot_migration.py` locks canonical/legacy identity for all six panels and verifies resolver scaling preserves scalar, array, and callable forms. Focused Plot-panel regression passes 28 tests. Black and Ruff pass for the touched canonical interaction modules and new contract test. Full `tests/visual` passes 170 tests with the single pre-existing `test_plot_vector_debug_figure` return-value warning; Windows VTK `wglMakeCurrent` shutdown stderr remains environmental noise only.
- Remaining review before archive refresh: commit this interaction-console cleanup/migration batch and record its exact commit in the formal reviewed-components ledger.

### `PlotPolyData` / `OptsPolyData` / `InteractPolyData`

- Canonical sources: `src/nematics3d/visual/plot_polydata.py` and `src/nematics3d/visual/qt/interact_polydata.py`; former `classes.visual` paths are compatibility shims. The legacy `as_polydata_input` re-export is preserved for compatibility.
- Ownership contract: `PlotPolyData` remains a direct-mesh adapter. It deep-copies geometry/topology from the normalized input and intentionally strips input point/cell/field arrays so Nematics3D exclusively owns the managed display arrays (`rgba`, `opacity`, `scalars`). The caller's original mesh is not mutated and later caller-side geometry edits do not affect `raw_poly`.
- Cleanup: removed the dead `_helper_bound_coords()` override because `PlotPolyData` is fixed to `clip_mode="mesh"`; corrected the `raw_poly` documentation so it is described as a geometry/topology template rather than a point-data template; simplified the materialization path while preserving behavior.
- Interaction cleanup: `InteractPolyData` migrated with the class, and its edge controls now synchronize host-side updates (`is_show_edges`, `edge_color`, `edge_width`) back into an open panel.
- Dependency cleanup: root package and `nematics3d.visual` exports use the canonical module; production code contains no direct `nematics3d.classes.visual.plot_polydata` imports.
- Validation: focused `tests/visual/test_plot_polydata.py` passes 11 tests, including canonical/legacy identity and independent geometry ownership. Full `tests/visual` passes 168 tests with the single pre-existing `test_plot_vector_debug_figure` return-value warning. Black passes. Ruff passes for production migration modules; the existing PolyData test harness has pre-existing E402 warnings because it mutates `sys.path`/`sys.modules` before imports. Windows VTK `wglMakeCurrent` shutdown stderr remains environmental noise only.
- Remaining review before archive refresh: commit this cleanup/migration batch, then record the exact migration commit and refreshed evidence in the formal reviewed-components ledger.

### `PlotExtent` / `PlotVector` / `OptsVector` / `PlotDelaunay` / `OptsDelaunay`

- Canonical sources: `src/nematics3d/visual/plot_extent.py`, `src/nematics3d/visual/plot_vector.py`, and `src/nematics3d/visual/plot_delaunay.py`. Their former `classes.visual` modules are compatibility imports only. `InteractVector` and `InteractDelaunay` were migrated to `src/nematics3d/visual/qt/` with legacy Qt shims preserved.
- PlotExtent: kept as a thin PlotTube specialization. Its eight-corner input is validated as before and converted to 12 disconnected two-point edge segments. No extent rendering semantics were changed; the stale TODO comments around raw corners/plane-grid were removed and focused coverage now locks the 24-point / 12-edge topology contract.
- PlotVector: corrected the stale class documentation that still described mesh generation as a placeholder even though shaft/tip geometry is fully implemented. Existing shaft/tip, anchor, resolver, coloring, clipping, and pick behavior was preserved. Vector center clipping now delegates containment to the public `Bounds.act_contains_points()` contract rather than duplicating oriented-box coordinate math.
- PlotDelaunay: preserved the existing 2D Delaunay reconstruction, surface display options, scalar/color mapping, and feature-edge silhouette behavior. Center clipping now also uses `Bounds.act_contains_points()` instead of a duplicated containment implementation.
- Dependency cleanup: package-level exports and production consumers now import these components from `nematics3d.visual`. `VectorPlane`, `QPlane`, and `QFieldObject` were switched to the canonical paths. Production code contains no direct `nematics3d.classes.visual.plot_extent`, `plot_vector`, or `plot_delaunay` imports.
- Compatibility: legacy modules forward to the exact canonical class objects. `tests/visual/test_plot_extent_vector_delaunay_migration.py` verifies identity for PlotExtent, PlotVector/OptsVector, and PlotDelaunay/OptsDelaunay.
- Validation: focused migration + existing Vector + Tube/Glyph regression passed 42 tests. The full `tests/visual` suite passed 167 tests with one pre-existing `test_plot_vector_debug_figure` return-value warning. Black and Ruff pass for all files touched by this migration batch. Windows VTK `wglMakeCurrent` shutdown stderr remains environmental noise and does not affect the passing exit status.
- Remaining review before archive refresh: commit this migration batch and then record the canonical paths, exact migration commit, and refreshed validation evidence in the formal reviewed-components ledger.

### `PlotTube` / `OptsTube` / `InteractTube` / `LightingConsole`

- Canonical sources: `src/nematics3d/visual/plot_tube.py`, `src/nematics3d/visual/qt/interact_tube.py`, and `src/nematics3d/visual/qt/lighting_console.py`; the former `classes.visual` paths are compatibility imports only.
- Status: PlotTube was cleaned up and migrated into the root `nematics3d.visual` namespace after the shared PlotGlyph/Figure/Qt infrastructure was migrated. The tiny InteractTube panel moved with it. LightingConsole was also migrated because a remaining legacy import from canonical `InteractGlyphBase` created an import-order-dependent cycle through `classes.visual`.
- Topology cleanup: consecutive `line_index` range construction is centralized in `_helper_iter_index_ranges`, and clipped raw-index runs are centralized in `_helper_split_contiguous_indices`, removing duplicated segmentation logic from clipping and PolyData construction.
- Bounds cleanup: center clipping now uses the public `Bounds.act_contains_points()` contract instead of duplicating the oriented-box coordinate transform and containment test inside PlotTube.
- Picking cleanup: the established Tube pick `u_percent = idx / N * 100` convention is intentionally preserved because downstream smooth-line/tangent behavior may depend on that historical parameterization. Multiple disconnected `line_index` paths were already rendered as separate PolyData line cells; the fix here is narrower: pick nearest-point resolution now evaluates each real path independently instead of treating the raw coordinate array as one continuous auxiliary polyline across path breaks.
- Compatibility/import cleanup: package exports, `quick.py`, `QFieldObject`, disclination-line consumers, Bounds visualization, PlotExtent, and Qt Bounds interaction now use `nematics3d.visual.plot_tube`. Production code contains no direct `nematics3d.classes.visual.plot_tube` imports. Legacy Tube and Qt paths forward to the canonical class objects.
- Validation: `tests/visual/test_plot_tube_contract.py` adds migration identity, disconnected-path pick, explicit non-bridging render-topology, preserved legacy `u_percent`, and public-Bounds-containment coverage. Focused Tube + PlotGlyph regression passes in fresh runs. The latest complete `tests/visual` run passes with one pre-existing `test_plot_vector_debug_figure` return-value warning. Black passes; Ruff passes for all Tube/migration-related files. `classes/disclination_line.py` still has a pre-existing E402 module-docstring ordering issue when linted as a whole, unrelated to this migration. Windows VTK `wglMakeCurrent` shutdown messages remain environmental stderr noise and do not affect the passing exit status.
- Remaining review before archive refresh: commit this migration/cleanup batch and then record the canonical paths, exact migration commit, and refreshed validation evidence in the formal reviewed-components ledger.

### `PlotSphere` / `OptsSphere` / `PlotRod` / `OptsRod`

- Canonical sources: `src/nematics3d/visual/plot_sphere.py` and `src/nematics3d/visual/plot_rod.py`; the former `classes.visual` modules are compatibility imports only.
- Status: the sphere and rod concrete glyph implementations have been migrated into the root `nematics3d.visual` namespace after their shared `PlotGlyph`, `PlotFigure`, scalar-bar, and Qt interaction dependencies were migrated.
- Preserved behavior: GPU `vtkGlyph3DMapper` instancing, resolver behavior, center-based bounds clipping, compatibility mesh materialization, scalar-bar integration, picking/highlighting, and Sphere/Rod interaction callbacks were intentionally left unchanged. In particular, no center-clipping refactor was performed during this migration.
- Dependency cleanup: package-level exports, `quick.py`, `QFieldObject`, QPlane/QSurface consumers, canonical Bounds interaction code, and remaining legacy Qt consumers now import Sphere/Rod from `nematics3d.visual`. Production code contains no direct `nematics3d.classes.visual.plot_sphere` or `plot_rod` imports.
- Compatibility: `classes.visual.plot_sphere` and `classes.visual.plot_rod` forward to the canonical class objects. Focused migration tests verify `OptsSphere`, `PlotSphere`, `OptsRod`, and `PlotRod` preserve object identity across legacy and canonical imports.
- Validation: focused Sphere/Rod migration plus PlotGlyph contract coverage passed 30 tests. The full `tests/visual` suite passed 143 tests with one pre-existing `test_plot_vector_debug_figure` return-value warning. Black and Ruff pass for the migrated modules, shims, updated production imports, and migration tests. Windows VTK `wglMakeCurrent` shutdown messages remain environmental stderr noise and do not affect the passing exit status.
- Remaining review before archive refresh: commit the migration and then record the canonical paths, exact migration commit, and refreshed validation evidence in the formal reviewed-components ledger.

### `PlotGlyph` / `OptsGlyph`

- Canonical source: `src/nematics3d/visual/glyph.py`; `src/nematics3d/classes/visual/glyph.py` is now a compatibility import only.
- Status: previously cleaned PlotGlyph base has been migrated into the root `nematics3d.visual` namespace; no glyph rendering, resolver, clipping, scalar-bar, silhouette, or pick behavior was intentionally changed during the migration.
- Dependency cleanup: PlotGlyph now imports Bounds explicitly from `nematics3d.classes.bounds` while its visual dependencies (`PlotFigure`, `ScalarBar`) come from canonical `nematics3d.visual` modules. PlotSphere, PlotRod, PlotTube, PlotVector, PlotPolyData, PlotDelaunay, and PlotContourSurface now import `OptsGlyph` / `PlotGlyph` from `nematics3d.visual.glyph`.
- Compatibility: the legacy `classes.visual.glyph` module forwards `PlotGlyph`, `OptsGlyph`, public mode aliases, and the existing private `_as_resolver_source_or_none` helper used by downstream visual subclasses. A focused identity test verifies the legacy and canonical class objects are identical.
- Preserved scope: the existing PlotGlyph center-clipping implementation was intentionally left untouched during this migration.
- Validation: focused resolver/empty-glyph/scalar-bar/PlotGlyph contract coverage passed 34 tests. The full `tests/visual` suite passed 141 tests with one pre-existing `test_plot_vector_debug_figure` return-value warning. Black and Ruff pass for the migrated glyph module, compatibility shim, updated imports, and focused migration test. Windows VTK `wglMakeCurrent` shutdown messages remain environmental stderr noise and do not affect the passing exit status.
- Remaining review before archive refresh: commit the migration and then update the formal reviewed-components ledger with the canonical source path, exact migration commit, and refreshed validation evidence.

### `PlotFigure` / `OptsFigure` / `PickManager` / `OptsPickManager` / `ScopedConsoleDock`

- Canonical sources: `src/nematics3d/visual/plot_figure.py`, `src/nematics3d/visual/pick_manager.py`, and `src/nematics3d/visual/qt/console.py`; the former `classes.visual` paths are compatibility imports only.
- Status: the previously reviewed PlotFigure and PickManager implementations have been migrated into the root `nematics3d.visual` namespace; `ScopedConsoleDock` was migrated with them because it is a small figure-owned Qt output dock required by both components. The prior formal review entries remain historical evidence but need their source paths/reviewed commit refreshed after this migration is committed.
- Dependency cleanup: `PlotFigure` now imports `PickManager`, `ScopedConsoleDock`, and `ScalarBarRegistry` only from canonical `nematics3d.visual` paths. Production users of `FigureData`/`PlotFigure` in glyphs, plot helpers, `quick.py`, `QFieldObject`, and bounds visualization were switched to the canonical figure module. The package-level `nematics3d` and `nematics3d.visual` exports now resolve PlotFigure through the new path.
- Compatibility: legacy `classes.visual.plot_figure`, `classes.visual.pick_manager`, and `classes.visual.qt.console` modules forward to the migrated class objects. Existing PickManager tests that import private `_ClickTracker` / `_Marker` continue to work through the legacy shim.
- Validation: focused PickManager/PlotFigure lifecycle regression passed 18 tests after migration. The full `tests/visual` suite passed 140 tests with one pre-existing `test_plot_vector_debug_figure` return-value warning; that warning now reports `nematics3d.visual.plot_figure.PlotFigure`, confirming the canonical class path is active. Black passed for the migrated modules and touched imports; Ruff passed for all migration modules and touched production import sites. The root `nematics3d/__init__.py` retains pre-existing wildcard-export Ruff noise unrelated to this migration. Windows VTK `wglMakeCurrent` shutdown messages remain environmental stderr noise and do not affect the passing exit status.
- Previous reviewed commits: PlotFigure `8c16b377d75d479a8fc6ecbda2f045097f3fb62c`; PickManager `af7aa29ca372f35f738fbd99f9890a3ef29722a6`.
- Remaining review before archive refresh: commit the migration, then update `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md` with the new canonical paths, migration commit, and refreshed validation evidence.

### `ScalarBar` / `OptsScalarBar` / `ScalarBarRegistry`

- Canonical sources: `src/nematics3d/visual/scalar_bar.py` and `src/nematics3d/visual/scalar_bar_registry.py`; the former `classes.visual` modules are compatibility imports only.
- Status: cleaned up, reviewed, and migrated into the root `nematics3d.visual` namespace; not yet archived in the formal beta reviewed-components ledger.
- Validation cleanup: fixed the broken non-`None` scalar-bar `position` validator and restored the missing `Any` import required by the module's type annotations.
- Backend lifecycle: scalar-bar create/update/rebuild/unregister/registry-clear paths are covered by focused tests; visibility updates preserve the existing backend when possible, while `is_interactive` changes rebuild the backend intentionally.
- Instanced-glyph adapter fix: raw `vtkGlyph3DMapper` sources now use a persistent per-`ScalarBar` PyVista mapper adapter instead of constructing a fresh adapter on every synchronization. This removes nondeterministic scalar-bar `cmap`/`clim` updates caused by the backend remaining bound to an older adapter instance.
- Interactive widget lifecycle: widget geometry is written back to `position`/`width`/`height`; the registry explicitly detaches its `EndInteractionEvent` observer before backend removal or widget replacement so stale widgets cannot continue mutating the declaration after unregister/rebuild.
- Source-display synchronization: focused coverage verifies scalar-bar title, LUT/clim updates, zero-label tick hiding, glyph-side scalar-bar disabling, visibility, interactive rebuilds, unregister, and registry clear cleanup.
- Migration: `PlotGlyph` now imports `ScalarBar` from `nematics3d.visual.scalar_bar`, and `PlotFigure` imports `ScalarBarRegistry` from `nematics3d.visual.scalar_bar_registry`; root `nematics3d.visual` exports the three public classes. Legacy imports resolve to the exact same class objects through compatibility shims.
- Validation: `tests/visual/test_scalar_bar_contract.py` now includes the migration alias contract and passes 16 focused tests; together with the existing glyph scalar-bar test, 17/17 focused tests pass. The full `tests/visual` suite passes 134 tests with one pre-existing `test_plot_vector_debug_figure` warning. Black and Ruff pass for the migrated modules, shims, imports, and tests. Windows VTK `wglMakeCurrent` shutdown messages remain environmental stderr noise and do not affect the passing exit status.
- Remaining review before archive: record the exact reviewed migration commit and final archive evidence in `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

### `Bounds` / `OptsBounds`

- Source: `src/nematics3d/classes/bounds.py`.
- Status: cleaned up and reviewed; not yet archived in the formal beta reviewed-components ledger.
- Geometry cleanup: resolved side lengths are centralized in `Bounds.lengths`; parallel or near-parallel `axis2` is rejected explicitly; clipping geometry is materialized lazily and invalidated on geometry updates; `act_contains_points()` provides a user-facing wrapper over the existing oriented-box point-selection helper.
- Lifecycle cleanup: subscriber weakref/sync bookkeeping was deduplicated; Bounds visualization lifecycle was moved out of the geometry class into `src/nematics3d/visual/bounds.py`.
- Visualization migration: `Bounds.act_visualize()` is now a thin public delegate to `nematics3d.visual.bounds.visualize_bounds`; the Bounds frame visualization owns its own weakref registry pruning/unregistration instead of depending on Bounds private helpers.
- Console migration: `InteractBounds` now lives at `src/nematics3d/visual/qt/interact_bounds.py`; the legacy `classes.visual.qt.interact_bounds` path is a compatibility import. Orientation/roll mathematics was moved into pure geometry helpers in `src/nematics3d/geometry/rotation.py`, and obsolete continuous-interaction silhouette suppression was removed.
- Validation: focused Bounds/rotation/visual regression suites passed after the cleanup and migration; the latest Bounds + visual run passed 133 tests with one pre-existing `test_plot_vector_debug_figure` warning. Black and Ruff passed for the touched Bounds/visual modules.
- Review commits: Bounds geometry/lifecycle cleanup `570879d`; Bounds interaction geometry cleanup `ed7f4c3`.
- Remaining review before archive: commit the current Bounds visualization migration/registry-decoupling changes, then record the exact reviewed commit and final archive evidence in `dev/public_beta_preparation/BETA_RELEASE_REVIEWED_COMPONENTS.md`.

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

