# Conda-forge and headless installation plan

## Purpose

This note records why Nematics3D is considering conda-forge as a primary or
recommended installation channel, especially for Linux HPC environments.  The
main motivation is not ordinary Python dependency management.  It is reliable
headless VTK/PyVista rendering and the native dependency stack required to make
that work.

This file is intentionally kept under `dev/onGoing/` because packaging and
dependency policy have not yet been finalized.

## The actual problem

Nematics3D uses PyVista/VTK for visualization.  On a desktop workstation this
normally works through a display-backed OpenGL implementation.  On an HPC
compute node, however, there may be no X server and no usable `DISPLAY` at all.
Nematics3D should still be able to perform off-screen rendering, for example to
generate figures from a batch job.

The relevant dependency chain is roughly

```text
Nematics3D
  -> PyVista
    -> VTK
      -> OpenGL backend
        -> EGL or OSMesa/Mesa
          -> native runtime libraries such as LLVM/libstdc++
```

The difficult part is therefore below the Python layer.

## Why PyPI alone is not enough for the headless target

VTK can support several rendering backends, including display-backed OpenGL,
EGL, and OSMesa.  However, support in VTK source code does not imply that an
arbitrary PyPI `vtk` wheel contains a self-contained OSMesa software-rendering
runtime suitable for an old HPC Linux system.

During investigation of Nematics3D packaging, ordinary PyPI VTK wheels were not
a satisfactory route to a robust no-display installation.  In particular, the
wheel/runtime combination did not provide the complete Mesa/OSMesa native
stack needed for the target HPCC environment.  Supplying and maintaining our
own special VTK/OSMesa wheels would make Nematics3D responsible for a large and
fragile native dependency problem that is outside the scope of the project.

This does **not** mean that Nematics3D should abandon PyPI.  PyPI remains a
reasonable installation route for ordinary desktop environments where the
system already provides a suitable graphics backend.

## Why conda-forge is attractive

Conda-forge can resolve native libraries together with Python packages.  This
is exactly what the headless use case requires: the environment solver can
select a VTK build compiled for OSMesa and install the compatible Mesa/native
runtime libraries alongside it.

This was tested successfully on the Brandeis HPCC.  A conda environment using
an OSMesa VTK build produced

```text
vtkOSOpenGLRenderWindow
```

and successfully rendered without a display.  The tested environment included
VTK 9.3.1 with an OSMesa build and PyVista 0.46.4.

That experiment is the main practical evidence behind the conda-forge plan: it
demonstrated that the desired HPC/headless behavior can actually be delivered
through the conda ecosystem instead of asking users to configure system Mesa,
Xvfb, or a custom VTK build themselves.

## Intended installation policy

The current direction is therefore:

- **PyPI remains supported** for normal Python/desktop installations.
- **Conda-forge becomes the recommended complete installation route**, in
  particular for Linux HPC and genuinely headless rendering.
- Do not promise that a pure-pip installation provides software-rendered
  headless VTK on every Linux/HPC system.
- Do not make Nematics3D maintain its own Mesa/OSMesa-enabled VTK wheels unless
  a compelling future reason appears.

In parallel with the conda-forge work, Nematics3D should migrate its tested GUI
stack from Qt5 to Qt6.  The motivation is not merely to adopt a newer Qt
version: the current VTK/Qt ecosystem is increasingly centered on Qt6, while
Qt5 is now a legacy branch.  Carrying a hard PyQt5 dependency into a new
conda-forge recipe would therefore bake an aging GUI stack into the packaging
policy just before the dependency matrix is being modernized.

The preferred architecture is to keep Nematics3D GUI code behind QtPy rather
than depending directly on PyQt6-specific APIs.  Qt6 should become the primary
and tested backend, with PySide6 compatibility retained where this can be done
without adding substantial maintenance burden.  PyQt5 should not remain the
long-term default dependency.

A possible future conda packaging structure is

```text
nematics3d
nematics3d-gui
nematics3d-headless
```

The exact package split is not decided.  The important requirement is that a
headless installation can explicitly select a compatible OSMesa VTK stack.

## Related dependency cleanup

The current Python packaging constraints should not simply be copied into a
conda-forge recipe without review.  At the time this note was written,
`pyproject.toml` still contained unusually strict version pins, including a
narrow Python version and exact versions of NumPy, PyVista, VTK, QtPy, and
PyQt5.

Previous investigation also found that native compatibility matters on older
HPC systems.  For example, an exact NumPy choice can determine whether an
appropriate manylinux wheel exists for an old glibc system.  Likewise, PyVista
and VTK versions must be selected as a compatible pair rather than upgraded
independently.

Consequently, dependency modernization and conda-forge packaging should be
treated as one coordinated task rather than changing individual pins ad hoc.

Qt migration is part of the same coordinated task.  Before the conda-forge
recipe is finalized, the repository should be audited for Qt5- or
PyQt5-specific APIs, migrated to Qt6-compatible usage through QtPy, and tested
against the selected modern VTK/PyVista stack.  At the same time, imports
should be reorganized so that core numerical analysis and genuinely headless
visualization do not initialize Qt unnecessarily.

## Recommended next steps

Before publishing a conda-forge package:

1. Audit the actual minimum and maximum supported versions of Python, NumPy,
   SciPy, PyVista, VTK, QtPy, and the Qt binding.
2. Audit the repository for Qt5/PyQt5-specific APIs and migrate the supported
   GUI path to Qt6 through QtPy.  Use Qt6 as the primary tested backend, and
   retain PySide6 compatibility when practical.
3. Decide which visualization/Qt dependencies are mandatory and which should
   be optional for non-GUI use.  In particular, verify that core numerical
   analysis and headless rendering do not require Qt initialization.
4. Define and test at least two environments:
   - normal desktop/GUI installation;
   - Linux headless installation using OSMesa VTK.
5. Test the desktop environment with the Qt6 GUI path and the selected
   VTK/PyVista versions.
6. Test the headless environment on an actual compute node with no `DISPLAY`,
   including creation of a PyVista plotter, rendering, and saving an image.
7. Verify that core nonvisual analysis can import and run without accidentally
   requiring Qt initialization.
8. Only after the compatibility matrix is established, relax and reorganize
   the version constraints in `pyproject.toml` and write the conda-forge recipe.
9. Document clearly which installation route is recommended for desktop,
   headless/HPC, and development use.

## Acceptance criterion for headless support

Do not call an installation "headless supported" merely because
`import vtk` or `import pyvista` succeeds.  A useful smoke test must run with no
display and verify an actual render, for example conceptually:

```text
DISPLAY unset
create off-screen Plotter
add a simple mesh
render
save screenshot
verify output exists and is non-empty
```

The renderer/backend should also be inspected so that the test confirms the
intended OSMesa/software-rendering path rather than accidentally succeeding
because a display server was available.

## Decision summary

The conda-forge effort is primarily an **HPC/headless rendering decision**, not
a claim that conda is intrinsically better than pip for Nematics3D's Python
code.  The decisive advantage is that conda-forge can distribute and resolve
the native VTK + OSMesa/Mesa stack required for reproducible no-display
rendering.  PyPI can continue to serve ordinary installations, while
conda-forge provides the route for a complete, reproducible headless
environment.

## Implementation log

### 2026-09-10 — import-boundary cleanup

- Audited direct `qtpy` and `pyvistaqt` imports across `src/nematics3d`.
  Direct QtPy imports are now confined to `visual/` and `visual/qt/`; the
  remaining direct `pyvistaqt` runtime imports are visualization-specific.
- Changed the package root and `quick` exports to lazy-load Q-field and
  visualization entry points.  A plain `import nematics3d` no longer loads
  `qtpy` or `pyvistaqt`.
- Removed the import-time `pyvistaqt.BackgroundPlotter` and `PlotFigure`
  dependency from `QFieldObject`.  Visualization classes/options are loaded
  only when a visualization action is actually invoked.  Consequently,
  `from nematics3d import QFieldObject` also leaves `qtpy` and `pyvistaqt`
  unloaded.
- Added subprocess regression tests for both import boundaries; these tests
  pass in the current development environment.
- A full `pytest tests` collection was attempted.  It currently stops on a
  duplicate test-module basename conflict between
  `tests/classes/test_class_base.py` and `tests/core/test_class_base.py`.
  The legacy `tests/classes/test_class_base.py` coverage was reviewed; useful
  assertions not already present were migrated into the canonical
  `tests/core/test_class_base.py`, and the obsolete duplicate file was
  deleted.  A subsequent full run completed successfully with 1158 passed and
  2 skipped tests.
- The remaining `tests/classes/` directory is not safe to delete wholesale:
  several files there still provide unique coverage (for example HostBase and
  migrated field/plane objects).  Its directory name is now stale rather than
  its contents necessarily obsolete.  A later test-layout cleanup should move
  those files to their current canonical subsystem directories and remove
  `tests/classes/` only after coverage has been preserved.
- Ruff on the touched set still reports the repository root's pre-existing
  wildcard re-export style (`F403`) plus explicit-re-export warnings; these
  should be handled as a separate package-API cleanup rather than mixed into
  the headless change.

### 2026-09-10 — Qt6 migration, phase 1

- Audited the Qt-facing source for common Qt5-only/legacy spellings.  No
  `exec_()` or direct `PyQt5` imports are present in `src/nematics3d`; Qt is
  accessed through QtPy.
- Converted the remaining flat Qt5-style enum usage to Qt6 scoped enums,
  including `Qt.WidgetAttribute`, `Qt.DockWidgetArea`, `Qt.WindowType`,
  `Qt.Orientation`, `Qt.AlignmentFlag`, and
  `QAbstractSpinBox.ButtonSymbols`.
- Switched the primary package dependency from `PyQt5==5.15.11` to
  `PyQt6==6.11.0`.  Version-range relaxation is intentionally deferred to the
  later dependency-compatibility phase.
- Current PyVistaQt upstream documentation recommends a Qt6 binding
  (`PyQt6` or `PySide6`) and describes PyQt5 as unsupported/end-of-life, so
  this direction matches the current upstream stack.
- Ran the visualization suite plus import-boundary tests after the source
  changes: 164 tests passed.  The package sdist/wheel build also succeeded.
- Important limitation: the current local development environment still uses
  the existing Qt5 binding, so the above proves source compatibility and
  packaging consistency but is not yet an actual PyQt6 runtime smoke test.
  A clean Qt6 environment test remains required before declaring the migration
  complete.

### 2026-09-10 — Qt6 runtime diagnosis

- After the local environment was changed to Qt6, the Nematics3D visual suite
  and import-boundary tests passed (164 passed), and the complete suite passed
  with 1159 passed and 2 skipped.
- Despite the passing Python tests, Windows reported native exception
  `0xc0000139` while QtPy attempted to import Qt.  A temporary diagnostic test
  established that the environment contained both `PyQt6 6.7.1` and
  `PySide6 6.8.3`.
- QtPy selected `PySide6` and reported Qt `6.8.3`; the PySide6 QtGui/QtWidgets
  imports and PyVistaQt import succeeded.
- A direct `from PyQt6 import QtCore` reproducibly failed with
  `ImportError: DLL load failed while importing QtCore`, accompanied by the
  same Windows `0xc0000139` exception.  This isolates the remaining problem to
  the local PyQt6/native Qt runtime rather than Nematics3D source code.
- The temporary diagnostic test was removed after diagnosis.  The local
  environment must be made internally consistent (one primary Qt binding and
  matching Qt6 runtime) before the PyQt6 runtime acceptance test is repeated.

### 2026-09-10 — dependency ranges and GUI extra

- Replaced exact runtime pins in `pyproject.toml` with compatibility ranges:
  NumPy `>=2.3,<3`, NumExpr `>=2.14,<3`, SciPy `>=1.16,<2`, PyVista
  `>=0.46,<1`, and VTK `>=9.3,<10`.
- Relaxed the build-time NumPy requirement from `==2.3.2` to `>=2.3,<3` so
  isolated builds are not forced onto one patch release.
- Moved the interactive Qt stack out of mandatory dependencies and into a
  `gui` extra: PyVistaQt `>=0.11.4,<1`, QtPy `>=2.4,<3`, and PyQt6
  `>=6.7,<7`.  The lower PyVistaQt bound intentionally includes the upstream
  Qt 6.10+ interaction fix released after 0.11.3.
- Kept Python itself at `>=3.12,<3.13` for now.  Python 3.13+ support should be
  widened only after the native qdiag extension and full test suite have been
  exercised there.
- This preserves one Nematics3D codebase while making GUI capability optional:
  ordinary/core and genuinely headless installations no longer need to pull a
  Qt binding merely because the package is installed.
- Package build validation succeeded after the metadata change: both sdist and
  wheel were produced, including compilation of the native qdiag extension in
  an isolated build environment using the relaxed NumPy build requirement.
- The complete repository test suite still passes in the existing development
  environment: 1159 passed and 2 skipped.  That old environment continues to
  emit the already-diagnosed Windows `0xc0000139` Qt DLL warning, so it is not
  used as evidence for the new clean Qt6 runtime.
- A fresh `Nematics3D-Qt6` conda-forge environment was created separately and
  reports QtPy backend `PyQt6` with Qt `6.11.1` without the previous DLL
  exception.  The next acceptance step is to install the repository into that
  environment and run the GUI/runtime smoke tests there.

### 2026-09-10 — Qt6 runtime test after environment switch

- Re-ran the visualization suite plus import-boundary tests after the local
  `Nematics3D` environment was switched toward Qt6: 164 tests passed.
- Re-ran the full repository suite: 1159 tests passed, 2 skipped, with only
  the two pre-existing warnings.
- However, both runs emitted Windows fatal-exception diagnostics with code
  `0xc0000139` while importing `qtpy.QtGui` / `qtpy.QtWidgets` through
  `pyvistaqt`.  Pytest still completed with exit code 0, so the Python-level
  behavior is largely intact, but the Qt runtime cannot yet be considered
  cleanly validated.
- Treat this as an environment/runtime issue to resolve before declaring the
  desktop Qt6 migration complete.  The remaining task is to verify that QtPy
  is actually selecting PyQt6 and that the installed Qt/PyQt6 native DLL set
  is internally consistent, without leftover Qt5 or mixed-channel runtime
  components.

## Implementation log

### 2026-09-10 — import-boundary cleanup, stage 1

- Changed the package root so `import nematics3d` no longer eagerly imports
  `QFieldObject`, `quick_visualize_q`, the Plot classes, or `visual.qt`.
- Preserved the existing root-level public names through lazy `__getattr__`
  exports.
- Made `nematics3d.quick.quick_visualize_q` lazy as well, so importing the root
  package does not indirectly load the visualization stack through `quick.q`.
- Added a subprocess regression test verifying that a clean
  `import nematics3d` does not load either `qtpy` or `pyvistaqt`.
- Relevant regression suite passed: 29 tests.

### 2026-09-10 — import-boundary cleanup, stage 2

- Removed eager `pyvistaqt.BackgroundPlotter` and `PlotFigure`/`OptsFigure`
  imports from `QFieldObject`.
- Moved visualization-only Plot option classes (`OptsTube`, `OptsRod`,
  `OptsSphere`, `OptsDelaunay`) behind lazy runtime helpers.  Type annotations
  use `TYPE_CHECKING` plus postponed annotations instead of forcing imports at
  module load time.
- `QFieldObject` now imports the Qt-backed figure implementation only when a
  visualization action actually needs it.  `BackgroundPlotter` support remains
  optional at runtime and plain `pyvista.Plotter` remains accepted.
- Added a subprocess regression test verifying that
  `from nematics3d import QFieldObject` does not load `qtpy` or `pyvistaqt`.
- Relevant regression suite passed: 30 tests.

This establishes the first practical headless boundary: importing Nematics3D
and importing its principal Q-field object no longer require the Qt GUI stack.
The next cleanup should inspect remaining non-GUI modules for indirect Qt
imports before changing the Qt binding or dependency version ranges.

## Implementation log

### 2026-09-10 — Phase 1A: root import boundary

Completed the first import-boundary cleanup toward a single codebase that can
serve both desktop/Qt and headless environments.

- `nematics3d.__init__` no longer eagerly imports `QFieldObject`, the Qt
  visualization modules, or `visual.qt`.
- Existing root-level public names such as `QFieldObject`, `PlotFigure`, and
  the plot classes are preserved through lazy module attributes rather than
  eager imports.
- `nematics3d.quick.quick_visualize_q` is now lazy as well, because the quick
  visualization workflow imports `QFieldObject` and would otherwise pull the
  visualization/Qt stack back into a plain `import nematics3d`.
- Added `tests/test_import_boundary.py`.  The regression test starts a fresh
  Python subprocess, imports `nematics3d`, and verifies that neither `qtpy` nor
  `pyvistaqt` has been loaded.
- Focused regression suite (`test_import_boundary`, `test_get_q`, and
  `test_quick`) passed: 29 tests.

This is only the first boundary layer.  `QFieldObject` itself still imports
`pyvistaqt.BackgroundPlotter` and `PlotFigure` at module import time, so
explicitly importing `QFieldObject` still requires the GUI stack.  The next
headless cleanup step should remove that coupling while preserving its current
visualization convenience methods.

The same modernization pass should also move the supported interactive GUI
stack to Qt6, preferably through QtPy, before the conda-forge dependency policy
is frozen.  This avoids publishing a new recipe around a legacy PyQt5 stack and
helps establish a cleaner separation between numerical core, headless
visualization, and interactive GUI dependencies.

### 2026-09-10 — Fresh Qt6 environment compatibility fixes

- Fresh conda-forge environment reports `QtPy=PyQt6`, Qt 6.11.1,
  PyVista 0.48.4, VTK 9.6.1, and PyVistaQt 0.13.1; imports succeed without the
  old `0xc0000139` DLL failure.
- Focused visual testing in that environment exposed three failures.  Two
  shared one root cause: `PlotFigure.act_check_is_alive()` relied on
  `vtkRenderWindow.GetGenericWindowId()`, which can remain falsey for a valid
  modern Qt6 `BackgroundPlotter`.  The lifecycle check now uses the plotter's
  own `_closed` state, which is common to PyVista Plotter and BackgroundPlotter.
- The remaining failure came from an existing scalar-bar actor retaining an
  older lookup table after mapper/LUT updates.  Scalar-bar synchronization now
  explicitly rebinds the actor to the mapper's current LUT after cmap/clim
  updates.
- The three affected test modules pass in the configured legacy development
  runner (28 passed).  The fresh Qt6 environment still needs to rerun those
  tests, followed by the full suite, before final desktop Qt6 acceptance.

### 2026-09-10 — Fresh Qt6 visual-suite acceptance

- Ran `pytest tests/visual tests/test_import_boundary.py` in the clean
  `Nematics3D-Qt6` conda-forge environment with QtPy=PyQt6, Qt 6.11.1,
  PyVista 0.48.4, VTK 9.6.1, and PyVistaQt 0.13.1.
- Result: 163 passed, 0 failed.  This validates the Qt6/PyVista/VTK visual
  stack and the import-boundary behavior in the clean runtime.
- Most remaining warnings are the known VTK 9.6.1 / NumPy 2.5 upstream
  deprecation in `vtkmodules.util.numpy_support`; they do not originate from
  Nematics3D.
- Removed two Nematics3D-owned `PyVistaFutureWarning` sources by explicitly
  selecting `algorithm="dataset_surface"` when normalizing non-PolyData
  datasets on PyVista versions that support that keyword, while falling back
  to the legacy no-keyword call on older supported PyVista releases.  Also
  avoided an unnecessary `extract_surface()` call in the PolyData test fixture.
- One separate PyVista runtime warning remains in the Q-plane migration test
  and will be evaluated independently rather than conflated with Qt6
  compatibility.

### 2026-09-10 — Headless test-boundary cleanup

- A clean `Nematics3D-Headless` environment initially reported 29 collection
  errors because several visualization modules imported Qt/PyVistaQt eagerly.
- `PlotFigure` now imports the interactive `BackgroundPlotter`, QtCore,
  PickManager, and console classes only when an interactive backend is actually
  requested. Off-screen `pyvista.Plotter` usage no longer requires the GUI
  extra merely to import the module.
- PlotSphere, PlotRod, PlotTube, PlotVector, PlotPolyData, PlotDelaunay, and
  PlotContourSurface now resolve their Qt interaction panels lazily when the
  interaction callback is invoked. `quick.q` likewise resolves visualization
  classes lazily while preserving its test monkeypatch seam.
- Added an import-boundary regression test covering
  `nematics3d.visual.plot_figure`; importing it must not load `qtpy` or
  `pyvistaqt`.
- Full regression in the configured development environment remains green:
  1160 passed. The clean headless environment then reached 996 passed,
  1 skipped, with only four runtime failures remaining.
- The four remaining failures were traced to eager GUI imports in
  `visual/vector_plane.py` and `SmoothedSurface.act_plot()`. These imports are
  now type-only or interaction-callback-local, respectively.

### 2026-09-10 — Headless test-boundary cleanup, stage 1

- A clean `Nematics3D-Headless` environment passed the import-boundary tests,
  but `pytest tests` initially stopped during collection with 29 errors because
  several test modules and visualization modules still imported the optional
  Qt/PyVistaQt stack eagerly.
- Refactored `visual.plot_figure` so PyVistaQt is optional at module-import
  time.  Off-screen figures can now use plain `pyvista.Plotter` without Qt;
  `BackgroundPlotter`, QtCore, PickManager, and the Qt console are imported only
  for interactive figures.  Requesting an interactive figure without the GUI
  extra now raises a targeted installation error instead of preventing module
  import.
- Refactored `quick.q` so its visualization classes are loaded only when the
  corresponding visualization path is executed.  This preserves importability
  of the quick-workflow helpers in a headless installation.
- Added pytest collection behavior that skips the explicitly GUI-oriented
  `tests/visual/` tree when QtPy/PyVistaQt are absent, while leaving non-visual
  core/headless tests active.  The legacy logger visualization script is also
  skipped when the GUI extra is unavailable.
- This is intentionally only stage 1: non-visual tests that exercise genuine
  off-screen PyVista behavior should continue to run headlessly rather than be
  reclassified as GUI-only.

### 2026-09-10 — Headless visualization import boundary

- Removed eager Qt interaction-panel imports from the primary Plot classes
  (`PlotPolyData`, `PlotContourSurface`, `PlotDelaunay`, `PlotRod`,
  `PlotSphere`, `PlotTube`, and `PlotVector`).  Their interaction panels are
  now imported only when a user actually opens an interactive control panel.
- Kept the existing plotting API intact while allowing off-screen plotting
  paths to reach these modules without loading Qt.
- Preserved the quick-workflow monkeypatch seam while making its visualization
  classes lazy, so unit tests and headless imports no longer depend on GUI
  modules being available at import time.
- Focused regression passed with 41 tests, and the complete configured
  development suite passed with 1160 tests after these changes.
- Added a subprocess regression asserting that importing
  `nematics3d.visual.plot_figure` itself does not load either `qtpy` or
  `pyvistaqt`.  This guards the off-screen/headless boundary directly.

### 2026-09-10 — Fresh Qt6 full-suite acceptance

- Ran the complete repository suite in the clean `Nematics3D-Qt6`
  environment. Result: 1160 passed, 0 failed.
- This completes functional acceptance of the current Qt6 desktop stack:
  PyQt6 through QtPy, PyVista 0.48.4, VTK 9.6.1, and NumPy 2.5 all work with
  the repository's full test coverage.
- The dominant remaining warning source is the upstream VTK 9.6.1
  `numpy_support.py` use of NumPy's deprecated direct shape assignment. This
  warning is external to Nematics3D and does not indicate a failing test.
- Consolidated Nematics3D's `extract_surface()` calls behind a compatibility
  helper. New PyVista versions explicitly use `algorithm="dataset_surface"`
  to avoid the announced default-change warning, while older supported
  PyVista releases transparently fall back to the legacy call signature.
- One PyVista runtime warning in the Q-plane migration test remains for a
  separate targeted cleanup.

### 2026-09-10 — Clean headless full-suite acceptance

- Re-ran the complete repository suite in the clean `Nematics3D-Headless`
  environment after the runtime import-boundary fixes. Result: 1000 passed,
  1 skipped, 0 failed.
- This environment intentionally has no PyQt6, QtPy, or PyVistaQt GUI stack,
  so the result confirms that the package's core analysis and non-GUI/off-screen
  visualization paths are usable without installing Qt.
- The only reported warning family (1700 warnings) is the already identified
  upstream VTK `numpy_support.py` NumPy 2.5 deprecation warning; there are no
  remaining Nematics3D headless failures in the test suite.
- The next headless acceptance gate is an actual no-display off-screen render
  and image save with renderer/backend inspection.  On Linux/HPC this must be
  performed with `DISPLAY` unset and should verify the expected OSMesa/EGL
  render-window backend rather than relying on test-suite success alone.

### 2026-09-10 — HPCC end-to-end OSMesa acceptance

- Updated `/home/yingyouma/nematics3d` on HPCC from exact Git commit
  `835c1142a34be115f66228f05c75af3c1d8772e9`; the previous copy was backed up
  before replacement.
- A negative-control environment using ordinary pip VTK 9.6.2 failed with
  `DISPLAY` unset: VTK attempted X11/EGL, reported that `libOSMesa` was absent,
  and segfaulted.  This confirms that an ordinary pip VTK wheel is not a
  sufficient HPC headless-rendering guarantee on this system.
- The conda-forge headless environment contains VTK 9.3.1's OSMesa build and
  `libOSMesa.so`.  A direct PyVista smoke test with `DISPLAY` unset produced a
  `vtkOSOpenGLRenderWindow` and successfully saved a 640x480 PNG without
  loading QtPy or PyVistaQt.
- The current headless environment's SciPy 1.14.1 `_arpack` binary requires
  `GLIBC_2.27`, which is newer than the HPCC system.  This is an independent
  environment binary-compatibility issue, not a Qt/headless regression.  For
  the final acceptance test, a temporary site-packages overlay supplied the
  known-working SciPy 1.16.0 from the existing Nematics3D environment while
  retaining VTK/PyVista from the OSMesa environment.
- With that temporary SciPy overlay, the current Nematics3D source imported
  from `/home/yingyouma/nematics3d` and executed an actual
  `PlotFigure(is_off_screen=True)` plus `PlotSphere` render with `DISPLAY`
  unset.  The render window was `vtkOSOpenGLRenderWindow`; QtPy and PyVistaQt
  remained unloaded before and after saving; and the resulting PNG was
  successfully written at 1900x1000 (81,890 bytes).
- This completes the functional end-to-end headless rendering acceptance for
  the current source.  The remaining deployment task is to repair/recreate the
  HPCC headless environment so its SciPy build is compatible with the cluster
  GLIBC without requiring the temporary overlay.

### 2026-09-10 — HPCC OSMesa backend acceptance

- Located the existing conda-forge headless environment at
  `/home/yingyouma/.conda/envs/Nematics3D_headless`.  It contains Python
  3.12.13, PyVista 0.46.4, and the conda-forge VTK 9.3.1 OSMesa build, with
  `libOSMesa.so` present in the environment.
- As a negative control, the separate pip-based VTK 9.6.2 environment at
  `/work/yingyouma/headless_pip_vtk96_test/env` failed with `DISPLAY` unset:
  VTK attempted X11/EGL, reported that `libOSMesa` was unavailable, and then
  segfaulted.  This reproduces the portability problem that motivated the
  conda-forge headless route.
- Ran a real no-display rendering smoke test in the conda-forge headless
  environment with `DISPLAY` unset.  The render-window class was
  `vtkOSOpenGLRenderWindow`; a 640x480 PNG was successfully rendered and saved
  to `/work/yingyouma/headless_osmesa_acceptance/osmesa_smoke.png` (22,801
  bytes), and neither `qtpy` nor `pyvistaqt` was loaded.
- This validates the HPCC rendering backend itself: conda-forge's OSMesa VTK
  can perform genuine software off-screen rendering without an X server or Qt.
- Limitation: the Nematics3D source tree currently present on HPCC is an older
  `/home/yingyouma/nematics3d` copy dated 2026-09-05, not the just-modified
  local repository.  Therefore the current acceptance evidence is the
  combination of (a) the current local code passing the clean no-Qt headless
  suite and (b) the target HPCC conda-forge OSMesa backend passing a real
  no-display render.  A final end-to-end Nematics3D-on-HPCC smoke test should
  be run after the current source is transferred there; do not treat the old
  HPCC source copy as equivalent to the current repository.
