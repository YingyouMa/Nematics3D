# Dependency compatibility audit

This note tracks the evidence used to relax Nematics3D's dependency metadata.
The goal is to distinguish versions that happened to be installed in the
development environment from versions that are actually required by the
source code and binary extension.

## 2026-09-10 — first static audit

### Python

- The source uses Python 3.10 syntax extensively, including PEP 604 union
  annotations such as `str | None` and built-in generic aliases such as
  `tuple[int, ...]`.
- More importantly, `src/nematics3d/visual/qt/panel_base.py` uses
  `@dataclass(slots=True, weakref_slot=True)` in two places.
  `weakref_slot` was added to `dataclasses.dataclass` in Python 3.11, so the
  current source has a genuine Python 3.11 runtime floor unless that code is
  rewritten.
- A follow-up scan found no obvious Python 3.12-only dependency such as
  `itertools.batched`, `Path.walk`, `TypeAliasType`, or `sys.monitoring`.
- The first experimental lower bound is therefore corrected to
  `requires-python = ">=3.11"`.
- This remains a source-level conclusion; Python 3.11 and newer interpreters
  still need explicit environment tests before the range is treated as fully
  validated.
- HPCC already has a Python 3.11.13 environment (`checkQ`).  Using that
  interpreter, the complete current `src/nematics3d` tree passes
  `python -m compileall`, and the runtime `dataclasses.dataclass` supports the
  required `weakref_slot` parameter.  This validates the source syntax on
  Python 3.11 independently of the Windows Python 3.12 development stack.

### NumPy

- The Python source does not currently use an obvious NumPy 2.x-only API.
- The native Q-diagonalization extension explicitly defines
  `NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION` and uses long-established NumPy
  C-API entry points (`PyArray_FROM_OTF`, `PyArray_SimpleNew`, `PyArray_SIZE`,
  etc.). Nothing found in the extension requires NumPy 2.3 specifically.
- NumPy's downstream-package guidance recommends building extensions against
  NumPy 2.x when a wheel is intended to run with both NumPy 1.x and 2.x.
- The first experimental build requirement is therefore `numpy>=2.0,<3`.
- The first experimental runtime floor is `numpy>=1.26,<3`. NumPy 1.26 is
  intentionally used as the initial 1.x compatibility target rather than
  claiming support for much older NumPy releases before testing them.

### SciPy

Nematics3D currently imports only mature SciPy APIs:

- `scipy.optimize.curve_fit`
- `scipy.spatial.cKDTree`
- `scipy.spatial.ConvexHull` and `QhullError`
- `scipy.spatial.transform.Rotation`
- `scipy.interpolate.splprep`, `splev`, `interp1d`, and
  `RegularGridInterpolator`
- `scipy.signal.savgol_filter`
- `scipy.ndimage.gaussian_filter`

The first experimental floor was `scipy>=1.10,<2`, but the minimum-stack
runtime test showed that this is too low: Nematics3D passes `axes=` to
`scipy.ndimage.gaussian_filter`, and SciPy 1.10.1 does not accept that keyword.
SciPy 1.11.0 added the `axes` argument to `gaussian_filter`, so the current
candidate floor is `scipy>=1.11,<2`.

### NumExpr

- Nematics3D uses `numexpr.evaluate`, `MAX_THREADS`, `get_num_threads`, and
  `set_num_threads`.
- `get_num_threads` predates the modern 2.10 series, so the source itself does
  not require NumExpr 2.14.
- NumExpr 2.10 added NumPy 2.0 support, making 2.10.1 a more appropriate floor
  for a package that intends to support both NumPy 1.26 and NumPy 2.x.
- The first experimental requirement is therefore `numexpr>=2.10.1,<3`.

### PyVista, VTK, and Qt

- The visualization-stack audit is now split into core rendering and GUI
  dependencies.
- Core rendering has been exercised below the original conservative bounds:
  PyVista 0.45.3, 0.44.2, and 0.43.10 all pass the complete no-display OSMesa
  suite with `1000 passed, 1 skipped, 38 subtests passed` when paired with the
  accepted Python 3.11 / NumPy 1.26 / SciPy 1.11 / NumExpr 2.10 minimum stack.
- PyVista 0.43.10 paired with VTK 9.2.6 OSMesa also passes the complete suite
  with the same result.  The conservative validated core bounds are therefore
  `pyvista>=0.43,<1` and `vtk>=9.2,<10`.
- The source already carries an `extract_surface_compat()` compatibility shim
  for PyVista API changes, which is consistent with supporting versions older
  than 0.46 rather than depending on a 0.46-only API.
- GUI imports were then checked with PyVista 0.43.10,
  `pyvistaqt==0.10.0`, `qtpy==2.2.1`, and PyQt6 6.8.1.  QtPy successfully
  exposed every scoped enum and utility used by Nematics3D, including
  `Qt.Orientation`, `AlignmentFlag`, `WidgetAttribute`, `WindowType`,
  `DockWidgetArea`, `QSignalBlocker`, and `QTimer`.  All reviewed
  `nematics3d.visual.qt` modules and `visual.plot_figure` imported successfully.
- `BackgroundPlotter` from PyVistaQt 0.10.0 imports successfully and exposes
  the expected `show` / `off_screen` constructor interface.  Attempting to
  construct it on the HPCC login node with `DISPLAY` unset aborts in
  `vtkXOpenGLRenderWindow`; this is the expected limitation of the Qt/X11 VTK
  build and is not an API-version failure.
- The GUI bounds are therefore relaxed conservatively to
  `pyvistaqt>=0.10,<1` and `qtpy>=2.2,<3`.
- `PyQt6>=6.7,<7` is retained for now.  PyQt6 6.4.2 has suitable upstream
  Python 3.11-era wheels, but those wheels require a newer manylinux/glibc
  baseline than the HPCC host and conda-forge's current main channel no longer
  offers that exact old build.  A lower PyQt6 floor should therefore be tested
  on a suitable desktop/CI platform rather than inferred from source APIs.

### Cross-package compatibility of the minimum candidate stack

- The candidate Python 3.11 stack is consistent with upstream package
  metadata: NumPy 1.26 supports Python >=3.9, SciPy 1.10.1 supports Python
  >=3.8,<3.12, NumExpr 2.10.1 supports Python >=3.9, and PyVista 0.46 supports
  Python >=3.9.
- VTK 9.3.1 publishes CPython 3.11 wheels, so the existing VTK lower bound is
  not an obstacle to testing Python 3.11.
- This establishes that the proposed minimum matrix is resolvable in
  principle; a full Nematics3D test run in that exact environment is still
  required before declaring the lower bounds validated.

## Validation still required

1. Run the full current desktop Qt6 suite with the relaxed GUI metadata on a
   machine with a real display-capable VTK backend.  HPCC can validate Qt API
   imports but its Qt/X11 VTK build cannot render with `DISPLAY` unset.
2. Test a PyQt6 release below 6.7 on a suitable desktop/CI platform before
   relaxing the remaining `PyQt6>=6.7` GUI floor.
3. Test Python 3.13 separately before treating the open-ended Python upper
   range as validated rather than merely syntactically permitted.
4. Build a release-style wheel against NumPy 2.x and verify that the native
   qdiag extension imports and behaves correctly with both NumPy 1.26 and
   NumPy 2.x runtimes.
5. Re-run the package build after restoring the local `build` tool in the
   rebuilt development environment; the current failure is `No module named
   build`, not a package-metadata failure.

## 2026-09-10 — HPCC Python 3.11 minimum-stack environment created

- Created `/home/yingyouma/.conda/envs/Nematics3D_min311` on HPCC with the
  intended candidate minimum runtime stack:
  Python 3.11.16, NumPy 1.26.4, SciPy 1.10.1, NumExpr 2.10.1,
  PyVista 0.46.4, VTK 9.3.1, and pytest 9.1.1.
- All of those packages import successfully when the environment library path
  is active.
- Installed the current code snapshot (`21748dc...`) into that environment in
  editable mode after applying only the candidate dependency metadata to the
  temporary HPCC copy.  The native qdiag extension built successfully under
  Python 3.11 + NumPy 1.26.4.
- The cached Python-3.11 VTK 9.3.1 build is the Qt/X11 variant, not the OSMesa
  variant.  Therefore a no-`DISPLAY` full pytest run is not a valid rendering
  acceptance test for this environment: collection reaches legacy
  `tests/smooth/test_smooth.py`, which renders at module import time and causes
  VTK to segfault without an X display.
- This rendering failure does not invalidate the numerical minimum-stack
  environment itself.  A full minimum-version correctness run should either
  use an X-capable session / GUI-capable node, or exclude/reclassify tests that
  render during collection.  Headless rendering remains covered separately by
  the accepted OSMesa reference environment.

## 2026-09-10 — strict minimum headless cross-check

- Created `/home/yingyouma/.conda/envs/Nematics3D_min311_headless` with
  Python 3.11.16, NumPy 1.26.4, SciPy 1.10.1, NumExpr 2.10.1,
  PyVista 0.46.4, and `vtk-base=9.3.1=*osmesa*`.
- Verified real no-`DISPLAY` off-screen rendering by writing a PNG through
  PyVista.  The installed VTK metadata is an `osmesa_py311` build.
- The first full pytest run produced `980 passed, 20 failed, 1 skipped,
  38 subtests passed`.
- Eighteen failures came from SciPy 1.10.1 rejecting the `axes=` keyword to
  `gaussian_filter`; SciPy 1.11.0 is the first release that supports it.
- Two failures came from NumPy-2-only calls to `np.asarray(..., copy=...)`.
  Further inspection showed a real compatibility issue in
  `SmoothedLine.__array__`: it forwarded `copy=None` to `np.asarray`, which
  also breaks ordinary `np.asarray(line)` under NumPy 1.26.  The implementation
  is now branched so `copy=None` uses the NumPy-1.x-compatible path, while
  NumPy 2.x still receives explicit `copy=True/False` semantics.  The tests
  now exercise the explicit `copy=` branch only when NumPy >=2.
- SciPy 1.11.0 was then tested explicitly, rather than inferring the floor
  from a later 1.11.x release.  Its `gaussian_filter` exposes the required
  `axes` parameter, and all 20 Gaussian-smoothing tests pass on SciPy 1.11.0.
- After applying the NumPy-1.26 compatibility branch to `SmoothedLine` and the
  version-aware tests, the complete strict minimum headless suite passes on
  HPCC with `1000 passed, 1 skipped, 38 subtests passed`.
- The exact accepted minimum-headless stack for this phase is therefore:
  Python 3.11.16, NumPy 1.26.4, SciPy 1.11.0, NumExpr 2.10.1,
  PyVista 0.46.4, and VTK 9.3.1 OSMesa.
- The NumPy-2.x side of the `SmoothedLine` array protocol was also rechecked
  in the local Windows environment: all 13 focused `SmoothedLine` tests pass.
- Ruff was restored after the local environment rebuild.  The touched
  `SmoothedLine` source and focused test file pass Ruff, and the focused local
  NumPy-2.x test remains 13/13 passing.

## 2026-09-11 — PyVista and VTK lower-bound audit

- After the local `Nematics3D` environment was rebuilt and Ruff was restored,
  the touched `SmoothedLine` source/test pair was linted again.  The remaining
  import-order and class-variable annotations were brought in line with the
  current repository style; Ruff reports `All checks passed!` and the focused
  `SmoothedLine` suite remains `13 passed`.
- Static inspection found no current Nematics3D use of an API that requires
  PyVista 0.46 specifically.  The repository already contains
  `extract_surface_compat()` to bridge older/newer `extract_surface()` API
  behavior.
- With the same Python 3.11 / NumPy 1.26.4 / SciPy 1.11.0 / NumExpr 2.10.1 /
  VTK 9.3.1 OSMesa baseline, the complete headless suite passes unchanged on
  PyVista 0.45.3, 0.44.2, and 0.43.10.  Each run produced
  `1000 passed, 1 skipped, 38 subtests passed`.
- A further combined lower-bound run used PyVista 0.43.10 with VTK 9.2.6
  OSMesa on Python 3.11.8, NumPy 1.26.4, SciPy 1.11.0, and NumExpr 2.10.1.
  The complete suite again produced
  `1000 passed, 1 skipped, 38 subtests passed`.
- Therefore the currently validated conservative core-visualization floors
  are `pyvista>=0.43,<1` and `vtk>=9.2,<10`.  Older releases are not being
  pursued merely to maximize nominal compatibility unless a concrete user or
  platform need appears.
- The GUI-specific lower bounds (`pyvistaqt`, `qtpy`, and `PyQt6`) remain to
  be audited separately before changing them.

## Final accepted support matrix

The dependency-relaxation phase is complete for the current release target.
The accepted metadata is:

- Python `>=3.11`;
- build-time NumPy `>=2.0,<3`;
- runtime NumPy `>=1.26,<3`;
- NumExpr `>=2.10.1,<3`;
- SciPy `>=1.11,<2`;
- PyVista `>=0.43,<1`;
- VTK `>=9.2,<10`;
- GUI extras: PyVistaQt `>=0.10,<1`, QtPy `>=2.2,<3`, and
  PyQt6 `>=6.7,<7`.

The strongest minimum-stack acceptance run is Python 3.11 with NumPy 1.26.4,
SciPy 1.11.0, NumExpr 2.10.1, PyVista 0.43.10, and VTK 9.2.6 OSMesa; the full
headless suite passes with `1000 passed, 1 skipped, 38 subtests passed`.

Python 3.13.15 passes source compilation, cp313 wheel construction, and direct
execution of the compiled `_core` extension under NumPy 2.5.3.  A full
Python-3.13 OSMesa pytest run is not currently possible on HPCC because the
available conda-forge solver has no matching `vtk-base=*osmesa*` combination.

The NumPy ABI cross-check also passes: one cp311 `_core` binary built against
NumPy 2.4.6 loads and executes successfully under both NumPy 1.26.4 and NumPy
2.4.6.  This directly supports the split build/runtime NumPy requirements.

On HPCC, fully isolated PEP 517 builds of modern NumPy may fall back to source
builds because the host cannot use current manylinux wheels; the system GCC
4.8 is then too old for current NumPy.  This is an HPCC toolchain limitation,
not a Nematics3D packaging defect.

The Python classifiers now include 3.11, 3.12, and 3.13. PyQt6 remains at the
intentionally conservative floor `>=6.7,<7`; no further lowering is part of
this release cycle.

## 2026-09-11 — GUI lower bounds and final compatibility checks

- The GUI dependency policy is now intentionally conservative rather than
  minimal.  `PyQt6>=6.7,<7` is retained by choice; older PyQt6 releases are no
  longer being pursued unless a concrete compatibility need appears.
- `qtpy==2.2.1` was exercised with PyQt6 on Python 3.11.  The Qt6 scoped enums
  used by Nematics3D (`Orientation`, `AlignmentFlag`, `WidgetAttribute`,
  `WindowType`, and `DockWidgetArea`) plus `QSignalBlocker`, `QTimer`, and
  `QFont.Monospace` all work through QtPy 2.2.1.
- `pyvistaqt==0.10.0` imports successfully with PyVista 0.43.10 and exposes a
  `BackgroundPlotter` constructor containing the options Nematics3D uses.
  The reviewed Nematics3D Qt modules and `visual.plot_figure` also import
  successfully with `pyvistaqt==0.10.0` and `qtpy==2.2.1`.
- A no-`DISPLAY` `BackgroundPlotter` construction attempt aborts in
  `vtkXOpenGLRenderWindow`.  This is the expected behavior of the Qt/X11 VTK
  package in that GUI test environment, not a QtPy/PyVistaQt API failure; the
  separate OSMesa environments remain the accepted headless-rendering path.
- The adopted GUI bounds are therefore `pyvistaqt>=0.10,<1`,
  `qtpy>=2.2,<3`, and `PyQt6>=6.7,<7`.

### NumPy build/runtime ABI cross-check

- A Python 3.11 build-only environment with NumPy 2.4.6 successfully built
  `nematics3d-0.9.0b1-cp311-cp311-linux_x86_64.whl` using the audited native
  Q-diagonalization extension.
- The exact `_core` extension from that single wheel was then loaded directly
  in two Python 3.11 runtimes: NumPy 1.26.4 and NumPy 2.4.6.
- In both runtimes the extension imported without an ABI error and
  `dominant_qfield5_into()` produced the same finite eigenvalue/eigenvector
  result for the same test tensors.
- This directly supports the packaging split used in `pyproject.toml`:
  build against NumPy 2.x (`numpy>=2.0,<3`) while supporting runtime NumPy
  1.26 and 2.x (`numpy>=1.26,<3`).
- PEP 517 isolated build on this HPCC host could not use the current NumPy
  PyPI wheel because of the host's old manylinux/glibc baseline; pip therefore
  fell back to building NumPy from source and hit the system GCC 4.8 compiler.
  A non-isolated wheel build against the already installed conda-forge NumPy
  2.x environment succeeds.  This is a host toolchain limitation rather than
  a Nematics3D extension compatibility failure.

### Python 3.13 cross-check

- Created a Python 3.13.15 + NumPy 2.5.3 core build environment on HPCC.
- The complete `src/nematics3d` tree passes `compileall` under Python 3.13.
- The package successfully builds a real
  `nematics3d-0.9.0b1-cp313-cp313-linux_x86_64.whl`.
- The `_core.cpython-313-...so` extension extracted from that wheel loads
  successfully under Python 3.13.15 + NumPy 2.5.3 and executes
  `dominant_qfield5_into()` with finite, correct-looking output.
- A full Python-3.13 headless acceptance run could not be constructed on this
  HPCC host because the current conda-forge solver cannot provide the requested
  Python-3.13 `vtk-base=*osmesa*` combination.  Python 3.13 is therefore
  accepted at the source/native-build level, while full OSMesa runtime
  acceptance remains an infrastructure gap rather than a known incompatibility.

## Current dependency conclusion

The audited metadata is now:

```toml
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.26,<3",
    "numexpr>=2.10.1,<3",
    "scipy>=1.11,<2",
    "pyvista>=0.43,<1",
    "vtk>=9.2,<10",
]

gui = [
    "pyvistaqt>=0.10,<1",
    "qtpy>=2.2,<3",
    "PyQt6>=6.7,<7",
]
```

The minimum Python-3.11 OSMesa stack is fully test-accepted, the NumPy native
ABI policy has been checked across NumPy 1.26/2.x runtimes, and Python 3.13
passes source compilation plus native wheel build/load.  No further dependency
relaxation is currently justified solely for the sake of advertising older
versions.
