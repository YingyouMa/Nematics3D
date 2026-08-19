# Nematics3D First Public Beta Release Checklist

This document tracks the work required before the first public beta release of
Nematics3D. The release should prioritize installability, scientific
correctness, API clarity, and useful feedback collection. Style compliance is
important, but it is not sufficient on its own to make the package ready for
external users.

## Release objective

The first public beta is ready when an external user can:

- install Nematics3D from a public package channel in a clean environment;
- run the documented minimum example successfully;
- obtain scientifically validated results for the core workflows;
- understand the supported platforms, public API, and known limitations; and
- report a reproducible problem through a clearly documented channel.

## Release-blocking work

### 1. Build and installation validation

- [ ] Build both a source distribution (`sdist`) and a wheel from a clean
      checkout.
- [ ] Run `twine check` against all generated distribution artifacts.
- [ ] Inspect the distribution contents and confirm that required source,
      license, metadata, and package data are included.
- [ ] Install the built wheel into a new environment rather than testing only
      from the repository source tree.
- [ ] Change to a directory outside the repository and verify
      `import nematics3d`.
- [ ] Run a minimal numerical workflow using the installed wheel.
- [ ] Run an off-screen visualization smoke test using the installed wheel.
- [ ] Upload a release candidate to TestPyPI and test installation from the
      uploaded artifact.
- [ ] Confirm that the same source archive can be used as the verified source
      for the conda-forge recipe.

### 2. Dependency and installation design

The initial metadata pins every runtime dependency to one exact version. Exact
pins are useful for reproducing a development environment, but are normally too
restrictive for a reusable library and can cause unnecessary pip and conda
dependency conflicts.

- [ ] Replace exact runtime pins with tested lower/upper bounds where
      appropriate.
- [ ] Keep a separate reproducible development or release environment if exact
      dependency versions are required internally.
- [ ] Define the smallest set of dependencies required for core numerical use.
- [ ] Decide whether `pyvistaqt`, `PyQt5`, and related desktop dependencies
      should move to a `gui` optional dependency group.
- [ ] Ensure that a headless/server installation does not require an
      unnecessary desktop GUI stack.
- [ ] Clearly distinguish installation modes such as:
  - core: `nematics3d`;
  - GUI: `nematics3d[gui]`;
  - tutorials: `nematics3d[tutorials]`.
- [ ] Verify that each advertised installation mode resolves and installs in a
      clean environment.
- [ ] Decide whether Python 3.12-only support is intentional for the first
      beta; document the decision explicitly.
- [ ] Test the Qt, VTK, PyVista, NumPy, and SciPy combinations that will be
      declared as supported.

### 3. Scientific correctness and regression protection

Unit tests must protect scientific meaning as well as code execution. Maintain
small, deterministic reference fields for important physical and geometric
cases.

- [ ] Test Q-tensor construction from supported input representations.
- [ ] Test Q-tensor diagonalization, including degenerate or nearly degenerate
      cases.
- [ ] Test director normalization and invalid input handling.
- [ ] Test periodic and non-periodic boundary behavior.
- [ ] Check expected defect-point counts on reference fields.
- [ ] Check disclination-line count, closed/open status, connectivity, and
      length ranges on reference fields.
- [ ] Check known topology/classification results on analytically understood or
      independently verified examples.
- [ ] Verify important geometric properties before and after line smoothing.
- [ ] Test coordinate transformations involving axis order, spacing, origin,
      offsets, cropping, and physical coordinates.
- [ ] Test behavior around invalid values, masks, zero order parameter, empty
      inputs, and undersized domains.
- [ ] Use documented numerical tolerances rather than exact floating-point
      equality.
- [ ] Record why each scientific tolerance is acceptable.
- [ ] Keep reference data small enough for reliable CI execution.
- [ ] Document how reference results were generated or independently checked.

### 4. Public API definition

- [ ] Identify the modules, functions, classes, methods, and options that form
      the supported public beta API.
- [ ] Mark internal implementation modules and names clearly.
- [ ] Decide whether wildcard or top-level exports expose only intentional
      public objects.
- [ ] Define a beta deprecation policy for renamed or removed public APIs.
- [ ] Define the expected compatibility policy between beta releases.
- [ ] Document which serialized objects or saved settings, if any, are expected
      to remain compatible.
- [ ] Confirm that public function and class docstrings describe:
  - accepted types and array shapes;
  - axis ordering and coordinate conventions;
  - units or dimensionless assumptions;
  - periodic-boundary behavior;
  - whether inputs are copied or modified;
  - return values and generated objects;
  - important exceptions and invalid states;
  - numerical limitations and applicable physical regimes.

### 5. User-facing documentation

- [ ] Replace README links containing local paths such as
      `/D:/Document/GitHub/Nematics3D/...` with repository-relative links or
      stable public URLs.
- [ ] Make public package installation (`pip` and conda) the primary README
      path after those channels are available.
- [ ] Keep installation from source as a separate developer/advanced-user
      section.
- [ ] Provide one minimal example that can be copied and run without hidden
      setup steps.
- [ ] Confirm that all README examples use the final public API.
- [ ] Document headless/server, notebook, and desktop GUI installation and use.
- [ ] Add troubleshooting guidance for Qt, VTK, display servers, off-screen
      rendering, and remote-cluster use.
- [ ] Explain expected logging/progress output so it is not mistaken for an
      error.
- [ ] Document current scientific and technical limitations honestly.
- [ ] Verify README rendering in the built package and on TestPyPI.
- [ ] Check all public documentation links.

### 6. Versioning, changelog, and release artifacts

The package metadata currently declares `0.9.0b1`, while the changelog ends at
`0.1.7`. These must be reconciled before release.

- [ ] Add a complete changelog entry for `0.9.0b1`.
- [ ] Summarize major additions, fixes, breaking changes, and known issues.
- [ ] Ensure the runtime `nematics3d.__version__`, package metadata, Git tag,
      and release title agree.
- [ ] Decide and document the progression from beta to stable, for example
      `0.9.0b1` to `0.9.0b2`, `0.9.0rc1`, and `0.9.0`.
- [ ] Create an annotated, immutable release tag such as `v0.9.0b1`.
- [ ] Ensure a version is built from exactly one reviewed commit.
- [ ] Create a GitHub Release containing installation instructions, highlights,
      compatibility information, and known issues.
- [ ] Verify that the source archive attached to the release includes the
      license and required package files.

### 7. Continuous integration for critical behavior

- [ ] Add a GitHub Actions test workflow.
- [ ] Run tests on Windows and Linux.
- [ ] Cover every Python version officially advertised by the project.
- [ ] Separate fast core tests from GUI or visualization tests where useful.
- [ ] Run headless visualization smoke tests in CI.
- [ ] Test the built wheel, not only an editable/source checkout.
- [ ] Require the release-blocking CI checks to pass before merging into
      `main`.
- [ ] If macOS is not tested, clearly state that it is currently unverified
      rather than implicitly promising support.

## Strongly recommended for the first beta

### 8. Automated style and quality checks

- [ ] Run `black --check` in CI.
- [ ] Add a focused Ruff configuration for high-confidence lint and import
      checks.
- [ ] Check packaging metadata and distribution builds in CI.
- [ ] Add documentation build or link validation.
- [ ] Introduce type checking incrementally for important public interfaces if
      it can be maintained consistently.
- [ ] Avoid a release-time whole-repository rewrite solely to satisfy a large
      new lint rule set.

### 9. Secure and reproducible publishing

- [ ] Publish PyPI releases through a dedicated GitHub Actions workflow.
- [ ] Prefer PyPI Trusted Publishing over a long-lived API token.
- [ ] Trigger publication only from an explicit release tag or protected GitHub
      environment.
- [ ] Require manual environment approval for the first few public releases if
      practical.
- [ ] Protect `main` and release tags against accidental history changes.
- [ ] Enable dependency and security alerts.
- [ ] Confirm that distribution artifacts do not contain credentials, local
      paths, private research data, caches, or development-only files.
- [ ] Add a `SECURITY.md` explaining how to report a vulnerability privately.

### 10. conda-forge preparation

- [ ] Publish or otherwise provide a stable, checksummed source archive.
- [ ] Generate a conda-forge recipe from the public package metadata and then
      review it manually.
- [ ] Declare accurate build, host, run, test, and optional GUI dependencies.
- [ ] Include the license file and SPDX-compatible license metadata.
- [ ] Add an import test and at least one minimal functional test to the recipe.
- [ ] Test the recipe locally where practical.
- [ ] Submit the recipe through `conda-forge/staged-recipes`.
- [ ] Identify the long-term feedstock maintainer(s).
- [ ] Plan to review dependency migration and version-update pull requests after
      the feedstock is created.

### 11. External-user feedback and contribution workflow

- [ ] Add a GitHub bug-report issue template.
- [ ] Add a feature-request or scientific-validation issue template.
- [ ] Ask bug reporters for:
  - Nematics3D version;
  - installation channel;
  - operating system and Python version;
  - relevant dependency versions;
  - complete traceback or log;
  - a minimal reproducer and suitably reduced input data.
- [ ] Explain how to remove confidential or oversized research data before
      uploading a reproducer.
- [ ] Expand `CONTRIBUTING.md` with environment setup, formatting, test, and PR
      commands.
- [ ] Add a code of conduct if the project intends to accept community
      contributions.
- [ ] Label issues for installation, numerical correctness, visualization,
      documentation, and API feedback.

## Work that may continue during later beta releases

These improvements are valuable but should not delay the first beta unless they
expose a concrete correctness or usability problem:

- [ ] Complete static type coverage across the entire codebase.
- [ ] Publish a performance benchmark dashboard.
- [ ] Build comprehensive generated API documentation.
- [ ] Add fully validated macOS support.
- [ ] Stabilize long-term serialization compatibility.
- [ ] Formalize extension/plugin interfaces.
- [ ] Guarantee a stable API beyond the explicitly documented beta surface.

## Suggested implementation order

1. Redesign dependency groups and compatible version ranges.
2. Add clean build, wheel installation, and smoke-test CI.
3. Add scientific reference and regression tests for core workflows.
4. Define and document the public beta API.
5. Repair README links and rewrite installation documentation for public users.
6. Reconcile the version, changelog, package metadata, tags, and release notes.
7. Automate formatting, linting, packaging, and secure publication.
8. Validate on TestPyPI.
9. Publish the tagged beta release.
10. Submit and validate the conda-forge recipe.

## Final release decision

The first public beta should not be released until all of the following are
true:

- [ ] A clean external environment can install the published artifact.
- [ ] The documented minimal workflow runs from the installed package.
- [ ] Core scientific results are protected by meaningful regression tests.
- [ ] Supported platforms and dependency constraints are explicit.
- [ ] Public and internal APIs are distinguishable.
- [ ] Known limitations and breaking changes are documented.
- [ ] Users have a clear, reproducible way to report problems.
