# GridFieldDataset Optimization Review

## Purpose

This document records proposed improvements for `FieldData` and
`GridFieldDataset` so they can be reviewed and implemented one at a time. It is
an engineering review, not an instruction to rewrite the classes immediately.

Relevant source files:

- `src/nematics3d/classes/grid_field/grid_field_dataset.py`
- `src/nematics3d/classes/grid_field/grid_field_dataset_derivatives.py`
- `src/nematics3d/classes/grid_field/grid_field_dataset_smoothing.py`
- `src/nematics3d/classes/grid_field/grid_interpolator.py`
- `src/nematics3d/classes/grid_field/input_grid_field.py`

## Overall recommendation

Do not replace `GridFieldDataset` in one large rewrite. The shared-grid model is
useful and the core file is still manageable. Apply staged improvements in this
order:

1. Fix confirmed behavioral defects and API inconsistencies.
2. Reduce eager coordinate-grid memory use.
3. Define geometry and field mutability contracts.
4. Improve module/class organization only after behavior is stable.

Each proposal below should receive its own review, focused tests, implementation
commit, and archive decision.

## Current responsibility map

`GridFieldDataset` currently owns or coordinates:

- shared lattice shape and boundary periodicity;
- index-to-physical grid transform and offset;
- full index-space and physical-space coordinate caches;
- corners, bounds, center, grid spacing, and periodic box size;
- physical-field registration and replacement;
- the reserved validity-mask field;
- interpolation through `FieldData`;
- spatial derivatives and tensor/vector differential operators;
- Gaussian smoothing and optional smoothing weights;
- result payload conversion and registration.

The responsibilities are related, but their cache and lifecycle contracts must
be made explicit before the whole component can be archived.

## Proposal checklist

### 1. Fix Gaussian smoothing along component axes

**Status:** Proposed; confirmed behavioral defect.

**Observed behavior**

Fields may have shapes such as `(Nx, Ny, Nz, 3)` or
`(Nx, Ny, Nz, 3, 3)`. Some smoothing paths pass a three-element sigma to
`scipy.ndimage.gaussian_filter` without restricting the filter axes. SciPy then
expects sigma to match the complete four- or five-dimensional rank and raises
an error.

**Physical contract**

Gaussian smoothing must operate only along the first three spatial lattice
axes. Director, vector, and tensor component axes are independent components
and must never be mixed by spatial smoothing.

**Suggested change**

- Explicitly restrict filtering to axes `(0, 1, 2)`, if the supported SciPy API
  permits it; otherwise extend sigma/order/mode to the full rank with zero
  smoothing on trailing component axes.
- Apply the same rule to weighted values and smoothing weights.
- Preserve every trailing component axis exactly.

**Acceptance criteria**

- Scalar `(Nx, Ny, Nz)`, vector `(Nx, Ny, Nz, C)`, and tensor
  `(Nx, Ny, Nz, C1, C2)` inputs all work.
- No cross-component mixing occurs.
- Periodic/non-periodic boundary modes apply only to spatial axes.
- Weighted and unweighted paths agree for uniform unit weights.
- Existing scalar-field behavior remains unchanged.

### 2. Resolve the `FieldData.interpolator` API mismatch

**Status:** Proposed; confirmed API inconsistency.

**Observed behavior**

`FieldData` stores `entity_interpolator`, while existing callers/tests expect a
readable `field.interpolator` surface. This currently produces an
`AttributeError` after `act_add_interpolator()` succeeds.

**Decision required**

Choose and document one public contract:

- Recommended: keep `entity_interpolator` as managed storage and expose a
  read-only `interpolator` property.
- Alternative: make `entity_interpolator` the only public name and migrate all
  callers/tests explicitly.

**Acceptance criteria**

- Repeated `act_add_interpolator()` calls return the same live object.
- The chosen readable API returns that same object.
- `QFieldObject.interpolator` and its canonical Q `FieldData` agree.
- No duplicate interpolators are silently created.

### 3. Make full coordinate grids lazy

**Status:** Proposed; high-value memory optimization.

**Observed behavior**

Initialization eagerly creates both:

- `calc_grid_index`, shape `(Nx, Ny, Nz, 3)`;
- `calc_grid`, shape `(Nx, Ny, Nz, 3)`.

For float64 data on a `256^3` grid, each array is approximately 384 MiB and the
pair is approximately 768 MiB. This is before storing Q, director, scalar-order,
mask, interpolation, or analysis arrays.

**Suggested change**

- Store only shape, transform, offset, spacing, corners, center, and bounds at
  construction.
- Generate complete coordinate grids only when requested.
- Decide whether requested grids are cached, weakly cached, or regenerated.
- Provide direct coordinate conversion for point subsets so most callers never
  require a full grid.

**Acceptance criteria**

- Constructing a dataset does not allocate either full coordinate grid.
- `calc_grid_index` and `calc_grid` preserve their current numerical convention
  when accessed.
- corners, center, bounds, interpolation, and defect transforms do not force
  full-grid allocation.
- A memory benchmark records the before/after initialization footprint for at
  least `128^3` and `256^3` grids.

### 4. Define grid geometry as immutable or fully invalidating

**Status:** Decision required; correctness risk.

**Observed behavior**

The following inputs jointly determine every geometry cache and interpolator:

- `raw_shape`;
- `raw_box_periodic_flag`;
- `raw_grid_offset`;
- `raw_grid_transform`.

Cache rebuilding currently relies on explicit calls to
`_helper_refresh_grid_cache()`. Public or in-place mutation could leave spacing,
coordinates, corners, bounds, periodic size, and interpolators inconsistent.

**Recommended contract**

Treat shared-grid geometry as immutable after dataset construction or initial
shape inference. To use different geometry, construct a new dataset.

**Alternative contract**

If mutation is required, implement one atomic update path that validates all
new geometry, invalidates every dependent cache/interpolator/result, and then
rebuilds consistent state.

**Acceptance criteria**

- No supported public operation can leave partially refreshed geometry.
- Stored transform/offset/periodicity arrays cannot be mutated in place.
- Shape inference happens at most once and is well tested.
- The effect on existing fields and interpolators is explicit.

### 5. Define physical-field value mutability and stale-result behavior

**Status:** Decision required; repository-wide policy may be needed.

**Observed behavior**

Compatible floating arrays may share memory with callers. A caller can mutate
field values after registration, while interpolators and derived results may
already exist. The dataset has no general stale-result mechanism.

**Options**

- Store read-only defensive copies for registered fields.
- Store read-only views while documenting that caller-owned backing storage
  must not be changed.
- Allow mutation only through an explicit replacement/update action that
  invalidates interpolators and derived results.

**Recommended direction**

Registered physical fields should be stable snapshots. Use an explicit
replacement operation for new data and invalidate/rebuild field-owned derived
objects there.

**Acceptance criteria**

- The chosen ownership rule is documented for `FieldData.raw_values`.
- Caller mutation cannot silently change a registered field, or is explicitly
  guaranteed and tested to invalidate dependents.
- Replacing a field handles any existing interpolator deterministically.

### 6. Preserve the construction-only validity-mask lifecycle

**Status:** Recommended to keep; align tests and documentation.

**Current design**

The reserved `mask` field can be provided at dataset construction but cannot be
added or replaced later with `act_add_field("mask", ...)`. This prevents defect,
smoothing, and interpolation results from becoming stale after a mask change.

**Suggested change**

- Keep the construction-only mask contract unless a complete invalidation
  system is introduced.
- Update obsolete tests that assume a dynamically registered field named
  `"mask"` can serve as smoothing weights.
- Allow ordinary soft smoothing weights under a non-reserved field name.
- Keep the physical validity mask strictly boolean; do not conflate it with
  continuous confidence/weight fields.

**Acceptance criteria**

- Dataset and Q-field initialization expose the same canonical mask.
- Defect filtering, interpolation validity, and mask-weighted operations use
  that canonical source.
- Reserved-name errors clearly explain how to supply a validity mask.
- Soft weights remain supported without being treated as validity flags.

### 7. Separate core geometry, field registry, and operators more explicitly

**Status:** Proposed after behavioral stabilization.

**Observed structure**

Core storage lives in `grid_field_dataset.py`, while derivative and smoothing
functions are imported and assigned to `GridFieldDataset` at module load time:

```python
GridFieldDataset.act_gradient = act_gradient
GridFieldDataset.act_gaussian_smooth = act_gaussian_smooth
```

The file split is useful, but runtime method assignment weakens static typing,
IDE navigation, API discovery, and the readability of the class definition.

**Suggested change**

After operator behavior is stable, consider explicit mixins:

```python
class GridFieldDataset(
    GridFieldDerivativeMixin,
    GridFieldSmoothingMixin,
    ClassBase,
):
    ...
```

Keep domain-independent geometry/field lifecycle in the core class and keep
operator-specific result metadata near each operator family.

**Acceptance criteria**

- No runtime monkey-patching is required for public methods.
- Public imports and method names remain compatible unless a migration is
  explicitly approved.
- Circular imports do not increase.
- Type checkers and IDEs can discover operator methods from the class MRO.

### 8. Review eager `Bounds` construction and replacement

**Status:** Proposed; lower priority than full-grid memory.

**Observed behavior**

Every geometry-cache refresh constructs a new `Bounds` domain object and
registers protected opts. If mutable geometry is retained, repeated refreshes
could leave external references to obsolete bounds objects.

**Suggested change**

- Under immutable geometry, construct bounds once.
- Under mutable geometry, specify whether bounds identity is preserved and
  updated or replaced with explicit invalidation.
- Confirm that `QFieldObject.objects` registers the canonical bounds only once.

**Acceptance criteria**

- Bounds identity/lifecycle is deterministic.
- No duplicate or obsolete bounds remain registered.
- corners, center, and bounds use the same cell-center/box-edge convention.

### 9. Clarify field replacement semantics

**Status:** Proposed.

**Observed behavior**

`is_replace=True` unregisters an existing `FieldData` and creates a new one.
Any external reference to the old field or its interpolator remains a live
Python object but is no longer canonical in the dataset.

**Suggested change**

- Document replacement as identity replacement, or update values in place if
  stable field identity is required.
- Explicitly detach or mark replaced fields/interpolators stale when practical.
- Test name reuse, owner relations, external references, and result metadata.

**Acceptance criteria**

- Dataset lookup always returns the new canonical field.
- Old field/interpolator behavior is explicit rather than accidental.
- Replacement cannot affect the reserved validity-mask field.

### 10. Establish performance and regression baselines before class archival

**Status:** Required before final component archival.

Record at minimum:

- construction time and peak memory for representative grid sizes;
- field registration and replacement costs;
- lazy versus realized coordinate-grid costs;
- interpolation creation and reuse;
- scalar/vector/tensor derivative correctness;
- scalar/vector/tensor smoothing correctness;
- periodic and non-periodic boundary behavior;
- mask-enabled Q-field initialization and defect filtering.

All known unrelated failures should either be fixed or recorded as explicit
remaining limitations before `FieldData` or `GridFieldDataset` is marked
confirmed in the beta-review ledger.

## Recommended implementation sequence

Use the following order to minimize overlapping changes:

1. Gaussian spatial-axis smoothing fix.
2. `FieldData.interpolator` public API decision and fix.
3. Validity-mask and soft-weight test alignment.
4. Geometry immutability decision.
5. Lazy full coordinate grids and memory benchmark.
6. Registered-field mutability/replacement contract.
7. Bounds lifecycle review.
8. Optional derivative/smoothing mixin conversion.
9. Full downstream regression and separate archival of `FieldData` and
   `GridFieldDataset`.

## Decisions explicitly deferred

- Whether `GridFieldDataset` should eventually inherit from `HostBase`.
- A repository-wide mutable-array policy for all `calc_` and `entity_` values.
- General stale-result propagation across unrelated physical domain objects.
- Broad renaming of established `raw_`, `calc_`, and `entity_` surfaces.

These questions are larger than the dataset alone and should not block the
focused correctness and memory improvements above.
