# Gaussian Spatial Smoothing Workflow

This note records a staged workflow for adding Gaussian-kernel spatial
smoothing to `GridFieldDataset`.

## Current Status

- Step 1 completed:
  API contract, sigma/boundary normalization, result metadata skeleton
- Step 2 completed:
  separable real-space Gaussian smoothing implementation
- Step 3 completed:
  result registration/save-load checks and smoothing-to-derivative chaining tests
- Step 4:
  document the intended user-facing workflow and usage patterns

## Goal

Add a dataset-level helper for smoothing scalar, vector, and tensor lattice
fields on the shared grid:

```python
dataset.act_gaussian_smooth(
    field_or_values,
    sigma,
    *,
    coord="physical",
    truncate=4.0,
    boundary="auto",
    is_result=False,
)
```

The helper should follow the same general usage style as the existing dataset
derivative operators.

## Current Geometry Assumption

Current `grid_transform` validation already rejects:

- shear
- reflections
- degenerate axes

It only allows a right-handed orthogonal basis with per-axis scaling. In
practice this means `grid_transform` is restricted to rotation plus axis-wise
scale. Under this constraint, a physical-space Gaussian remains separable along
the dataset lattice axes after converting `sigma` into index units.

This is the key reason the first implementation can use three 1D convolutions
instead of a fully general dense 3D kernel.

## High-Level Workflow

1. Define the smoothing semantics.
2. Convert `sigma` into per-axis index-space widths.
3. Build truncated 1D Gaussian kernels.
4. Apply separable convolution along the three spatial axes.
5. Handle periodic and non-periodic boundaries per axis.
6. Return either raw values or a result object.
7. Optionally register the result as a derived dataset field.

## Step 1: Define the API Contract

Recommended first-pass signature:

```python
dataset.act_gaussian_smooth(
    field_or_values,
    sigma,
    *,
    coord="physical",
    truncate=4.0,
    boundary="auto",
    is_result=False,
)
```

Recommended semantics:

- `field_or_values`:
  registered field name, field object, or temporary array
- `sigma`:
  either a scalar or a length-3 input
- `coord`:
  `"physical"` or `"index"`
- `truncate`:
  Gaussian cutoff radius in units of sigma
- `boundary`:
  first implementation uses `"auto"` as the main user-facing mode
- `is_result`:
  matches the existing dataset derivative workflow

## Step 2: Convert Sigma to Index Units

All convolution is executed along the first three lattice axes, so smoothing
width must be represented in index units.

### Case A: `coord="index"`

Interpret `sigma` directly in lattice-index units.

Examples:

- scalar `sigma=1.5` -> `(1.5, 1.5, 1.5)`
- vector `sigma=(1.0, 2.0, 3.0)` -> unchanged

### Case B: `coord="physical"`

Use `dataset.calc_grid_spacing`:

```python
sigma_index = sigma_physical / grid_spacing
```

Axis by axis:

```python
sigma_i = sigma_x / dx
sigma_j = sigma_y / dy
sigma_k = sigma_z / dz
```

Because current `grid_transform` is restricted to orthogonal axes with
scaling, no extra cross-term handling is needed here.

## Step 3: Build 1D Gaussian Kernels

For each spatial axis, construct:

```python
g[n] = exp(-0.5 * (n / sigma_axis) ** 2)
```

with truncation radius:

```python
radius = ceil(truncate * sigma_axis)
```

Then normalize each kernel so that its sum is `1.0`.

Notes:

- if `sigma_axis <= 0`, skip smoothing on that axis
- if `radius == 0`, that axis also becomes a no-op

## Step 4: Apply Separable Convolution

Apply the 1D kernel independently along the first three axes:

1. smooth along axis 0
2. smooth along axis 1
3. smooth along axis 2

Trailing axes are treated as component axes and must be preserved.

This means the helper should work for:

- scalar fields: `(Nx, Ny, Nz)`
- vector fields: `(Nx, Ny, Nz, 3)`
- tensor fields: `(Nx, Ny, Nz, ...)`

## Step 5: Boundary Handling

Recommended first-pass `boundary="auto"` behavior:

- periodic axis -> `wrap`
- non-periodic axis -> `reflect`

This matches the dataset's per-axis periodic metadata and gives sensible
default behavior for non-periodic grids without artificial zero-padding.

Mixed periodicity must be supported naturally, for example:

- `(True, True, True)` -> wrap on all axes
- `(True, False, False)` -> wrap on one axis, reflect on two axes
- `(False, False, False)` -> reflect on all axes

Possible future expansion:

- `"nearest"`
- `"constant"`
- explicit per-axis boundary override

## Step 6: Return Type

Keep the output contract aligned with the existing derivative helpers.

### `is_result=False`

Return the smoothed `np.ndarray`.

### `is_result=True`

Return a result object carrying:

- smoothed values
- source field name if available
- source shape
- `coord`
- user `sigma`
- converted `sigma_index`
- `truncate`
- boundary mode
- periodic flags
- grid transform
- grid offset

This result should be compatible with the existing dataset pattern:

```python
smooth = dataset.act_gaussian_smooth("Q", sigma=2.0, is_result=True)
dataset.act_add_result_field("Q_smooth", smooth)
```

## Step 7: Recommended User Workflow

When smoothing is used before derivatives or other spatial operators:

```python
Q_smooth = dataset.act_gaussian_smooth("Q", sigma=1.5, is_result=True)
dataset.act_add_result_field("Q_smooth", Q_smooth)

grad_Q = dataset.act_gradient("Q_smooth")
lap_Q = dataset.act_componentwise_laplacian("Q_smooth")
```

This keeps:

- the original field
- the smoothed field
- downstream derived fields

all available inside the same dataset.

## Step 4: User-Facing Usage Notes

The current implementation is now usable through `GridFieldDataset`.

### Basic usage

If only the smoothed values are needed immediately:

```python
values_smooth = dataset.act_gaussian_smooth(
    "Q",
    sigma=1.5,
)
```

### Keep smoothing metadata and register the result

If the smoothed field should remain inside the dataset for later reuse:

```python
smooth_result = dataset.act_gaussian_smooth(
    "Q",
    sigma=1.5,
    is_result=True,
)
dataset.act_add_result_field("Q_smooth", smooth_result)
```

### Smooth in physical coordinates

Use `coord="physical"` when `sigma` should be interpreted in real-space units:

```python
smooth_result = dataset.act_gaussian_smooth(
    "Q",
    sigma=2.0,
    coord="physical",
    is_result=True,
)
```

The implementation converts the requested physical width into per-axis index
widths using `dataset.calc_grid_spacing`.

### Smooth in index coordinates

Use `coord="index"` when `sigma` should be interpreted directly in lattice-grid
units:

```python
values_smooth = dataset.act_gaussian_smooth(
    "Q",
    sigma=(1.0, 2.0, 3.0),
    coord="index",
)
```

### Boundary behavior

The current implementation uses real-space separable convolution for both
periodic and non-periodic datasets.

- periodic axis -> `wrap`
- non-periodic axis -> `reflect`

This is the behavior selected by:

```python
boundary="auto"
```

### Typical downstream workflow

The intended main workflow is:

```python
Q_smooth = dataset.act_gaussian_smooth("Q", sigma=1.5, is_result=True)
dataset.act_add_result_field("Q_smooth", Q_smooth)

grad_Q = dataset.act_gradient("Q_smooth")
lap_Q = dataset.act_componentwise_laplacian("Q_smooth")
```

This keeps the original field, the smoothed field, and the derivative fields in
one dataset-managed chain.

## Implementation Breakdown

Suggested implementation sequence:

1. Add sigma-normalization helper.
2. Add 1D Gaussian-kernel builder.
3. Add 1-axis convolution helper with boundary handling.
4. Add dataset-level real-space smoothing helper.
5. Add public `act_gaussian_smooth(...)`.
6. Add result metadata type if needed.
7. Add tests.

## Testing Checklist

Minimum tests for first implementation:

1. Constant field remains unchanged.
2. Delta-like peak smooths into the expected Gaussian profile.
3. `sigma=0` behaves like no smoothing.
4. Mixed periodic and non-periodic axes use the expected boundary behavior.
5. Trailing component axes are preserved exactly.
6. Scalar and vector/tensor inputs both work.
7. `coord="physical"` correctly uses `calc_grid_spacing`.
8. Result metadata is populated correctly when `is_result=True`.
9. Saved/reloaded result workflow still works if a result class is introduced.

## Open Design Questions

Questions to settle before implementation:

1. Should `sigma` accept only scalar and length-3 inputs, or also NumPy arrays?
2. Should first-pass support only `boundary="auto"`, or expose explicit modes now?
3. Should Gaussian smoothing get its own result dataclass, or reuse the existing
   result pattern with a parallel metadata structure?
4. Should there also be a convenience method that smooths and immediately
   registers a named field?

## Proposed First Execution Order

Recommended order for the next steps:

1. Completed: freeze the public API.
2. Completed: decide the metadata/result shape.
3. Completed: implement kernel and boundary helpers.
4. Completed: implement dataset smoothing.
5. Completed: add core tests.
6. Completed: verify smoothing -> derivative chained examples.
7. Next: decide whether to add public docs/example files.
8. Later: decide whether to add a periodic Fourier fast path.
