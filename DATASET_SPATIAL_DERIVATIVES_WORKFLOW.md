# Dataset Spatial Derivatives Workflow

Temporary design note for the planned `GridFieldDataset` spatial derivative
workflow. Delete or migrate this file after the implementation settles.

## Core Decision

Spatial derivative operations should be defined on `GridFieldDataset`, because
the dataset owns the shared lattice shape, periodic-boundary flags,
`grid_transform`, and `grid_offset`.

The finite-difference stencil should be applied in lattice/index space first.
If physical-coordinate derivatives are requested, transform only the derivative
direction axis afterward using the chain rule.

For the repository grid convention:

```text
x_physical = x_index @ A + b
```

where:

- `A = dataset.raw_grid_transform`
- `b = dataset.raw_grid_offset`

the gradient conversion is:

```text
grad_physical = grad_index @ inv(A).T
```

`grid_offset` does not affect derivatives.

## Data Layout

Field values keep the current repository convention:

```text
field shape:      (Nx, Ny, Nz, ...)
gradient shape:   (Nx, Ny, Nz, ..., 3)
```

The final length-3 axis is the derivative direction axis:

```text
[..., 0] = derivative along x/index-0
[..., 1] = derivative along y/index-1
[..., 2] = derivative along z/index-2
```

For scalar fields, the gradient shape is `(Nx, Ny, Nz, 3)`.
For vector fields, the gradient shape is `(Nx, Ny, Nz, 3, 3)`, where the
second-to-last axis is the vector component axis and the final axis is the
derivative direction axis.

For compressed tensor fields such as Q5, the generic gradient should preserve
the compressed component axis, for example `(Nx, Ny, Nz, 5, 3)`. Tensor-specific
contractions or Q5-to-Q9 expansion should belong to the higher-level Q-field
workflow, not to the generic dataset derivative helper.

## Suggested API Shape

The first implementation target should be a minimal, reliable gradient helper:

```python
grad = dataset.act_gradient("Q", coord="physical")
```

Accepted input should include both:

- a field name registered in `dataset.fields`
- a temporary NumPy array whose first three axes match `dataset.raw_shape`

This allows chained expressions without forcing temporary derived quantities to
be stored in the dataset.

Useful follow-up helper:

```python
dA_dx = dataset.act_derivative("A", direction="x", coord="physical")
```

`act_derivative()` can be a thin convenience wrapper around `act_gradient()`.

## Storage Policy

Derivative outputs should not be stored in the dataset by default.

They should be ordinary returned arrays, because derived quantities can be large
and depend on derivative options such as coordinate mode, boundary handling,
stencil choice, and edge order.

If storage is needed later, make it explicit, for example by adding a derived
field manually:

```python
grad_Q = dataset.act_gradient("Q", coord="physical")
dataset.act_add_field("grad_Q", grad_Q)
```

When metadata should be preserved, attach it through the field `info` surface:

```python
grad_result = dataset.act_gradient("Q", coord="physical", is_result=True)
dataset.act_add_field("grad_Q", grad_result.raw_values, info=grad_result.raw_info)
```

The preferred convenience form is:

```python
dataset.act_add_result_field("grad_Q", grad_result)
```

which stores `grad_result.raw_values` as the field payload and
`grad_result.raw_info` as the field metadata.

Field `info` is intentionally free-form provenance or metadata. Core numerical
logic should not depend on its structure.

## Result Metadata

Derivative helpers may return an inspectable `ResultBase` dataclass when the
caller asks for metadata explicitly:

```python
grad_result = dataset.act_gradient("Q", coord="physical", is_result=True)
```

The default return value remains a plain `np.ndarray` so chained calculations
stay lightweight.

The result object should keep the computed array in `raw_values`, not `values`,
because `ResultBase` already exposes a dict-like `.values()` method. Metadata
should live in `raw_info`, a payload-free `SpatialDerivativeInfo` result object
that describes the immediate operation, including:

- `operator`
- `source`
- `source_shape`
- `coord`
- `derivative_axis`
- `component_axis`
- `input_component_shape`
- `output_shape`
- `box_periodic_flag`
- `grid_transform`
- `grid_offset`
- `stencil`
- `edge_order`

## Boundary Handling

Finite differences should operate on the first three axes.

Periodic dimensions should use periodic stencils, likely based on `np.roll`.
Non-periodic dimensions should use centered differences in the interior and
one-sided differences at the boundary.

The boundary behavior should follow `dataset.raw_box_periodic_flag`.

## Coordinate Modes

Recommended coordinate options:

- `coord="physical"`: return physical Cartesian derivatives.
- `coord="index"`: return derivatives with respect to lattice/index axes.

The default should probably be `coord="physical"`, because user-facing physical
quantity calculations usually expect physical derivatives. Internally, both
modes still compute finite differences in index space first.

`GRID_TRANSFORM_IDENTITY` and `None` should use a fast path and avoid a matrix
inverse.

## Chained Expression Example

For an expression like:

```text
d/dx( (dA/dx) * B )
```

the intended workflow is:

```python
B = dataset["B"].raw_values

dA_dx = dataset.act_derivative("A", direction="x", coord="physical")
flux = dA_dx * B
result = dataset.act_derivative(flux, direction="x", coord="physical")
```

The temporary `flux` array is not stored unless the caller explicitly stores it.

## Higher-Level Operations

After the gradient helper is stable, common named operations can be layered on
top:

```python
dataset.act_curl("v", coord="physical")
dataset.act_divergence("v", coord="physical")
dataset.act_laplacian("S", coord="physical")
dataset.act_symmetric_gradient("v", coord="physical")
dataset.act_antisymmetric_gradient("v", coord="physical")
```

These should reuse the generic gradient/derivative implementation and keep
their tensor contractions explicit and well-tested.

`act_symmetric_gradient()` and `act_antisymmetric_gradient()` are vector-field
only. They split the vector gradient into symmetric and antisymmetric parts over
the vector-component and derivative axes.

`act_laplacian()` is scalar-only for now. Component-wise vector or tensor
Laplacians should not be inferred silently; add an explicit API if that behavior
is needed later.

Component-wise Laplacians are exposed explicitly:

```python
dataset.act_componentwise_laplacian("n", coord="physical")
```

This preserves any trailing component axes and applies the scalar Laplacian to
each component independently.

`act_curl()` is vector-field only. Tensor-valued fields should use the
tensor-specific curl helper so the tensor component axis convention is explicit:

```python
dataset.act_tensor_curl("T", vector_axis=-1, coord="physical")
```

`act_tensor_curl()` applies the vector curl along the selected length-3
component axis and preserves any other trailing component axes.

`act_laplacian()` and `act_componentwise_laplacian()` use direct second
derivative stencils along each lattice axis for index coordinates and for
physical coordinates whose axes align with lattice axes. For non-diagonal
physical transforms, they fall back to repeated physical derivatives so mixed
second-derivative contributions are preserved.

Advanced users can still consume the returned gradient arrays and apply their
own `np.einsum()` expressions, but raw einsum strings should not be the primary
user-facing API.
