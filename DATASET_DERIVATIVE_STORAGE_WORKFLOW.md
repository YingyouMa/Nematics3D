# Dataset Derivative Storage Workflow

Temporary design note for future out-of-core and chunked execution support for
large `GridFieldDataset` derivative results. Delete or migrate this file after
the implementation settles.

## Motivation

Spatial derivatives can multiply memory use quickly:

```text
scalar:   (Nx, Ny, Nz)          -> gradient (Nx, Ny, Nz, 3)
vector:   (Nx, Ny, Nz, 3)       -> gradient (Nx, Ny, Nz, 3, 3)
Q5:       (Nx, Ny, Nz, 5)       -> gradient (Nx, Ny, Nz, 5, 3)
Q9:       (Nx, Ny, Nz, 3, 3)    -> gradient (Nx, Ny, Nz, 3, 3, 3)
```

For large fields, keeping every derived result in memory is not practical.

Two future strategies should be considered together:

- disk-backed derivative results that can be read, reused, and deleted
  explicitly;
- chunked execution over smaller spatial blocks with halo/ghost cells.

The current in-memory API should remain the default behavior for small data,
tests, and interactive work.

## Disk-Backed Results

Disk storage should be explicit. Derivative helpers should not silently write to
disk by default.

Possible future APIs:

```python
result = dataset.act_gradient("Q", is_result=True)
handle = dataset.act_save_result(result, path="...")
```

or:

```python
handle = dataset.act_gradient_to_disk("Q", path="...")
```

The result handle should record enough metadata to reload and inspect the data:

- storage kind, for example `"disk"` or `"memmap"`;
- absolute path;
- shape;
- dtype;
- derivative metadata, preferably a `SpatialDerivativeInfo`;
- creation options such as `coord`, `stencil`, `edge_order`, and any chunking
  parameters.

The payload should not be duplicated inside metadata. The field or handle owns
the array storage; metadata only describes provenance and interpretation.

Suggested initial storage backend:

- `.npy` for simple save/load of completed arrays;
- `np.memmap` for chunked writes and partial reads.

Avoid introducing HDF5/Zarr until the simpler file-backed design is not enough.

## Cleanup

Temporary disk results must have explicit cleanup semantics.

Possible API:

```python
handle.act_delete()
```

or dataset-level cleanup:

```python
dataset.act_delete_disk_field("grad_Q")
```

Temporary files should not be removed implicitly while live dataset fields still
refer to them.

## Chunked Execution

Chunked derivative execution should iterate over spatial blocks and write only
the interior of each block to the output.

General pattern:

```python
for block in dataset.iter_spatial_blocks(block_shape, halo=halo_width):
    values_halo = read source block plus halo
    result_halo = compute derivative on halo block
    write result interior to output block
```

The first three axes are spatial axes. Any trailing axes are component axes and
should be carried through unchanged.

## Halo Width

Finite-difference stencils require halo cells from neighboring blocks.

Current first-derivative centered stencils need:

```text
halo_width = 1
```

Composed operators can need a larger effective halo. The current Laplacian uses
direct second-derivative stencils for index coordinates and axis-aligned
physical coordinates, so those paths need only the direct second-derivative
halo. Non-diagonal physical transforms preserve mixed derivative contributions
through composed physical derivatives and should be treated as a multi-pass
operator for chunked execution.

Future operators should declare or compute their required halo width rather than
hard-coding it inside the block iterator.

## Boundary Handling

Periodic and non-periodic boundaries must be handled at the global dataset
boundary, not merely at chunk boundaries.

- Periodic dimensions should wrap halo reads across the global box.
- Non-periodic dimensions should use one-sided stencils only at the true global
  boundary.
- Interior chunk boundaries should always be supplied with halo data, so they
  should not behave like physical boundaries.

## Disk Fields

If disk-backed arrays become first-class dataset fields, the field abstraction
may need to distinguish:

```text
FieldData.raw_values      -> in-memory ndarray
DiskFieldData.raw_path    -> disk path or memmap path
DiskFieldData.raw_shape   -> array shape
DiskFieldData.raw_dtype   -> array dtype
DiskFieldData.raw_info    -> metadata/provenance
```

Alternatively, keep `FieldData` as the in-memory field type and add a separate
disk-result handle until the use case is clearer.

## Suggested Implementation Order

1. Keep current in-memory derivative API unchanged.
2. Design a small disk-result handle that stores path, shape, dtype, and
   payload-free metadata.
3. Add explicit save/load/delete helpers for completed in-memory results.
4. Add a spatial block iterator that yields interior slices and halo slices.
5. Implement one chunked operator first, likely `act_gradient_to_disk()`.
6. Extend chunked execution to divergence, curl, Laplacian, and tensor helpers
   only after gradient chunking is stable.

## Open Questions

- Should disk-backed fields be registered in the same `fields` registry as
  in-memory `FieldData`, or should they live in a separate registry?
- Should chunked results always be disk-backed, or should chunking also support
  in-memory output arrays?
- Should `.npy` be used only for completed arrays and `np.memmap` for active
  chunked writes?
- Should cache paths live under a project-managed cache directory, a user-given
  path, or a temporary directory?
- How should cleanup behave if multiple dataset fields refer to the same disk
  payload?
