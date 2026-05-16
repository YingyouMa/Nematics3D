# Contour Surface Design Notes

This note records the current design direction for 3D contour-surface support.
It is intentionally a staging document, not an implementation spec.

## Scope

Current scope:

- input a 3D scalar array with shape `(Nx, Ny, Nz)`
- input one or more contour levels
- organize the resulting contour surfaces as repo objects
- support plotting as a downstream capability

Out of scope for now:

- detailed implementation
- exact VTK/PyVista extraction path
- console panel layout
- smoothing and advanced post-processing

## Layering

We currently want three layers:

1. `ContourSurfaceSet`
2. `ContourSurface`
3. `PlotContourSurface`

The intended split is:

- `ContourSurfaceSet`: analysis/data host for a family of contour surfaces
- `ContourSurface`: one concrete contour surface at one level
- `PlotContourSurface`: one visual object for one `ContourSurface`

Plotting is not the identity of the top-level object. It is only one of its
main actions.

## Object Roles

### `ContourSurfaceSet`

Role:

- main user-facing entry for a scalar field and a list of contour levels
- owns the raw scalar array and shared grid mapping metadata
- manages creation, refresh, and removal of per-level `ContourSurface` objects
- may create plot objects, but is not itself a plot object

Expected relations:

- contains or registers many `ContourSurface` objects
- may expose a registry-like interface instead of only a plain list

Tentative input parameters:

```python
ContourSurfaceSet(
    values,
    levels,
    name="contour-surface-set",
    grid_offset=None,
    grid_transform=None,
    bounds=None,
    opts_defaults_override=None,
    visual_default=None,
    figure=None,
    is_plot=False,
)
```

Parameter intent:

- `values`: 3D scalar array, shape `(Nx, Ny, Nz)`
- `levels`: one float or a sequence of floats
- `name`: object name
- `grid_offset`: offset of the grid in physical space
- `grid_transform`: linear map from index space to physical space
- `bounds`: optional shared bounds or clipping context
- `opts_defaults_override`: repo-style defaults override hook
- `visual_default`: default visual options to use when plots are created
- `figure`: optional plotting target if auto-plotting is requested
- `is_plot`: whether to create visuals immediately on initialization

Tentative outputs / exposed results:

- a set of `ContourSurface` child objects
- shared raw data and shared geometry metadata
- actions like add/remove/refresh/plot for the full family

### `ContourSurface`

Role:

- represents one contour surface at one level
- bridges the scalar-field host layer and the visual layer
- may later hold per-surface geometry or analysis quantities

This layer is not expected to be strongly user-facing.

Expected relations:

- belongs to one `ContourSurfaceSet`
- may own or register one or more `PlotContourSurface` visuals

Tentative input parameters:

```python
ContourSurface(
    level,
    owner,
    name=None,
)
```

Parameter intent:

- `level`: the contour value for this surface
- `owner`: the parent `ContourSurfaceSet`
- `name`: optional object name

Tentative outputs / exposed results:

- one contour-surface object bound to one level
- cached mesh/polydata once extraction is performed
- downstream plot creation for this single surface

### `PlotContourSurface`

Role:

- visual representation of one `ContourSurface`
- owns figure binding, actor state, opts, and interactive visual behavior
- should not own a separate copy of the full raw 3D scalar array

Expected relations:

- belongs to one `ContourSurface`
- may inherit geometry context from the parent surface/set

Tentative input parameters:

```python
PlotContourSurface(
    surface,
    figure=None,
    opts=None,
    bounds=None,
    name=None,
    opts_defaults_override=None,
    **kwargs,
)
```

Parameter intent:

- `surface`: the source `ContourSurface`
- `figure`: optional target figure/plotter
- `opts`: visual options object
- `bounds`: optional visual clipping override or inherited bounds context
- `name`: optional visual name
- `opts_defaults_override`: repo-style defaults override hook
- `**kwargs`: direct visual-option overrides merged into `opts`

Tentative outputs / exposed results:

- one plotted contour-surface actor or visual object
- editable visual opts
- standard visual actions such as commit/remove/refresh

## Data Ownership

Current direction:

- only `ContourSurfaceSet` should own the raw `(Nx, Ny, Nz)` scalar array
- `ContourSurface` should refer back to its owner for source data
- `PlotContourSurface` should refer to `ContourSurface`, not to the raw array

This keeps the layers separated and avoids multiple copies of the field data.

## Naming Summary

Current chosen names:

- `ContourSurfaceSet`
- `ContourSurface`
- `PlotContourSurface`

Reasoning:

- the top layer is a non-plot host object
- the middle layer is one per-level surface object
- the bottom layer is purely visual

## Current Open Questions

- whether `ContourSurfaceSet` should auto-create visuals by default
- whether one `ContourSurface` should support multiple visuals on different figures
- how the console should split controls among set/surface/plot layers
- what the final opts class for `PlotContourSurface` should contain
- how bounds/clipping should be shared between analysis and visual layers
- exact extraction backend details for contour mesh generation

## Status

This document is a checkpoint for the current design discussion only.
Implementation details should be decided incrementally.
