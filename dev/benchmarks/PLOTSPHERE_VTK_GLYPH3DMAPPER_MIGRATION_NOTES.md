# PlotSphere `vtkGlyph3DMapper` migration notes

## Purpose

This document records the current performance-migration idea for `PlotSphere`, the
benchmarks already completed, the compatibility tests already completed, and the
remaining questions that should be reviewed before production code is changed.

The current production implementation is intentionally unchanged at this stage.
All work so far is isolated under `dev/benchmarks/`.

---

## Current production pipeline

`PlotSphere` currently builds a unit sphere and then materializes one complete
polygonal sphere for every input point:

```python
unit_sphere = pv.Sphere(
    theta_resolution=self.opts.sides,
    phi_resolution=self.opts.sides,
    radius=1.0,
)
mesh = poly.glyph(geom=unit_sphere, scale="radius", orient=False)
```

Conceptually:

```text
N centers + N radii
    -> duplicate unit-sphere geometry N times
    -> construct one large PolyData
    -> give that full mesh to the mapper
```

This has two important consequences:

1. construction/update cost grows with the amount of fully materialized geometry;
2. `actor.mapper.dataset` is the final polygonal geometry for all spheres.

The second point is important because some existing functionality implicitly relies
on the rendered geometry being materialized on the CPU.

---

## Proposed fast backend

The proposed backend uses `vtkGlyph3DMapper` directly while retaining PyVista for
the surrounding figure/scene-management layer.

Conceptually:

```text
                    PlotFigure
                     PyVista
                        |
              vtkGlyph3DMapper
                        |
          +-------------+-------------+
          |                           |
 one unit-sphere source      N instance records
                             - coords
                             - radius
                             - RGBA/scalar
```

The renderer receives one source topology and one point-data table rather than a
fully duplicated sphere mesh.

Per-instance attributes can still vary. The prototype has already demonstrated
independent per-sphere:

- position;
- radius;
- RGB color;
- opacity.

The intended production architecture is therefore not a full PyVista-to-VTK
rewrite. PyVista remains responsible for `PlotFigure`, renderer management,
camera, screenshots, widgets, and general scene integration. Raw VTK is used only
for the glyph mapper backend where the pipeline change provides a material
performance benefit.

---

## Benchmark prototype

Benchmark file:

```text
dev/benchmarks/plot_sphere_vtk_glyph_mapper_benchmark.py
```

Results:

```text
dev/benchmarks/results/plot_sphere_vtk_glyph_mapper_benchmark.md
dev/benchmarks/results/plot_sphere_vtk_glyph_mapper_benchmark.json
```

The benchmark compares the current real `PlotSphere` implementation against an
independent raw-VTK `VtkGlyphSpherePrototype`. Production code is not monkeypatched
or replaced.

### Test data

For each sphere count, deterministic random arrays are generated for:

```text
coords[N, 3]
radius[N]
rgba[N, 4]
```

Independent update arrays are also generated for coordinates, radii, and colors.

The benchmark sizes are:

```text
500
2,000
5,000
10,000
50,000
```

with five interleaved repeats per implementation.

### Metrics

The benchmark records:

- initial construction;
- radius update;
- color/opacity update;
- coordinate update;
- screenshot time.

The screenshot metric is especially important because a plain `render()` call may
not force all framebuffer work to complete. Screenshot timing is therefore the
better approximation to complete rendered-frame cost in this off-screen test.

### Current benchmark result

| N | metric | current (s) | vtk mapper (s) | speedup | time saved |
|---:|---|---:|---:|---:|---:|
| 500 | construct | 0.013637 | 0.000455 | 30.00x | 96.7% |
| 500 | radius update | 0.008141 | 0.000014 | 569.32x | 99.8% |
| 500 | color update | 0.008191 | 0.000008 | 1011.21x | 99.9% |
| 500 | coords update | 0.007026 | 0.000004 | 1633.86x | 99.9% |
| 500 | screenshot | 0.095725 | 0.097128 | 0.99x | -1.5% |
| 2,000 | construct | 0.029208 | 0.000382 | 76.38x | 98.7% |
| 2,000 | radius update | 0.019109 | 0.000013 | 1436.75x | 99.9% |
| 2,000 | color update | 0.018886 | 0.000006 | 3372.48x | 100.0% |
| 2,000 | coords update | 0.019047 | 0.000006 | 3023.38x | 100.0% |
| 2,000 | screenshot | 0.105419 | 0.105409 | 1.00x | 0.0% |
| 5,000 | construct | 0.056707 | 0.000415 | 136.68x | 99.3% |
| 5,000 | radius update | 0.041788 | 0.000016 | 2548.07x | 100.0% |
| 5,000 | color update | 0.042565 | 0.000008 | 5067.25x | 100.0% |
| 5,000 | coords update | 0.041125 | 0.000011 | 3772.94x | 100.0% |
| 5,000 | screenshot | 0.122644 | 0.109120 | 1.12x | 11.0% |
| 10,000 | construct | 0.098212 | 0.000457 | 215.00x | 99.5% |
| 10,000 | radius update | 0.073192 | 0.000020 | 3734.29x | 100.0% |
| 10,000 | color update | 0.071928 | 0.000010 | 7121.54x | 100.0% |
| 10,000 | coords update | 0.071457 | 0.000024 | 3002.41x | 100.0% |
| 10,000 | screenshot | 0.137256 | 0.098972 | 1.39x | 27.9% |
| 50,000 | construct | 0.429366 | 0.000536 | 801.06x | 99.9% |
| 50,000 | radius update | 0.338399 | 0.000049 | 6948.64x | 100.0% |
| 50,000 | color update | 0.354007 | 0.000037 | 9619.76x | 100.0% |
| 50,000 | coords update | 0.365756 | 0.000088 | 4156.32x | 100.0% |
| 50,000 | screenshot | 0.335884 | 0.153190 | 2.19x | 54.4% |

### Interpretation

The huge construction/update speedups should not be interpreted as equivalent
frame-rate speedups. They mainly show that materializing and rebuilding duplicated
polygon meshes is expensive.

The screenshot result is the more conservative rendering result:

```text
500 spheres      approximately equal
2,000 spheres    approximately equal
5,000 spheres    ~1.12x faster
10,000 spheres   ~1.39x faster
50,000 spheres   ~2.19x faster
```

The benefit increases strongly with glyph count. The benchmark currently uses
`sides=8`; higher sphere resolution is expected to penalize the materialized-mesh
pipeline more strongly because duplicated source geometry becomes larger.

The extremely small mapper-side update timings should be interpreted as CPU-side
attribute-update cost. Complete next-frame rendering still has GPU/framebuffer cost,
which is better represented by the screenshot metric.

---

## Compatibility prototype

Compatibility test file:

```text
dev/benchmarks/test_plot_sphere_vtk_glyph_mapper_compatibility.py
```

Current result:

```text
6 passed in 4.06s
Black passed
Ruff passed
```

The compatibility prototype verifies that a `vtkGlyph3DMapper` backend can support
the main instance-level rendering features needed by `PlotSphere`.

### Confirmed compatible

The following have been experimentally verified rather than only inferred from VTK
documentation:

- independent per-sphere radius;
- independent per-sphere RGBA, including alpha;
- scalar-colormap mapping;
- scalar-bar-compatible lookup-table use;
- center clipping with synchronized indexing of coordinates, radii, colors, and
  scalars;
- standard actor lighting properties;
- PBR/metallic/roughness properties;
- a replacement highlight strategy based on a dedicated actor for a selected
  instance.

---

## Important semantic change

With the current implementation:

```text
actor.mapper.dataset
```

is the fully materialized sphere geometry.

With `vtkGlyph3DMapper`, the mapper input is instead the instance table:

```text
N input points
radius/color/scalar arrays
```

The unit-sphere polygon geometry exists separately as the mapper source. The final
rendered collection of N sphere surfaces is not automatically materialized as one
CPU-side `PolyData`.

The compatibility test explicitly confirms this distinction.

This semantic change is the source of the remaining compatibility issues below.

---

## Existing feature: mesh clipping

This is a real Nematics3D feature and is the most important compatibility problem.

Center clipping is straightforward with instancing:

```text
test each center
    -> keep selected instances
    -> slice coords/radius/color/scalar arrays with the same indices
```

Mesh clipping is different. Its intended semantics require the fully generated
sphere surface to intersect the clipping geometry. A sphere can therefore be kept
partially and physically cut at the clipping surface.

That operation depends on real polygonal sphere geometry and cannot be reproduced
by simply filtering instance centers.

### Proposed handling

Do not weaken mesh-clipping semantics.

Recommended design:

```text
ordinary PlotSphere rendering
    -> vtkGlyph3DMapper backend

center clipping
    -> vtkGlyph3DMapper backend

mesh clipping requested
    -> materialize / fall back to current glyph-mesh pipeline
```

This keeps the fast path fast while preserving the existing feature exactly when it
is explicitly requested.

Whether the fallback should be automatic and reversible requires further design
review before implementation.

---

## Existing feature: silhouette / highlight

The current silhouette implementation relies on the final materialized mesh, for
example by accessing the mapper dataset and extracting/triangulating its surface.

That implementation cannot be carried over unchanged because the mapper dataset in
the new backend is only the instance input.

However, this does not mean the user-visible highlight feature needs to be removed.

The compatibility prototype demonstrates a simple replacement strategy:

```text
selected instance index
    -> selected center + radius
    -> draw an independent slightly enlarged outline/wireframe sphere actor
```

This is arguably better aligned with glyph semantics because highlighting one
instance does not require constructing silhouettes for every sphere.

Therefore:

```text
silhouette/highlight feature: retain
current silhouette implementation: replace
```

This should be reviewed separately from mesh clipping because it does not require a
fallback to the slow materialized backend.

---

## Full final mesh / mesh filters / export

This item is mainly a semantic and architectural concern. It has not yet been
established that Nematics3D exposes a public `PlotSphere` feature that depends on
arbitrary processing of the final combined sphere mesh.

With the current backend, advanced code can naturally obtain the final mesh and run
operations such as:

```text
clean
triangulate
extract_surface
connectivity
boolean/filter operations
save/export the combined polygon mesh
```

With `vtkGlyph3DMapper`, this final mesh does not exist unless explicitly generated.

If current Nematics3D code does not rely on this behavior, it is not necessarily a
user-facing regression. It should nevertheless be treated as an intentional
semantic change.

If a future feature requires the final geometry, the recommended pattern is lazy
materialization:

```text
rendering
    -> stay instanced

explicit geometry operation/export
    -> materialize only on demand
```

---

## Items not yet fully verified

### Picking

Picking has not yet been exercised end-to-end with the new mapper.

The likely user-level requirement is to recover the corresponding original input
point/instance, not the triangle ID on the rendered sphere surface. That should be
compatible with glyph selection or a nearest-instance strategy, but it must still be
tested against the current Nematics3D picking contract before production migration.

### Translucent overlap parity

Per-instance alpha rendering works in the prototype. What has not yet been tested is
strict visual parity between the current materialized-mesh backend and the instanced
backend when many translucent spheres overlap.

This needs side-by-side or image-based validation because translucent depth sorting
can depend on renderer/pipeline details.

### Full production API contract

The current compatibility prototype verifies individual backend capabilities, but a
production migration should ultimately run the existing `PlotSphere` / `PlotGlyph`
black-box behavior tests against the new backend.

The relevant question is not only whether VTK can render the feature, but whether
existing Nematics3D calls preserve the same:

- opts/state behavior;
- resolver behavior;
- actor lifecycle;
- clipping semantics;
- scalar-bar lifecycle;
- interaction behavior;
- empty-data behavior;
- live-update behavior.

---

## Recommended migration design

The current working hypothesis is:

```text
PlotSphere logical data / HostBase / OptsBase
                 |
                 v
        choose rendering backend
                 |
        +--------+---------+
        |                  |
        v                  v
vtkGlyph3DMapper     materialized glyph mesh
 default fast path   compatibility fallback
        |                  |
 ordinary render      mesh clipping and
 center clipping      operations requiring
 live attributes      real final geometry
```

Important design rule:

> Do not bypass Nematics3D's logical data model merely to gain rendering speed.

`raw_*`, `calc_*`, option resolution, clipping indices, and user-facing state should
remain authoritative. Only the final rendering representation should change.

This is therefore best treated as a rendering-backend replacement rather than a
rewrite of `PlotSphere` itself.

---

## Proposed next review steps

Before production modification, review the following in order:

1. inspect exactly how current `mesh clipping` enters the generic `PlotGlyph`
   pipeline and determine the cleanest fallback boundary;
2. inspect current picking contract and test instance selection with
   `vtkGlyph3DMapper`;
3. produce controlled current-vs-instanced screenshots for opaque and overlapping
   translucent spheres;
4. identify every internal use of `entity_actor.mapper.dataset` that assumes final
   rendered geometry;
5. decide whether the backend switch belongs in `PlotSphere` only or should be
   abstracted so `PlotRod` can use the same instancing infrastructure later;
6. only after those points are understood, build a production prototype and run the
   existing visual black-box tests against both backends.

`PlotRod` is likely the next major beneficiary because its current `poly.tube(...)`
pipeline also materializes repeated geometry, but sphere migration should be
understood first before generalizing the backend architecture.

---

## Current status

At the time this note was written:

- production `PlotSphere` code has not been modified for `vtkGlyph3DMapper`;
- the performance prototype exists under `dev/benchmarks/`;
- performance results exist under `dev/benchmarks/results/`;
- the compatibility prototype exists under `dev/benchmarks/`;
- compatibility tests pass;
- the migration is not yet committed as a production change;
- mesh clipping remains the clearest real feature requiring a compatibility
  fallback;
- silhouette/highlight appears retainable through a different implementation;
- picking and translucent-overlap parity remain to be tested before production
  adoption.

