# Surface Director Projection Plan

## Objective

Add a focused geometry-analysis function that projects directors onto the local
tangent planes of a reconstructed surface and reports how strongly each input
director deviates from the reconstructed surface.

This design is independent of `QSurface`. The initial workflow assumes that:

- `coords` has shape `(N, 3)` and contains sampled surface positions;
- `directors` has shape `(N, 3)` and `directors[i]` belongs to `coords[i]`;
- `triangulate_surface_points(coords)` returns a triangular `pyvista.PolyData`
  whose vertex order is identical to the input point order.

The word *plane* below means the local tangent plane at each surface vertex.
For a curved surface there is no single shared projection plane.

## Public API Split

Use two public functions because directors defined at the surface sampling
vertices and directors queried independently of the surface point cloud require
different normal-resolution algorithms. Avoid one function whose behavior
changes substantially when an optional `surface_points` argument is supplied.

### Directors defined at the surface sampling points

Implement this path first:

```python
def project_surface_point_directors(
    surface_points,
    directors,
    *,
    max_tilt_degrees=None,
    tangent_tolerance=1e-10,
) -> SurfaceDirectorProjectionResult:
    ...
```

This function requires:

```python
surface_points.shape == directors.shape == (N, 3)
surface_points[i] <-> directors[i]
```

It calls `triangulate_surface_points(surface_points)`. Because the reconstructed
mesh preserves the input vertex order, point normal `i` is used directly for
director `i`. This path requires neither nearest-neighbor matching nor normal
interpolation and is the preferred initial implementation.

### Directors independent of the surface sampling points

Add this separately after the vertex-aligned path is stable:

```python
def project_directors_to_point_cloud_surface(
    director_coords,
    directors,
    surface_points,
    *,
    point_match_tolerance=None,
    max_tilt_degrees=None,
    tangent_tolerance=1e-10,
) -> PointCloudSurfaceDirectorProjectionResult:
    ...
```

This function permits:

```python
director_coords.shape == directors.shape == (N, 3)
surface_points.shape == (M, 3)
```

Resolve the normal for each director coordinate as follows:

1. triangulate `surface_points` and compute smooth vertex normals;
2. detect director coordinates that coincide with surface vertices within
   `point_match_tolerance` and use those vertex normals directly;
3. for every remaining coordinate, find its closest location and containing
   triangle on the reconstructed surface;
4. compute barycentric coordinates at that surface location;
5. interpolate the triangle's three vertex normals and normalize the result.

Do not use an arbitrarily selected incident face normal when a director lies on
a mesh vertex, and do not use only the nearest vertex normal for an interior
query. Both choices make the resolved normal change discontinuously as the
query moves across face or nearest-vertex boundaries.

The independent-coordinate result should additionally report:

- closest locations on the reconstructed surface;
- distances from director coordinates to those locations;
- matched surface-vertex indices, using a documented sentinel for coordinates
  resolved through triangle interpolation.

These diagnostics distinguish director tilt from the separate error that a
director coordinate may not lie exactly on the reconstructed surface.

## Shared Projection Core

The two public functions should share one private numerical core rather than
duplicate projection, validation, threshold, and warning logic:

```python
def _project_directors_with_normals(
    directors,
    surface_normals,
    *,
    max_tilt_degrees,
    tangent_tolerance,
) -> SurfaceDirectorProjectionResult:
    ...
```

The public functions differ only in how they obtain one local surface normal
for each director. The shared core performs tangent projection, computes all
error measures, collects threshold violations, emits the consolidated warning,
and constructs the common result.

## Result Type

Return a frozen, slotted dataclass derived from `ResultBase`:

```python
@dataclass(slots=True, frozen=True, repr=False)
class SurfaceDirectorProjectionResult(ResultBase):
    projected_directors: np.ndarray
    surface_normals: np.ndarray
    tilt_angles_degrees: np.ndarray
    normal_fractions: np.ndarray
    tangent_fractions: np.ndarray
    is_projectable: np.ndarray
    exceeded_indices: np.ndarray
    max_tilt_degrees: float | None
```

The class should define `__result_name__` and `__field_docs__` following the
existing `ResultBase` subclasses.

Field meanings:

- `projected_directors`: unit directors projected onto the local tangent
  planes, with shape `(N, 3)`;
- `surface_normals`: unit vertex normals used for the projection;
- `tilt_angles_degrees`: unsigned angles away from the local tangent planes;
- `normal_fractions`: absolute normalized director components along the local
  surface normals;
- `tangent_fractions`: fractions of director magnitude retained by tangent
  projection;
- `is_projectable`: whether a director is nonzero and has a nonzero tangent
  projection within the configured numerical tolerance;
- `exceeded_indices`: all indices whose tilt is strictly greater than
  `max_tilt_degrees`;
- `max_tilt_degrees`: the configured threshold, or `None` when threshold
  checking is disabled.

The independent-coordinate function may wrap this common result:

```python
@dataclass(slots=True, frozen=True, repr=False)
class PointCloudSurfaceDirectorProjectionResult(ResultBase):
    projection: SurfaceDirectorProjectionResult
    surface_locations: np.ndarray
    surface_distances: np.ndarray
    matched_vertex_indices: np.ndarray
```

This keeps the projection semantics identical between both modes while making
the point-cloud query diagnostics explicit.

## Geometry and Error Definition

For an input director `d` and unit local normal `N`, first normalize the
director:

```text
d_hat = d / ||d||
```

The signed normal coefficient and tangent projection are:

```text
c           = dot(d_hat, N)
d_tangent   = d_hat - c N
```

The unsigned normal and tangent fractions are:

```text
normal_fraction  = abs(c)
tangent_fraction = ||d_tangent||
```

The angle away from the tangent plane is:

```text
tilt_degrees = degrees(asin(clip(normal_fraction, 0, 1)))
```

This definition respects nematic symmetry because both `d` and `-d` produce
the same unsigned tilt. The signed coefficient must still be used when
subtracting the normal component.

Normalize `d_tangent` only when its norm exceeds `tangent_tolerance`. A zero
input director or a director parallel to the surface normal is not
projectable. The initial implementation should represent invalid projected
directions consistently and document that representation.

## Maximum-Tilt Warning

`max_tilt_degrees` should accept `None` or a finite number in `[0, 90]`.
Invalid thresholds should raise `ValueError`.

When the threshold is enabled, collect all violations with:

```python
exceeded_indices = np.flatnonzero(
    tilt_angles_degrees > max_tilt_degrees
)
```

If any points exceed the threshold, emit one consolidated warning rather than
one warning per point. The complete index array must remain available in the
result. To prevent extremely large log messages, the warning may display only
the first fixed number of indices and report the total count.

Because the function genuinely uses the injected logger for this warning, the
repository logging decorator is appropriate. The message should state that
the measured tilt is relative to the *reconstructed* surface.

## Surface Normals

The initial implementation can obtain smooth vertex normals from the clean,
triangulated surface through PyVista:

```python
surface_with_normals = surface.compute_normals(
    cell_normals=False,
    point_normals=True,
    split_vertices=False,
    consistent_normals=True,
    auto_orient_normals=False,
    inplace=False,
)
```

The normal sign does not affect tangent projection or unsigned tilt because
the projection operator `I - N N^T` is invariant under `N -> -N`.

## Interpretation and Noise Limitation

The result measures director tilt relative to the tangent planes of the
reconstructed mesh, not directly relative to an unknown exact surface.
Observed tilt contains both:

1. physical or numerical director deviation from the true surface; and
2. error in the reconstructed local surface normal.

Dense sampling alone is insufficient. Reliable normals also require reasonably
uniform sampling, low positional noise, correct triangulation connectivity,
and triangles that are not extremely thin. High curvature relative to the
point spacing also increases normal error.

Returning the actual `surface_normals` is therefore important for diagnosis.
If mesh-derived point normals prove too noisy, a later extension can add a
local-PCA normal estimator with an explicit neighborhood size or radius. That
extension should not be included until real data demonstrates the need,
because the neighborhood scale introduces its own curvature-versus-noise
tradeoff.

## Validation

Focused tests should cover:

- directors exactly tangent to a plane produce zero tilt;
- directors normal to a plane produce 90-degree tilt and are not projectable;
- changing a director from `d` to `-d` does not change its error;
- projected directors are tangent and unit length when projectable;
- a sphere with analytically constructed tangent directors gives small tilt;
- threshold comparison is strict (`>` rather than `>=`);
- one consolidated warning contains offending indices;
- `exceeded_indices` retains all violations even if warning display is
  truncated;
- invalid director shapes, nonfinite values, mismatched point counts, and
  invalid thresholds fail clearly.

## Future Relationship to Surface Streamlines

The projected unit directors and validity mask can later feed a separate
surface-streamline integrator. This projection function should not contain
streamline seeding, interpolation, or integration logic.
