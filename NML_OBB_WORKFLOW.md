# NML OBB Workflow Notes

This note records the planned workflow for two related pieces of geometry and
Q-field analysis:

1. building an approximate minimum oriented bounding box from a point cloud;
2. using that box as the geometric seed for an iterative local N/M/L analysis.

The current goal is design alignment, not final API commitment.

## Terminology

- **points**: an input point cloud, usually defect-loop coordinates.
- **OBB**: oriented bounding box.
- **seed bounds**: the initial approximate minimum OBB built from the input
  points.
- **minimal bounds in axes**: the smallest box, in a specified orthonormal
  axes frame, that wraps a set of points.
- **expanded bounds**: a bounds object after applying expansion factors and
  minimum side-length floors.
- **N/M/L axes**: axes obtained by diagonalizing the local mean Q tensor, with
  eigenvalues ordered descending.

## Approximate Minimum OBB Workflow

Given a set of input points, find a good approximate minimum-volume oriented
box. This step is intentionally approximate rather than an exact 3D minimum OBB
solver.

### 1. Convex Hull Reduction

Proposed function:

```python
compute_convex_hull_points(points)
```

Input:

- raw point cloud, shape `(N, 3)`.

Output:

- convex hull vertices, shape `(M, 3)`.

Reasoning:

- The minimum enclosing box is determined by the convex hull.
- Interior points do not affect the box.
- Reducing to hull vertices makes later projection and optimization cheaper.

Open handling detail:

- If `ConvexHull` fails for degenerate point sets, such as nearly coplanar,
  collinear, or very small point clouds, decide whether to fall back to the
  original points or to a lower-dimensional special case.

### 2. PCA Initial OBB

Proposed function:

```python
fit_obb_pca(hull_points)
```

Input:

- convex hull vertices.

Output:

- an OBB fit result containing:
  - axes / rotation;
  - center;
  - side lengths;
  - local min/max coordinates;
  - volume.

Reasoning:

- PCA gives a deterministic, fast, interpretable initial frame.
- For loop-like objects, the PCA axes often provide sensible main-extension,
  secondary-extension, and thin directions.
- PCA is not guaranteed to minimize box volume, so it is only an initial guess.

### 3. Random-Search Refinement

Proposed function:

```python
refine_obb_random_search(hull_points, initial_fit, ...)
```

Input:

- convex hull vertices;
- PCA OBB fit;
- random-search controls such as seed, trial count, and angle scales.

Output:

- refined OBB fit result with updated axes, center, lengths, local min/max, and
  volume.

Reasoning:

- For any candidate axes frame, the enclosing box is easy to compute:
  project points into that frame, take per-axis min/max, and multiply extents.
- Random perturbations around the PCA frame can improve the volume without
  implementing an exact minimum-volume OBB algorithm.
- A multi-scale search is preferred: start with larger angular perturbations,
  keep improvements, then reduce the perturbation scale.

Important properties:

- This is an approximate stochastic optimizer, not an exact solver.
- It should support reproducibility through an explicit random seed.
- Degenerate thin or planar point sets may produce very small side lengths; the
  later sampling bounds should still apply minimum length floors.

### 4. Combined Fit Entry Point

Proposed function:

```python
fit_obb_approx(points, method="pca_random", ...)
```

Workflow:

```text
points
  -> compute_convex_hull_points
  -> fit_obb_pca
  -> refine_obb_random_search
  -> OBB fit result
```

This function should return a pure geometry fit result, not a `Bounds` object.

## Bounds Integration

The pure OBB fitting functions should be independent of the repository
`Bounds` class. Conversion to repository objects can live beside `Bounds`.

Likely placement:

- `src/nematics3d/geometry.py`
  - `compute_convex_hull_points`
  - `fit_obb_pca`
  - `refine_obb_random_search`
  - `fit_obb_approx`
  - the lightweight OBB fit dataclass
- `src/nematics3d/classes/bounds.py`
  - `bounds_from_obb_fit`
  - `minimal_bounds_wrapping_points`
  - `expanded_bounds`

Possible object conversion:

```python
bounds = bounds_from_obb_fit(fit, name="seed bounds")
```

The resulting `Bounds` should use:

- `alignment="center"`;
- `axis1 = fit.axes[:, 0]`;
- `axis2 = fit.axes[:, 1]`;
- `length1/2/3 = fit.lengths`.

## Minimal Bounds In Given Axes

This is a separate operation from approximate minimum OBB fitting.

Given points and a specified axes frame, build the smallest box in that frame
that wraps the points.

Proposed function:

```python
minimal_bounds_wrapping_points(points, axes, origin=None, ...)
```

Input:

- points to wrap;
- an orthonormal axes frame;
- optional reference origin for local projection.

Output:

- a `Bounds` object or lower-level fit result representing the smallest
  axes-aligned box in the supplied frame.

Workflow:

```text
local = (points - origin) @ axes
local_min = local.min(axis=0)
local_max = local.max(axis=0)
local_center = 0.5 * (local_min + local_max)
lengths = local_max - local_min
world_center = origin + axes @ local_center
```

Important distinction:

- This is minimal only for the supplied axes.
- It does not search over all possible rotations.

## Bounds Expansion Workflow

Given a minimal bounds object, build an expanded box for sampling.

Proposed function:

```python
expanded_bounds(bounds, expand_factors, min_lengths=None, ...)
```

Input:

- a base bounds object;
- per-axis expansion factors;
- optional per-axis minimum side lengths.

Output:

- a new expanded `Bounds` object.

Rule:

```text
expanded_lengths = max(base_lengths * expand_factors, min_lengths)
```

The center and axes should remain unchanged unless a later use case explicitly
requires asymmetric expansion.

## NML Self-Consistent Bounds Workflow

The revised NML analysis should use the approximate minimum OBB as a geometric
seed. This differs from the original trial script, which rebuilt each iteration
from the loop points directly.

### 1. Build Seed Bounds

Input:

- original loop coordinates.

Workflow:

```text
loop_points
  -> fit_obb_approx
  -> bounds_from_obb_fit
  -> seed_bounds
```

Purpose:

- Convert the loop point cloud into a stable geometric proxy.
- Later iterations must preserve this seed geometry.

### 2. Use Seed Corners As Required Geometry

For the NML iteration, use:

```python
required_points = seed_bounds.corners
```

instead of:

```python
required_points = loop_points
```

Meaning:

- Each iteration constructs a box that contains the initial seed OBB.
- The loop is no longer re-wrapped directly point by point at every iteration.
- The seed bounds becomes the geometric object that must stay enclosed.

### 3. Iterative NML Analysis

Inputs:

- an existing `QFieldObject`;
- seed bounds or seed bounds corners;
- initial axes, defaulting to lab-frame `np.eye(3)`;
- expansion factors;
- minimum sampling box lengths;
- sampling spacing;
- angle tolerance;
- maximum iteration count.

Per-iteration workflow:

```text
current axes
  -> minimal bounds wrapping seed_bounds.corners in current axes
  -> expanded bounds
  -> sample points inside expanded bounds
  -> interpolate Q at sample points
  -> diagonalize Q to recover local directors
  -> rebuild S=1 Q tensors from directors
  -> average those Q tensors
  -> diagonalize mean Q
  -> align eigenvector signs to previous axes
  -> update N/M/L axes
  -> stop if max axis-angle change < tolerance
```

The iteration should save both:

- the minimal bounds for the current axes;
- the expanded bounds actually used for Q sampling.

This makes debugging and reporting clearer than storing only lengths and bbox
numbers.

## Difference From The Initial Trial Script

Original trial script:

```text
current axes
  -> wrap loop points directly
  -> expand/min-floor box
  -> sample Q
  -> update axes
```

Revised workflow:

```text
loop points
  -> approximate minimum OBB seed bounds

current axes
  -> wrap seed bounds corners
  -> expand/min-floor box
  -> sample Q
  -> update axes
```

Conceptual change:

- The initial loop is first converted into a stable oriented box.
- The NML iteration then operates on that box as the required enclosed
  geometry.
- This can make the geometry constraint more stable, but it can also enlarge
  the sampled region if the seed OBB is thicker than the raw loop point cloud.

## Open Questions

- Should `fit_obb_approx` expose the random-search refinement directly, or
  should the random search remain private until benchmarked?
- How should degenerate convex hull cases be handled?
- Should the OBB objective be volume only, or should we later allow alternate
  objectives for thin loop-like objects?
- Should `minimal_bounds_wrapping_points` return a pure fit object first, with
  a separate conversion to `Bounds`, or directly return `Bounds`?
- Where should the NML iteration live: a standalone analysis module, or a thin
  `QFieldObject.act_...` method wrapping a standalone function?
