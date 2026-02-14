def generate_polar_ring_lattice(
    center,
    normal,
    theta0_axis,
    R_max,
    dr,
    target_arc_length,
    include_center,
):
    """
    Generate a concentric-ring point set on a plane using approximately constant radial spacing
    and approximately constant arc-length spacing along each ring, with golden-angle staggering
    between rings.

    Parameters
    ----------
    center : array_like, shape (3,)
        Center of the disk in 3D world coordinates.
    normal : array_like, shape (3,)
        Plane normal vector. Does not need to be normalized; will be normalized internally.
    theta0_axis : array_like, shape (3,)
        In-plane reference axis defining theta = 0 direction. It does not need to be normalized.
        It must not be parallel to `normal`. It will be projected onto the plane and normalized.
    R_max : float
        Maximum radius of the disk (>= 0).
    dr : float
        Radial spacing between rings (> 0). Rings are placed at r_i = (i + 0.5) * dr.
    target_arc_length : float
        Target arc-length spacing between neighboring points along a ring (> 0).
        For ring radius r, the number of points is N_theta ≈ round(2*pi*r / target_arc_length).
    include_center : bool
        If True, include the center point (r=0) as a single point.

    Returns
    -------
    points_xyz : (N, 3) ndarray
        3D coordinates of generated points.
    ring_index : (N,) ndarray of int
        Ring index for each point. The center point (if included) has ring_index = -1.
        Rings start at index 0 (the smallest nonzero radius ring).
    polar_rt : (N, 2) ndarray
        (r, theta) for each point, where theta is in [0, 2*pi).
        The center point (if included) has r=0 and theta=0.

    Notes
    -----
    - This function uses "ring + equal arc-length" sampling and applies golden-angle staggering:
        theta_{i,k} = 2*pi*k/N_i + i*alpha   (mod 2*pi)
      where alpha is the golden angle alpha = pi*(3 - sqrt(5)).
    - The returned `polar_rt` is defined relative to the in-plane basis constructed from
      `theta0_axis` and `normal`.
    """
    # ---- Validate scalars ----
    R_max = float(R_max)
    dr = float(dr)
    target_arc_length = float(target_arc_length)
    if R_max < 0:
        raise ValueError(f"R_max must be >= 0. Got {R_max}.")
    if dr <= 0:
        raise ValueError(f"dr must be > 0. Got {dr}.")
    if target_arc_length <= 0:
        raise ValueError(f"target_arc_length must be > 0. Got {target_arc_length}.")

    # ---- Build orthonormal basis (e1, e2) on the plane, with e1 aligned to theta0_axis ----
    center = np.asarray(center, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    a = np.asarray(theta0_axis, dtype=float).reshape(3)

    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("normal must be nonzero.")
    n = n / n_norm

    # Project theta0_axis onto the plane: a_perp = a - (a·n) n
    a_proj = a - np.dot(a, n) * n
    a_proj_norm = np.linalg.norm(a_proj)
    if a_proj_norm == 0:
        raise ValueError("theta0_axis must not be parallel to normal (projection is zero).")
    e1 = a_proj / a_proj_norm
    e2 = np.cross(n, e1)  # right-handed: e1 x e2 = n

    # ---- Golden angle for ring staggering ----
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    # ---- Generate rings ----
    points = []
    ring_idx = []
    polar = []

    if include_center:
        points.append(center.copy())
        ring_idx.append(-1)
        polar.append([0.0, 0.0])

    # Place rings at r_i = (i + 0.5)*dr up to R_max
    # Note: if R_max is very small (< dr/2), this generates no rings (only center if requested).
    num_rings = int(np.floor(R_max / dr + 0.5))  # so last ring center radius <= R_max approximately
    # However, we are using r_i = (i+0.5)*dr, so compute until r_i <= R_max.
    i = 0
    while True:
        r = (i + 0.5) * dr
        if r > R_max:
            break

        # Points per ring: approx equal arc length
        n_theta = int(np.round(2.0 * np.pi * r / target_arc_length))
        n_theta = max(1, n_theta)

        phi = (i * golden_angle) % (2.0 * np.pi)

        # Angles on this ring
        thetas = (2.0 * np.pi * np.arange(n_theta) / n_theta + phi) % (2.0 * np.pi)

        # Convert to 3D: center + r*cos(theta)*e1 + r*sin(theta)*e2
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        ring_points = center + (r * cos_t)[:, None] * e1[None, :] + (r * sin_t)[:, None] * e2[None, :]

        points.append(ring_points)
        ring_idx.append(np.full(n_theta, i, dtype=int))
        polar.append(np.column_stack([np.full(n_theta, r), thetas]))

        i += 1

    # ---- Concatenate ----
    if not points:
        # This can happen only if include_center is False and no rings are generated.
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=int), np.empty((0, 2), dtype=float)

    points_xyz = np.vstack([p if p.ndim == 2 else p[None, :] for p in points])
    ring_index = np.concatenate([ri if isinstance(ri, np.ndarray) else np.array([ri], dtype=int) for ri in ring_idx])
    polar_rt = np.vstack([p if isinstance(p, np.ndarray) else np.array([p], dtype=float) for p in polar])

    return points_xyz, ring_index, polar_rt


if __name__ == "__main__":
    # Example usage
    pts, rings, rt = generate_polar_ring_lattice(
        center=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        theta0_axis=[1.0, 0.0, 0.0],
        R_max=5.0,
        dr=0.2,
        target_arc_length=0.2,
        include_center=False,
    )

spheres = PlotSphere(pts, radius=0.05)