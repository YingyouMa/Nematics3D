# Uniform-grid trilinear interpolation benchmark

This experiment estimates whether Nematics3D could benefit from a specialized
compiled trilinear interpolation backend instead of relying exclusively on
`scipy.interpolate.RegularGridInterpolator`.

The benchmark intentionally does **not** modify production code.  It compares
the current generic SciPy backend with a direct NumPy implementation and with
`scipy.ndimage.map_coordinates(order=1)` as a compiled proxy for a future
uniform-grid C kernel.  It also tests the important Q5 + strict-validity case,
where a custom backend could reuse the same cell lookup for field values and
mask validity.

Run the pytest file in this directory.  It writes the measured timings and a
short interpretation to `RESULTS.md`.
