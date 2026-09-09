# Uniform trilinear interpolation benchmark

Grid shape: `(72, 68, 64)`. Best of 3 runs.

`map_coordinates(order=1)` is used only as a compiled proxy for a future specialized C kernel; it is not proposed as the production API.

## Backend-only interpolation

| queries | field | RegularGridInterpolator | NumPy trilinear | compiled proxy | proxy speedup vs RGI |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100,000 | scalar | 0.025031 s | 0.012088 s | 0.004408 s | 5.68x |
| 100,000 | Q5 | 0.088468 s | 0.103281 s | 0.044352 s | 1.99x |
| 500,000 | scalar | 0.234201 s | 0.181731 s | 0.042426 s | 5.52x |
| 500,000 | Q5 | 0.650229 s | 0.724134 s | 0.404018 s | 1.61x |

## Q5 value + strict validity

Current-style baseline performs one RGI interpolation for Q5 values and a second RGI interpolation for the float mask. The proxy performs two compiled uniform-grid interpolation passes as well, so it reproduces the current >= 1 - 1e-9 validity semantics exactly. A real fused C kernel could reuse cell indices and weights across both outputs and should have additional headroom beyond this proxy.

| queries | current-style | compiled proxy + direct validity | speedup |
| ---: | ---: | ---: | ---: |
| 100,000 | 0.121925 s | 0.043943 s | 2.77x |
| 500,000 | 0.832713 s | 0.415101 s | 2.01x |

## Full-ish Q5 path with mixed boundaries

Periodic axes: `(True, False, True)`. Timings include copying query points, out-of-domain detection, periodic modulo, non-periodic clipping, and interpolation. Periodic endpoint arrays are constructed outside the timed region, matching GridInterpolator construction-time behavior.

| queries | current no mask | proxy no mask | speedup | current + validity | proxy + validity | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0.091673 s | 0.051860 s | 1.77x | 0.174497 s | 0.099325 s | 1.76x |
| 500,000 | 0.637407 s | 0.450955 s | 1.41x | 0.837074 s | 0.486967 s | 1.72x |

## Interpretation

- If the compiled proxy is materially faster than RGI, a dedicated C kernel has plausible headroom because Nematics3D can additionally fuse coordinate preparation, periodic wrapping, clipping, component interpolation, and validity in one pass.
- Note: the production validity rule is implemented as interpolated mask >= 1 - 1e-9. This is almost, but not exactly, equivalent to requiring all eight support voxels to be valid; an invalid corner with a sufficiently tiny interpolation weight can still pass. The first exploratory run found 1 such case among 500,000 random points.
- If the direct NumPy trilinear path is slower, that does not argue against C; it mostly shows that eight fancy-index gathers and temporary arrays are expensive in NumPy.
- This benchmark deliberately uses in-domain index-space points. Physical-coordinate transforms and periodic wrapping should be benchmarked separately before production work.
