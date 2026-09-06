# PlotGlyph lazy-silhouette benchmark

Each condition uses 5 timed repeats; figure construction is excluded.

| object | N | eager median (s) | lazy median (s) | speedup | time saved |
|---|---:|---:|---:|---:|---:|
| points | 500 | 0.023717 | 0.012714 | 1.865x | 46.4% |
| points | 2000 | 0.110372 | 0.033430 | 3.302x | 69.7% |
| points | 5000 | 0.244235 | 0.063698 | 3.834x | 73.9% |
| rods | 500 | 0.022770 | 0.017511 | 1.300x | 23.1% |
| rods | 2000 | 0.052335 | 0.025169 | 2.079x | 51.9% |
| rods | 5000 | 0.100330 | 0.043296 | 2.317x | 56.8% |
| surface | 500 | 0.021051 | 0.013938 | 1.510x | 33.8% |
| surface | 2000 | 0.039374 | 0.026759 | 1.471x | 32.0% |
| surface | 5000 | 0.066460 | 0.053592 | 1.240x | 19.4% |
| combined | 500 | 0.089670 | 0.053293 | 1.683x | 40.6% |
| combined | 2000 | 0.185255 | 0.091836 | 2.017x | 50.4% |
| combined | 5000 | 0.409366 | 0.183326 | 2.233x | 55.2% |
