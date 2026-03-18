## Possible Future Improvements

- Measure the runtime overhead of the logging decorator, especially in hot paths and repeated small function calls.
- Explore a repository-wide switch that can disable the logging decorator globally when performance is the priority.
- Rebuild `PlaneGrid.act_debug_plot()` around the new `bounds` relation and `Bounds.act_visualize()` instead of the legacy `PlotExtent` path.

- Decide whether `PlaneGridPolar.act_debug_plot()` should be restored, redesigned around the current plotting stack, or removed permanently.
