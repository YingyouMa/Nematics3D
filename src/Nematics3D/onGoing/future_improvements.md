## Possible Future Improvements

- Measure the runtime overhead of the logging decorator, especially in hot paths and repeated small function calls.
- Explore a repository-wide switch that can disable the logging decorator globally when performance is the priority.
- Rebuild `PlaneGrid.act_debug_plot()` around the new `bounds` relation and `Bounds.act_visualize()` instead of the legacy `PlotExtent` path.

- Decide whether `PlaneGridPolar.act_debug_plot()` should be restored, redesigned around the current plotting stack, or removed permanently.
- Design a future smoothing stage for `SmoothedLineFunc` that stays consistent with `SmoothedLine`, especially for periodic and wrap-mode behavior.
- Add a direct numeric input box alongside `PanelBase` sliders so control panels support fine-grained value entry in addition to drag-based adjustment.
- Unify the Qt binding used by `ScopedConsoleDock` with the rest of the visual Qt stack, which currently prefers `qtpy` while `console.py` still imports `PyQt5` directly.
- Make the `ScopedConsoleDock` public output API internally consistent by deciding whether `clear()` should follow the same signal-based UI-update path as `write()` / `println()`.
- Add a `clip_mode="none"` option for glyph/plot objects so a bound `bounds` object can remain attached while clipping is temporarily disabled for visual comparison.
- Add an opt-in state on `SmoothedLineFunc` for automatic refresh so it can resample itself whenever the paired `SmoothedLine` updates its smoothing opts.
- Let `SmoothedLineFunc` optionally smooth its sampled values with the same smoothing parameters used by the paired `SmoothedLine`, so function sampling can stay visually and numerically consistent with the line itself.
- Add a one-shot detailed info/summary printer for `SmoothedLine` and `SmoothedLineFunc` so users can inspect the full current smoothing, sampling, and cache state without manually checking many fields.
