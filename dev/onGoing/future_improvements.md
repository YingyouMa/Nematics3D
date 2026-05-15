## Possible Future Improvements

- Measure the runtime overhead of the logging decorator, especially in hot paths and repeated small function calls.
- Explore a repository-wide switch that can disable the logging decorator globally when performance is the priority.
- Rebuild `PlaneGrid.act_debug_plot()` around the new `bounds` relation and `Bounds.act_visualize()` instead of the legacy `PlotExtent` path.
- Clean up the `Bounds` subscription model, especially the relation between a `Bounds` host and its own visualized bounds objects, so shared-bounds behavior stays easy to inspect and reason about.

- Decide whether `PlaneGridPolar.act_debug_plot()` should be restored, redesigned around the current plotting stack, or removed permanently.
- Design a future smoothing stage for `SmoothedLineFunc` that stays consistent with `SmoothedLine`, especially for periodic and wrap-mode behavior.
- Add a direct numeric input box alongside `PanelBase` sliders so control panels support fine-grained value entry in addition to drag-based adjustment.
- Unify the Qt binding used by `ScopedConsoleDock` with the rest of the visual Qt stack, which currently prefers `qtpy` while `console.py` still imports `PyQt5` directly.
- Make the `ScopedConsoleDock` public output API internally consistent by deciding whether `clear()` should follow the same signal-based UI-update path as `write()` / `println()`.
- Add a `clip_mode="none"` option for glyph/plot objects so a bound `bounds` object can remain attached while clipping is temporarily disabled for visual comparison.
- Improve scalar bar control and management so visual tests and multi-glyph figures can enable, suppress, reuse, and update scalar bars more predictably.
- Add a scene-state snapshot workflow so the current interactive figure/object state can be saved and restored later as a named checkpoint.
- Add console commands for direct save/load of the current scene state so interactive sessions can checkpoint and restore without leaving the console.
- Add an opt-in state on `SmoothedLineFunc` for automatic refresh so it can resample itself whenever the paired `SmoothedLine` updates its smoothing opts.
- Let `SmoothedLineFunc` optionally smooth its sampled values with the same smoothing parameters used by the paired `SmoothedLine`, so function sampling can stay visually and numerically consistent with the line itself.
- Add a one-shot detailed info/summary printer for `SmoothedLine` and `SmoothedLineFunc` so users can inspect the full current smoothing, sampling, and cache state without manually checking many fields.
- Add an optional anisotropy diagnostic to radial Fourier-spectrum averaging so the result can report whether the radial average is likely meaningful for the underlying spectrum.
- Design a general ClassBase/HostBase policy for mutable ndarray outputs: derived `calc_`/`entity_` arrays and any opts-side arrays may need read-only views, defensive copies, or a documented convention so users cannot accidentally mutate internal caches in place.
- Define a repository-wide convention for `as_*` normalizers that accept existing `ClassBase`/`HostBase` instances, such as `as_bounds()` and `as_plotfigure()`: decide whether fixedness flags like `is_fixed` should be ignored, applied in place, copied into a new fixed object, or rejected when the input is already an object instance.
- Revisit the naming and overlap between `ClassBase.show_attr_doc()` and `show_attr_desc()`: the current split between pure doc lookup and formatted description output is easy to confuse, so a future cleanup may rename them more explicitly or merge them behind one clearer inspection helper.
- Consider adding an optional `__repr_order__` class variable for `OptsBase` subclasses as a display-only override for `repr(opts)`. Default behavior should remain `__attrs__` insertion order so related options can stay grouped by human intent; `__repr_order__`, if added, should affect only presentation and should leave `act_asdict()`, JSON export, defaults, and validation order unchanged.
