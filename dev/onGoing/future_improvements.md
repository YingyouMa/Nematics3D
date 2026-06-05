## Possible Future Improvements

- Measure the runtime overhead of the logging decorator, especially in hot paths and repeated small function calls.
- Explore a repository-wide switch that can disable the logging decorator globally when performance is the priority.
- Rebuild `PlaneGrid.act_debug_plot()` around the new `bounds` relation and `Bounds.act_visualize()` instead of the legacy `PlotExtent` path.
- Add a `bounds` relation to `SurfaceSampling` so it can be clipped to a `Bounds` object in the same way `PlaneGrid` does: only surface area inside the bounds participates in sampling, and the sampled points update automatically when the bounds geometry changes. The subscription and sync-task wiring between `SurfaceSampling`, its attached glyph, and the bounds should follow the same pattern used by `PlaneGrid` and its downstream glyphs.

- Decide whether `PlaneGridPolar.act_debug_plot()` should be restored, redesigned around the current plotting stack, or removed permanently.
- Add a `clip_mode="none"` option for glyph/plot objects so a bound `bounds` object can remain attached while clipping is temporarily disabled for visual comparison.
- Add a scene-state snapshot workflow so the current interactive figure/object state can be saved and restored later as a named checkpoint.
- Add console commands for direct save/load of the current scene state so interactive sessions can checkpoint and restore without leaving the console.
- Design an `act_copy()` for `PlotFigure` that produces a standalone duplicate of the full figure — all glyphs, visuals, and scalar bars are reproduced with their current opts and geometry, but all `source` / `owner` / `bounds` relations pointing back to physical host objects are dropped so the copy is self-contained and does not keep any host alive.
- Add an opt-in state on `SmoothedLineFunc` for automatic refresh so it can resample itself whenever the paired `SmoothedLine` updates its smoothing opts.
- Fix the synchronization contract between `SmoothedLine` and `SmoothedLineFunc`: when the owner line geometry or cached smoothing result changes, existing line functions should become stale in a visible way or refresh automatically instead of silently keeping old sampled/interpolated values.
- Add a one-shot detailed info/summary printer for `SmoothedLine` and `SmoothedLineFunc` so users can inspect the full current smoothing, sampling, and cache state without manually checking many fields.
- Design a general ClassBase/HostBase policy for mutable ndarray outputs: derived `calc_`/`entity_` arrays and any opts-side arrays may need read-only views, defensive copies, or a documented convention so users cannot accidentally mutate internal caches in place.
- Decide whether `ClassBase` and `HostBase` should be more strictly separated: currently `AttrDef.is_reapply_opts_after_raw` is a `HostBase`-only concept that leaks into the `ClassBase` layer. A future cleanup could subclass `AttrDef` in `HostBase` or move the field there so `ClassBase` stays free of opts-related concerns.
- Revisit the naming and overlap between `ClassBase.show_attr_doc()` and `show_attr_desc()`: the current split between pure doc lookup and formatted description output is easy to confuse, so a future cleanup may rename them more explicitly or merge them behind one clearer inspection helper.
- Consider adding an optional `__repr_order__` class variable for `OptsBase` subclasses as a display-only override for `repr(opts)`. Default behavior should remain `__attrs__` insertion order so related options can stay grouped by human intent; `__repr_order__`, if added, should affect only presentation and should leave `act_asdict()`, JSON export, defaults, and validation order unchanged.
- Audit zero-vector director handling across validation, Q construction, defect detection, interpolation, and visualization paths so masked or missing-director data stays compatible with every director workflow.
