from pathlib import Path

path = Path("src/nematics3d/classes/smoothed_line.py")
text = path.read_text()

old = '''    # ==================== OVERRIDE ====================
    # SmoothedLine overrides HostBase.__init__ because it must validate and cache
    # raw coordinates before the host opts pipeline is initialized, then trigger
    # the first smoothing pass immediately after opts finalization.
    # ==================================================
'''
new = '''    # ==================== OVERRIDE ====================
    # SmoothedLine lets HostBase initialize and validate its raw/state inputs,
    # then finalizes opts and triggers the first smoothing pass explicitly.
    # ==================================================
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''
        line_coord_input = (
            type(self)
            .__attr_defs__["raw_coords"]
            .validator(
                line_coord_input,
                type(self).__attr_defs__["raw_coords"].doc,
            )
        )

        is_window_warning = (
            type(self)
            .__attr_defs__["state_is_window_warning"]
            .validator(
                is_window_warning,
                type(self).__attr_defs__["state_is_window_warning"].doc,
            )
        )
        object.__setattr__(self, "raw_coords", line_coord_input)
        object.__setattr__(self, "calc_coords", self.raw_coords)
        object.__setattr__(self, "calc_result", self.raw_coords)
'''
new = '''
        object.__setattr__(self, "calc_coords", line_coord_input)
        object.__setattr__(self, "calc_result", line_coord_input)
'''
assert text.count(old) == 1
text = text.replace(old, new)

old = '''        object.__setattr__(self, "calc_is_smoothed", False)
        object.__setattr__(self, "state_is_window_warning", is_window_warning)
        object.__setattr__(self, "calc_status", "Failure, reason unknown.")

        super().__init__(
            OptsSmooth,
            opts,
            opts_defaults_override,
            name=name,
            name_replace="line",
            **kwargs,
        )
'''
new = '''        object.__setattr__(self, "calc_is_smoothed", False)
        object.__setattr__(self, "calc_status", "Failure, reason unknown.")

        super().__init__(
            OptsSmooth,
            opts,
            opts_defaults_override,
            name=name,
            name_replace="line",
            raw_coords=line_coord_input,
            state_is_window_warning=is_window_warning,
            **kwargs,
        )
'''
assert text.count(old) == 1
text = text.replace(old, new)

path.write_text(text)
