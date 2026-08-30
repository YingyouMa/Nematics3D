from pathlib import Path


SMOOTHED_LINE = Path("src/nematics3d/classes/smoothed_line.py")
TEST_FILE = Path("tests/smooth/test_smoothed_line.py")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one replacement target, found {count}: {old[:80]!r}")
    return text.replace(old, new)


def update_smoothed_line() -> None:
    text = SMOOTHED_LINE.read_text()

    text = replace_once(
        text,
        '"window_ratio":         lambda v, d: None if v is None else as_number(v, name=d),',
        '"window_ratio":         lambda v, d: None if v is None else as_number(v, name=d, value_range=(1e-12, np.inf)),',
    )

    text = replace_once(
        text,
        '''        if kwargs:\n            if "window_ratio" in kwargs and "window_length" not in kwargs:\n                object.__setattr__(self.opts, "window_length", None)\n            if "window_ratio" not in kwargs and "window_length" in kwargs:\n                object.__setattr__(self.opts, "window_ratio", None)\n\n        with self.opts.act_internal_update():\n            cover_value(\n                self.opts,\n                is_allow_cover_target_set=True,\n                is_allow_unset_source=False,\n                **kwargs,\n            )\n''',
        '''        with self.opts.act_internal_update():\n            if kwargs:\n                if "window_ratio" in kwargs and "window_length" not in kwargs:\n                    self.opts.window_length = None\n                if "window_ratio" not in kwargs and "window_length" in kwargs:\n                    self.opts.window_ratio = None\n\n            cover_value(\n                self.opts,\n                is_allow_cover_target_set=True,\n                is_allow_unset_source=False,\n                **kwargs,\n            )\n''',
    )

    marker = '''    def _helper_fallback_no_smooth(self, reason: str) -> None:\n        object.__setattr__(self, "calc_is_smoothed", False)\n        object.__setattr__(self, "calc_result", self.calc_coords)\n        object.__setattr__(self, "entity_tck", None)\n        object.__setattr__(\n            self,\n            "calc_status",\n            f"The line `{self.name}` is not smoothed, reason: {reason}.",\n        )\n'''
    helpers = marker + '''\n    def _helper_resolve_window_opts(self, *, logger=None) -> None:\n        """Resolve the active smoothing window and synchronize both window opts."""\n        window_length = self.opts.window_length\n        window_ratio = self.opts.window_ratio\n\n        if window_length is None:\n            if window_ratio is None:\n                raise SmoothingConfigError(\n                    "No input value provided for smooth window length."\n                )\n            if self.calc_num_init <= 0:\n                raise SmoothingConfigError("Cannot smooth an empty line.")\n            window_length = int(self.calc_num_init / window_ratio / 2) * 2 + 1\n        else:\n            if (\n                window_ratio is not None\n                and self.state_is_window_warning\n                and logger is not None\n            ):\n                logger.warning(\n                    f"Window_length is manual input as {window_length}. "\n                    f"window_ratio ({window_ratio}) would be ignored and reset."\n                )\n            window_length = int(window_length)\n            if window_length % 2 == 0:\n                window_length += 1\n\n        resolved_ratio = self.calc_num_init / window_length\n        with self.opts.act_internal_update():\n            self.opts.window_length = window_length\n            self.opts.window_ratio = resolved_ratio\n\n    def _helper_resolve_spline_u(self, u_percent) -> float:\n        """Validate a percent spline parameter and map it to FITPACK's [0, 1] domain."""\n        if getattr(self, "entity_tck", None) is None:\n            raise RuntimeError(\n                "Spline cache `entity_tck` is missing. "\n                "Probably the line is not properly initialized or successfully smoothed."\n            )\n\n        u_percent = as_number(\n            u_percent,\n            value_range=(0, 100),\n            name="Continuous spline parameter along the curve",\n        )\n        u = float(u_percent) / 100.0\n        if self.opts.mode == "wrap":\n            u = float(np.mod(u, 1.0))\n        return u\n'''
    text = replace_once(text, marker, helpers)

    text = replace_once(
        text,
        '''        try:\n            if self.opts.window_length is None:\n                if self.opts.window_ratio is None:\n                    reason = "No input value provided for smooth window length."\n                    raise SmoothingConfigError(reason)\n                object.__setattr__(\n                    self.opts,\n                    "window_length",\n                    int(self.calc_num_init / self.opts.window_ratio / 2) * 2 + 1,\n                )\n                object.__setattr__(\n                    self.opts,\n                    "window_ratio",\n                    self.calc_num_init / self.opts.window_length,\n                )\n            else:\n                if self.opts.window_ratio is not None and self.state_is_window_warning:\n                    logger.warning(\n                        f"Window_length is manual input as {self.opts.window_length}. "\n                        f"window_ratio ({self.opts.window_ratio}) would be ignored and reset."\n                    )\n                window_length = int(self.opts.window_length)\n                if window_length % 2 == 0:\n                    window_length += 1\n                object.__setattr__(self.opts, "window_length", window_length)\n                object.__setattr__(\n                    self.opts,\n                    "window_ratio",\n                    self.calc_num_init / self.opts.window_length,\n                )\n\n            if self.calc_num_init < self.opts.min_line_length:\n''',
        '''        try:\n            self._helper_resolve_window_opts(logger=logger)\n\n            if self.calc_num_init < self.opts.min_line_length:\n''',
    )

    text = replace_once(
        text,
        '''                self._helper_fallback_no_smooth(reason)\n                raise SmoothingConfigError(reason)\n''',
        '''                raise SmoothingConfigError(reason)\n''',
    )

    text = replace_once(
        text,
        '''    def act_calc_tangent(self, u_percent, is_return_coord=False):\n\n        tck = getattr(self, "entity_tck", None)\n        if tck is None:\n            raise RuntimeError(\n                "Spline cache `entity_tck` is missing."\n                "Probably the line is not properly initialized or successfully smoothed."\n            )\n\n        u_percent = as_number(\n            u_percent,\n            value_range=(0, 100),\n            name="Continuous spline parameter along the curve",\n        )\n        u_percent /= 100\n        if self.opts.mode == "wrap":\n            u_percent = np.mod(u_percent, 1.0)\n        dr_dx = np.asarray(splev(u_percent, self.entity_tck, der=1), dtype=float)\n        length = float(np.linalg.norm(dr_dx))\n        if (not np.isfinite(length)) or length < 1e-9:\n            raise ValueError(\n                f"Degenerate spline derivative at {u_percent}: ||dr/dx||={length}."\n            )\n\n        t_hat = dr_dx / length\n        if not is_return_coord:\n            return t_hat\n\n        coord = np.asarray(splev(u_percent, self.entity_tck, der=0), dtype=float)\n        return t_hat, coord\n\n    def act_calc_pos(self, u_percent):\n        tck = getattr(self, "entity_tck", None)\n        if tck is None:\n            raise RuntimeError(\n                "Spline cache `entity_tck` is missing."\n                "Probably the line is not properly initialized or successfully smoothed."\n            )\n\n        u_percent = as_number(\n            u_percent,\n            value_range=(0, 100),\n            name="Continuous spline parameter along the curve",\n        )\n        u_percent /= 100\n        if self.opts.mode == "wrap":\n            u_percent = np.mod(u_percent, 1.0)\n\n        return np.asarray(splev(u_percent, self.entity_tck, der=0), dtype=float)\n''',
        '''    def act_calc_tangent(self, u_percent, is_return_coord=False):\n        u = self._helper_resolve_spline_u(u_percent)\n        is_return_coord = as_bool(\n            is_return_coord,\n            name="Whether to return the spline coordinate",\n        )\n\n        dr_dx = np.asarray(splev(u, self.entity_tck, der=1), dtype=float)\n        length = float(np.linalg.norm(dr_dx))\n        if (not np.isfinite(length)) or length < 1e-9:\n            raise ValueError(f"Degenerate spline derivative at {u}: ||dr/dx||={length}.")\n\n        t_hat = dr_dx / length\n        if not is_return_coord:\n            return t_hat\n\n        coord = np.asarray(splev(u, self.entity_tck, der=0), dtype=float)\n        return t_hat, coord\n\n    def act_calc_pos(self, u_percent):\n        u = self._helper_resolve_spline_u(u_percent)\n        return np.asarray(splev(u, self.entity_tck, der=0), dtype=float)\n''',
    )

    SMOOTHED_LINE.write_text(text)


def update_tests() -> None:
    text = TEST_FILE.read_text()
    marker = "def test_smoothed_line_window_resolution_and_fallback_contract():"
    if marker in text:
        raise RuntimeError("Focused SmoothedLine cleanup tests already exist.")

    text += '''\n\n\ndef test_smoothed_line_window_resolution_and_fallback_contract():\n    """Cover window normalization plus representative recoverable fallbacks."""\n    _, noisy = _build_noisy_line()\n\n    ratio_line = SmoothedLine(\n        noisy,\n        window_ratio=15,\n        order=3,\n        num_out_ratio=1,\n        min_line_length=2,\n        mode="interp",\n    )\n    assert ratio_line.calc_is_smoothed is True\n    assert ratio_line.opts.window_length == 9\n    np.testing.assert_allclose(\n        ratio_line.opts.window_ratio,\n        ratio_line.calc_num_init / ratio_line.opts.window_length,\n    )\n\n    even_line = SmoothedLine(\n        noisy,\n        window_length=8,\n        order=3,\n        num_out_ratio=1,\n        min_line_length=2,\n        mode="interp",\n    )\n    assert even_line.calc_is_smoothed is True\n    assert even_line.opts.window_length == 9\n\n    short_line = SmoothedLine(\n        noisy[:8],\n        window_length=5,\n        order=3,\n        min_line_length=50,\n        mode="interp",\n    )\n    assert short_line.calc_is_smoothed is False\n    assert short_line.entity_tck is None\n    np.testing.assert_array_equal(short_line.result, short_line.calc_coords)\n    assert "minimum length" in short_line.calc_status.lower()\n\n\ndef test_smoothed_line_query_helpers_and_wrap_boundary():\n    """Keep position/tangent parameter handling shared and periodic at 100%."""\n    _, noisy = _build_noisy_line()\n    line = SmoothedLine(\n        noisy,\n        window_length=9,\n        order=3,\n        num_out_ratio=1,\n        min_line_length=2,\n        mode="wrap",\n    )\n\n    np.testing.assert_allclose(line.act_calc_pos(100), line.act_calc_pos(0), atol=1e-12)\n    tangent, coord = line.act_calc_tangent(100, is_return_coord=True)\n    np.testing.assert_allclose(coord, line.act_calc_pos(0), atol=1e-12)\n    np.testing.assert_allclose(np.linalg.norm(tangent), 1.0, atol=1e-12)\n\n\ndef test_smoothed_line_zero_window_ratio_does_not_reach_division():\n    """A non-positive ratio must be rejected before smoothing window arithmetic."""\n    _, noisy = _build_noisy_line()\n    line = SmoothedLine(\n        noisy,\n        window_ratio=0,\n        order=3,\n        min_line_length=2,\n        mode="interp",\n    )\n    assert line.calc_is_smoothed is False\n    assert line.entity_tck is None\n    assert "no input value" in line.calc_status.lower()\n'''
    TEST_FILE.write_text(text)


if __name__ == "__main__":
    update_smoothed_line()
    update_tests()
