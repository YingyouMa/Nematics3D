from pathlib import Path

src = Path("src/nematics3d/classes/smoothed_line.py")
text = src.read_text()

text = text.replace(
    "from .registry_base import RegistryBase\n",
    "from .registry_base import RegistryBase\nfrom .result_base import ResultBase\n",
    1,
)

text = text.replace(
'''    def act_create_linefunc(\n        self,\n        func,\n        u_samples,\n        func_kwargs: Mapping[str, Any] | None = None,\n        is_follow_owner_opts: bool = True,\n        name: str | None = None,\n    ):''',
'''    def act_create_linefunc(\n        self,\n        func,\n        u_samples,\n        func_kwargs: Mapping[str, Any] | None = None,\n        result_value_attr: str = "value",\n        is_follow_owner_opts: bool = True,\n        name: str | None = None,\n    ):''',
    1,
)
text = text.replace(
'''            func_kwargs=func_kwargs,\n            is_follow_owner_opts=is_follow_owner_opts,''',
'''            func_kwargs=func_kwargs,\n            result_value_attr=result_value_attr,\n            is_follow_owner_opts=is_follow_owner_opts,''',
    1,
)

old_defs = '''        "raw_func": AttrDef(\n            doc=(\n                "Numerical sampling function mapping one u_percent to a value "\n                "or a (value, metric) / (value, metric, payload_samples) / "\n                "(value, metric, payload_samples, payload_shared) tuple."\n            ),\n            kind="raw",\n            validator=lambda v, d: v if callable(v) else (_raise_type_error(d, v)),\n        ),\n        "raw_u_samples": AttrDef(\n'''
new_defs = '''        "raw_func": AttrDef(\n            doc=(\n                "Numerical sampling function mapping one u_percent to a "\n                "ResultBase instance."\n            ),\n            kind="raw",\n            validator=lambda v, d: v if callable(v) else (_raise_type_error(d, v)),\n        ),\n        "raw_result_value_attr": AttrDef(\n            doc=(\n                "ResultBase attribute whose per-sample value is smoothed and "\n                "interpolated."\n            ),\n            kind="raw",\n            validator=lambda v, d: SmoothedLineFunc._helper_validate_result_value_attr(\n                v, name=d\n            ),\n        ),\n        "raw_u_samples": AttrDef(\n'''
if old_defs not in text:
    raise RuntimeError("raw_func AttrDef block not found")
text = text.replace(old_defs, new_defs, 1)

old_calc_defs = '''        "calc_values": AttrDef(\n            doc="Values returned by the numerical function at each sampling location.",\n            kind="calc",\n        ),\n        "calc_metrics": AttrDef(\n            doc="Per-sample metrics returned by the numerical function, or None if unavailable.",\n            kind="calc",\n        ),\n        "calc_payload_samples": AttrDef(\n            doc="Per-sample payload objects returned by the numerical function, or None if unavailable.",\n            kind="calc",\n        ),\n        "calc_payload_shared": AttrDef(\n            doc="Shared payload returned for the full sampled function, or None if unavailable.",\n            kind="calc",\n        ),\n'''
new_calc_defs = '''        "calc_results": AttrDef(\n            doc="Raw ResultBase objects returned at each sampling location.",\n            kind="calc",\n        ),\n        "calc_values": AttrDef(\n            doc=(\n                "Smoothed sample values extracted from the configured "\n                "ResultBase attribute."\n            ),\n            kind="calc",\n        ),\n'''
if old_calc_defs not in text:
    raise RuntimeError("old calc metadata AttrDef block not found")
text = text.replace(old_calc_defs, new_calc_defs, 1)

marker = '''    @staticmethod\n    def _helper_validate_func_kwargs(\n'''
helper = '''    @staticmethod\n    def _helper_validate_result_value_attr(\n        result_value_attr,\n        *,\n        name: str = "`result_value_attr`",\n    ) -> str:\n        result_value_attr = as_str(result_value_attr, name=name)\n        if not result_value_attr:\n            raise ValueError(f"{name} must be a non-empty string.")\n        return result_value_attr\n\n'''
if marker not in text:
    raise RuntimeError("func kwargs helper marker not found")
text = text.replace(marker, helper + marker, 1)

text = text.replace(
'''        func_kwargs: Mapping[str, Any] | None = None,\n        is_follow_owner_opts: bool = True,\n        name: str = "smoothed line function",''',
'''        func_kwargs: Mapping[str, Any] | None = None,\n        result_value_attr: str = "value",\n        is_follow_owner_opts: bool = True,\n        name: str = "smoothed line function",''',
    1,
)

needle = '''        object.__setattr__(\n            self,\n            "raw_u_samples",\n'''
insert = '''        object.__setattr__(\n            self,\n            "raw_result_value_attr",\n            type(self)\n            .__attr_defs__["raw_result_value_attr"]\n            .validator(\n                result_value_attr,\n                type(self).__attr_defs__["raw_result_value_attr"].doc,\n            ),\n        )\n'''
if needle not in text:
    raise RuntimeError("constructor raw_u_samples marker not found")
text = text.replace(needle, insert + needle, 1)

text = text.replace(
'''        object.__setattr__(self, "calc_values", None)\n        object.__setattr__(self, "calc_metrics", None)\n        object.__setattr__(self, "calc_payload_samples", None)\n        object.__setattr__(self, "calc_payload_shared", None)\n''',
'''        object.__setattr__(self, "calc_results", None)\n        object.__setattr__(self, "calc_values", None)\n''',
    1,
)

start = text.find('''        values = []\n        metrics = []\n''')
end = text.find('''        interpolator, values_smooth = linefunc_build_smoothed_interpolator(\n''', start)
if start < 0 or end < 0:
    raise RuntimeError("old sampling protocol block not found")
new_sampling = '''        results = []\n        values = []\n        for u in self.raw_u_samples:\n            u_float = float(u)\n            sample_result = self.raw_func(u_float, **self.raw_func_kwargs)\n            if not isinstance(sample_result, ResultBase):\n                raise TypeError(\n                    "SmoothedLineFunc `raw_func` must return a ResultBase instance "\n                    f"at every sample; got {type(sample_result).__name__} at "\n                    f"u_percent={u_float}."\n                )\n            if not hasattr(sample_result, self.raw_result_value_attr):\n                raise AttributeError(\n                    f"SmoothedLineFunc `raw_func` returned "\n                    f"{type(sample_result).__name__} at u_percent={u_float}, but "\n                    f"it has no attribute {self.raw_result_value_attr!r}. Set "\n                    "`result_value_attr` to the ResultBase attribute that should "\n                    "be smoothed."\n                )\n\n            results.append(sample_result)\n            values.append(\n                np.asarray(getattr(sample_result, self.raw_result_value_attr))\n            )\n\n        values = np.stack(values, axis=0)\n\n'''
text = text[:start] + new_sampling + text[end:]

text = text.replace(
'''        object.__setattr__(self, "impl_owner_opts_snapshot", dict(opts_snapshot))\n        object.__setattr__(self, "calc_values", values_smooth)\n        object.__setattr__(self, "calc_metrics", metrics)\n        object.__setattr__(self, "calc_payload_samples", payload_samples)\n        object.__setattr__(self, "calc_payload_shared", payload_shared)\n        object.__setattr__(self, "entity_interpolator", interpolator)\n''',
'''        object.__setattr__(self, "impl_owner_opts_snapshot", dict(opts_snapshot))\n        object.__setattr__(self, "calc_results", tuple(results))\n        object.__setattr__(self, "calc_values", values_smooth)\n        object.__setattr__(self, "entity_interpolator", interpolator)\n''',
    1,
)

text = text.replace(
'''    Users provide a callable `func(u_percent, **func_kwargs)` together with\n    normalized sample locations in `[0, 100]`. The object evaluates that\n    callable on the current line parameter domain, stores sampled outputs, and\n    exposes a linear interpolator for later reuse.\n''',
'''    Users provide a callable `func(u_percent, **func_kwargs)` together with\n    normalized sample locations in `[0, 100]`. The callable must return a\n    ResultBase instance at every sample. The configured `result_value_attr`\n    (default `"value"`) selects which result attribute is smoothed and\n    interpolated, while the complete raw result objects remain available in\n    `calc_results`.\n''',
    1,
)

src.write_text(text)

# Update registry tests to use the new ResultBase protocol.
test_registry = Path("tests/smooth/test_smoothed_line_func_registry.py")
t = test_registry.read_text()
t = t.replace("import types\nimport unittest\n", "import types\nimport unittest\nfrom dataclasses import dataclass\n", 1)
t = t.replace(
    "from nematics3d.classes.registry_base import RegistryBase\n",
    "from nematics3d.classes.registry_base import RegistryBase\nfrom nematics3d.classes.result_base import ResultBase\n",
    1,
)
insert_marker = '''class WrapLineFuncModeLine(SmoothedLine):\n'''
insert_text = '''@dataclass(repr=False)\nclass ScalarResult(ResultBase):\n    value: float\n\n\ndef scalar_result(u):\n    return ScalarResult(float(u))\n\n\ndef doubled_scalar_result(u):\n    return ScalarResult(2.0 * float(u))\n\n\n'''
t = t.replace(insert_marker, insert_text + insert_marker, 1)
t = t.replace("lambda u: u", "scalar_result")
t = t.replace("lambda u: 2 * u", "doubled_scalar_result")
test_registry.write_text(t)

# Add focused protocol tests.
test_func = Path("tests/smooth/test_smoothed_line_func.py")
t = test_func.read_text()
t = t.replace("import types\n\nimport numpy as np\n", "import types\nfrom dataclasses import dataclass\n\nimport numpy as np\nimport pytest\n", 1)
t = t.replace(
'''from nematics3d.classes.smoothed_line import (\n    linefunc_kernel_weights,\n''',
'''from nematics3d.classes.result_base import ResultBase\nfrom nematics3d.classes.smoothed_line import (\n    SmoothedLine,\n    linefunc_kernel_weights,\n''',
    1,
)
append = '''\n\n@dataclass(repr=False)\nclass ValueResult(ResultBase):\n    value: float\n    diagnostic: float\n\n\n@dataclass(repr=False)\nclass AngleResult(ResultBase):\n    angle: float\n    diagnostic: float\n\n\ndef _build_protocol_line():\n    x = np.linspace(0.0, 1.0, 60)\n    coords = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))\n    return SmoothedLine(coords, window_length=5, min_line_length=2)\n\n\ndef test_linefunc_stores_full_resultbase_samples_and_uses_default_value_attr():\n    line = _build_protocol_line()\n    linefunc = line.act_create_linefunc(\n        lambda u: ValueResult(value=u, diagnostic=u + 1.0),\n        [0.0, 25.0, 50.0, 75.0, 100.0],\n    )\n\n    assert linefunc.raw_result_value_attr == "value"\n    assert isinstance(linefunc.calc_results, tuple)\n    assert len(linefunc.calc_results) == 5\n    assert all(isinstance(result, ValueResult) for result in linefunc.calc_results)\n    assert linefunc.calc_results[2].value == 50.0\n    assert linefunc.calc_results[2].diagnostic == 51.0\n    assert linefunc.calc_values.shape == (5,)\n\n\ndef test_linefunc_can_select_custom_result_value_attr():\n    line = _build_protocol_line()\n    linefunc = line.act_create_linefunc(\n        lambda u: AngleResult(angle=2.0 * u, diagnostic=-u),\n        [0.0, 25.0, 50.0, 75.0, 100.0],\n        result_value_attr="angle",\n    )\n\n    assert linefunc.raw_result_value_attr == "angle"\n    assert linefunc.calc_results[2].angle == 100.0\n    assert linefunc.calc_results[2].diagnostic == -50.0\n    assert linefunc.calc_values.shape == (5,)\n\n\ndef test_linefunc_rejects_non_resultbase_sample_return():\n    line = _build_protocol_line()\n    with pytest.raises(TypeError, match="must return a ResultBase instance"):\n        line.act_create_linefunc(lambda u: u, [0.0, 50.0, 100.0])\n\n\ndef test_linefunc_rejects_missing_configured_result_attribute():\n    line = _build_protocol_line()\n    with pytest.raises(AttributeError, match="has no attribute 'value'"):\n        line.act_create_linefunc(\n            lambda u: AngleResult(angle=u, diagnostic=0.0),\n            [0.0, 50.0, 100.0],\n        )\n'''
t += append
test_func.write_text(t)
