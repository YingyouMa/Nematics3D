from pathlib import Path

path = Path("src/nematics3d/classes/disclination_line.py")
text = path.read_text()

old_helper = '''def _helper_sample_beta_from_smooth(\n    u_percent: float,\n    *,\n    smooth,\n    opts_grid: OptsPlaneGridPolar | None = None,\n    opts_grid_defaults_override: Mapping[str, Any] | None = None,\n    **grid_kwargs,\n) -> tuple[float, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:\n    """Evaluate beta and return per-sample plus shared diagnostics."""\n    result = smooth.act_calc_omega(\n        u_percent,\n        opts_grid=opts_grid,\n        opts_grid_defaults_override=opts_grid_defaults_override,\n        **grid_kwargs,\n    )\n    tangent = np.asarray(smooth.act_calc_tangent(u_percent), dtype=float)\n    payload_sample = {\n        "omega": np.asarray(result.omega, dtype=float),\n        "tangent": tangent,\n    }\n    payload_shared = {\n        "R": result.R,\n        "num_directors": result.num_directors,\n        "layer": result.layer,\n    }\n    return result.beta, dict(result.metric), payload_sample, payload_shared\n'''
new_helper = '''def _helper_sample_beta_from_smooth(\n    u_percent: float,\n    *,\n    smooth,\n    opts_grid: OptsPlaneGridPolar | None = None,\n    opts_grid_defaults_override: Mapping[str, Any] | None = None,\n    **grid_kwargs,\n) -> DefectSectionOmegaResult:\n    """Evaluate and return the complete beta-section result at one sample."""\n    return smooth.act_calc_omega(\n        u_percent,\n        opts_grid=opts_grid,\n        opts_grid_defaults_override=opts_grid_defaults_override,\n        **grid_kwargs,\n    )\n'''
if old_helper not in text:
    raise RuntimeError("old beta linefunc helper not found")
text = text.replace(old_helper, new_helper, 1)

text = text.replace(
'''        "position": "Wrapped real-space section origin used for the polar grid.",\n    }\n\n    beta: float\n    u_percent: float\n    position: np.ndarray\n''',
'''        "position": "Wrapped real-space section origin used for the polar grid.",\n        "tangent": "Unit tangent of the smoothed line at the sampled section.",\n    }\n\n    beta: float\n    u_percent: float\n    position: np.ndarray\n    tangent: np.ndarray\n''',
    1,
)

text = text.replace(
'''            },\n            name=name,\n        )\n''',
'''            },\n            result_value_attr="beta",\n            name=name,\n        )\n''',
    1,
)

text = text.replace(
'''            beta=beta,\n            u_percent=float(u_percent),\n            position=origin,\n        )\n''',
'''            beta=beta,\n            u_percent=float(u_percent),\n            position=origin,\n            tangent=np.asarray(tangent, dtype=float),\n        )\n''',
    1,
)

path.write_text(text)

# Update the existing beta integration assertions to the new complete-result model.
test_path = Path("tests/classes/test_q_field_object_phase2.py")
t = test_path.read_text()
old = '''        self.assertIsNotNone(beta_func.calc_metrics)\n        self.assertIsNotNone(beta_func.calc_payload_samples)\n        self.assertIsNotNone(beta_func.calc_payload_shared)\n        self.assertEqual(\n            len(beta_func.calc_payload_samples), len(beta_func.raw_u_samples)\n        )\n        self.assertIsInstance(beta_func.calc_metrics[0], dict)\n        self.assertIn("omega", beta_func.calc_payload_samples[0])\n        self.assertIn("tangent", beta_func.calc_payload_samples[0])\n        self.assertIn("R", beta_func.calc_payload_shared)\n        self.assertIn("num_directors", beta_func.calc_payload_shared)\n        self.assertIn("layer", beta_func.calc_payload_shared)\n'''
new = '''        self.assertEqual(beta_func.raw_result_value_attr, "beta")\n        self.assertIsInstance(beta_func.calc_results, tuple)\n        self.assertEqual(len(beta_func.calc_results), len(beta_func.raw_u_samples))\n        result0 = beta_func.calc_results[0]\n        self.assertIsInstance(result0, DefectSectionOmegaResult)\n        self.assertIsInstance(result0.metric, dict)\n        self.assertEqual(result0.omega.shape, (3,))\n        self.assertEqual(result0.tangent.shape, (3,))\n        self.assertGreater(result0.R, 0)\n        self.assertGreater(result0.num_directors, 0)\n        self.assertGreaterEqual(result0.layer, 0)\n'''
if old not in t:
    raise RuntimeError("old beta payload assertions not found")
t = t.replace(old, new, 1)

# Ensure the concrete result class is available in the test module.
t = t.replace(
    "from nematics3d.classes.q_field_object import QFieldObject\n",
    "from nematics3d.classes.q_field_object import QFieldObject\nfrom nematics3d.classes.disclination_line import DefectSectionOmegaResult\n",
    1,
)
test_path.write_text(t)
