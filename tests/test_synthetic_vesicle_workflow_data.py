from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


GENERATOR = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "workflows"
    / "_data_generation"
    / "generate_synthetic_vesicle_q.py"
)


def _load_generator():
    spec = spec_from_file_location("synthetic_vesicle_generator", GENERATOR)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_synthetic_vesicle_generator_has_reusable_ground_truth(tmp_path):
    generator = _load_generator()
    data = generator.generate_synthetic_vesicle(shape=(32, 34, 36), seed=17)

    assert data["Q"].shape == (32, 34, 36, 5)
    assert data["Q"].dtype == np.float32
    assert np.all(np.isfinite(data["Q"]))
    assert np.mean(np.abs(data["Q"] - data["Q_clean"])) > 1e-3

    mask = data["mask_vesicle"].astype(bool)
    n = data["director_ground_truth"]
    assert mask.any() and (~mask).any()

    # The interior and exterior should not collapse to the same texture.
    inside_abs_nz = float(np.mean(np.abs(n[..., 2][mask])))
    outside_abs_nz = float(np.mean(np.abs(n[..., 2][~mask])))
    assert abs(outside_abs_nz - inside_abs_nz) > 0.15

    lines = data["defect_lines"]
    assert set(lines) == {"span_A", "span_B", "edge_C", "loop_D"}
    assert len(lines["span_A"]) > len(lines["edge_C"])
    assert len(lines["span_B"]) > len(lines["edge_C"])
    np.testing.assert_allclose(lines["loop_D"][0], lines["loop_D"][-1], atol=1e-6)

    generator.write_dataset(tmp_path, shape=(32, 34, 36), seed=17)
    expected = {
        "Q.npy",
        "Q_clean_ground_truth.npy",
        "director_ground_truth.npy",
        "S_ground_truth.npy",
        "mask_vesicle.npy",
        "defect_lines_ground_truth.npz",
        "metadata.json",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
