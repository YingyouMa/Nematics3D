from pathlib import Path

path = Path("src/nematics3d/classes/smoothed_line.py")
text = path.read_text()

old_loop = '''    values_flat = values.reshape(len(u_samples), -1)\n    output = np.empty_like(values_flat, dtype=float)\n    deltas_all = _linefunc_sample_delta_matrix(u_samples, mode=mode)\n\n    for idx, delta in enumerate(deltas_all):\n        kernel_weights = linefunc_kernel_weights(\n'''
new_loop = '''    values_flat = values.reshape(len(u_samples), -1)\n    output = np.empty_like(values_flat, dtype=float)\n\n    for idx, u_center in enumerate(u_samples):\n        delta = u_samples - u_center\n        if mode == "wrap":\n            delta = (delta + 50.0) % 100.0 - 50.0\n\n        kernel_weights = linefunc_kernel_weights(\n'''
if old_loop not in text:
    raise RuntimeError("target linefunc smoothing loop not found")
text = text.replace(old_loop, new_loop, 1)

start_marker = '''def _linefunc_sample_delta_matrix(\n'''
end_marker = '''def _linefunc_as_u_samples(u_samples) -> np.ndarray:\n'''
start = text.find(start_marker)
end = text.find(end_marker)
if start < 0 or end < 0 or end <= start:
    raise RuntimeError("delta-matrix helper block not found")
text = text[:start] + text[end:]

path.write_text(text)
