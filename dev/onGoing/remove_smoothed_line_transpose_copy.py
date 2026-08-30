from pathlib import Path

path = Path("src/nematics3d/classes/smoothed_line.py")
text = path.read_text()
old = "                line_points_spline.T.copy(),\n"
new = "                line_points_spline.T,\n"
if old not in text:
    raise RuntimeError("Expected splprep transpose copy not found")
path.write_text(text.replace(old, new, 1))
