from pathlib import Path

path = Path("src/nematics3d/classes/smoothed_line.py")
text = path.read_text()
old = '''        object.__setattr__(self, "calc_coords", line_coord_input)\n        object.__setattr__(self, "calc_result", line_coord_input)\n'''
if text.count(old) != 1:
    raise RuntimeError("Expected exactly one calc bootstrap block")
path.write_text(text.replace(old, ""))
