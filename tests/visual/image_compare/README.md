# Visual Image Comparison

This folder stores small utilities for image-regression checks in `tests/visual`.

Current tool:

- `compare_png.py`: compare two PNG files, report numeric difference metrics, and
  optionally save a visual diff PNG.

Example:

```powershell
conda run -n Nematics3D python .\tests\visual\image_compare\compare_png.py `
  .\tests\visual\test_sphere\baseline\test_sphere_offscreen.png `
  .\tests\visual\test_sphere\output\test_sphere_offscreen.png `
  --diff-out .\tests\visual\test_sphere\output\test_sphere_offscreen_diff.png
```

