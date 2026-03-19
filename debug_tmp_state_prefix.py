from pathlib import Path
files = [
    Path(r'D:\Document\GitHub\Nematics3D\src\Nematics3D\classes\visual\glyph.py'),
    Path(r'D:\Document\GitHub\Nematics3D\src\Nematics3D\classes\visual\plot_rod.py'),
    Path(r'D:\Document\GitHub\Nematics3D\src\Nematics3D\classes\visual\plot_sphere.py'),
    Path(r'D:\Document\GitHub\Nematics3D\src\Nematics3D\classes\visual\plot_surface.py'),
    Path(r'D:\Document\GitHub\Nematics3D\src\Nematics3D\classes\visual\plot_tube.py'),
]
for p in files:
    text = p.read_text(encoding='utf-8')
    text = text.replace('raw_clip_mode', 'state_clip_mode')
    text = text.replace('raw_is_clip_inside', 'state_is_clip_inside')
    p.write_text(text, encoding='utf-8')
