import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Nematics3D

DATA_DIR = Path(__file__).resolve().parent / "data"

n = np.load(DATA_DIR / "n_example_global.npy")
S = np.load(DATA_DIR / "S_example_global.npy")

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True, name="testQ")
Q.act_lines_smooth()
Q.act_visualize_disclination_lines(is_wrap=False, extent_color=(0.5, 0.5, 0.5), bounds=None)
Q.act_visualize_disclination_lines(
    is_new=True,
    is_wrap=True,
    line_shading_type="pbr",
    figure_azimuth=0,
    figure_elevation=45,
)
