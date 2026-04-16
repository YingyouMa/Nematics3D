import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nematics3d


DATA_DIR = Path(__file__).resolve().parent / "data"

Q_data = np.load(DATA_DIR / "Q_example_workflow.npy")

# Q = nematics3d.QFieldObject(Q=Q_data, name="workflow_Q")

# Q.act_lines_smooth(
#     window_length=31,
#     min_line_length=61,
# )

# Q.act_visualize_disclination_lines(
#     min_line_length=61,
#     line_color=(0.45, 0.45, 0.45),
#     line_radius=0.35,
#     extent_color=(0.15, 0.15, 0.15),
#     extent_radius=0.08,
# )

# Q.act_visualize_n_near_defect(u_percent=0, index_line=6)

Q_data = Q_data[145:175, 13:51, 64:94]

Q = nematics3d.QFieldObject(Q=Q_data, name="workflow_Q")

Q.act_lines_smooth(
    window_length=31,
    min_line_length=61,
)

Q.act_visualize_disclination_lines(
    min_line_length=61,
    line_color=(0.45, 0.45, 0.45),
    line_radius=0.35,
    extent_color=(0.15, 0.15, 0.15),
    extent_radius=0.08,
)

Q.act_visualize_n_near_defect(u_percent=0, index_line=0, grid_layers=5, grid_dr=0.61)

norm = Q.lines[0].act_calc_norm()
for u in range(0, 100, 5):
    omega = Q.lines[0].smooth.act_calc_omega(u_percent=u)[0]
    beta = np.arccos(-norm@omega) / np.pi * 180
    print(u, beta)