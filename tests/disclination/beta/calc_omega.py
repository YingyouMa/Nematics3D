from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import nematics3d


DATA_PATH = Path(__file__).with_name("Q_1630.npy")
Q = np.load(DATA_PATH)[0]

# Q = nematics3d.QFieldObject(
#     Q=Q,
#     name="WT",
# )
# Q.act_lines_smooth()
# Q.act_visualize_disclination_lines(
#     min_line_length=0,
#     title="WT disclination lines",
# )

# Q.act_visualize_n_plane(
#     grid_origin=(256,20,20),
#     grid_normal=(0,0,1),
#     grid_spacing=4,
#     grid_size=400
# )

Q = Q[168:185, 5:32, 10:35]

Q = nematics3d.QFieldObject(
    Q=Q,
    name="WT",
)
Q.act_lines_smooth(window_length=28)
Q.act_visualize_disclination_lines(
    min_line_length=0,
    title="WT disclination lines",
)
Q.act_visualize_n_near_defect(u_percent=0)