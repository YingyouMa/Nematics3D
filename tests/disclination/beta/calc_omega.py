from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import nematics3d


DATA_PATH = Path(__file__).with_name("WT.npy")


def main():
    Q = np.load(DATA_PATH)

    q_obj = nematics3d.QFieldObject(
        Q=Q,
        name="WT",
    )
    q_obj.act_lines_smooth()
    q_obj.act_visualize_disclination_lines(
        min_line_length=0,
        title="WT disclination lines",
    )

    return q_obj, q_obj.figs.active_fig


if __name__ == "__main__":
    Q_OBJ, FIGURE = main()
