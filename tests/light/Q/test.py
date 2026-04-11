"""Run the lightweight Q analysis on the bundled workflow example."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from q_defect_light import analyze_Q


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "example" / "data"

    q_path = data_dir / "Q_example_workflow.npy"

    print(f"Loading Q from: {q_path}")
    Q = np.load(q_path)
    print(f"Q shape: {Q.shape}")
    
    now = time.time()
    result = analyze_Q(Q)
    print(time.time()-now)

    print("\nLightweight analysis result")
    print(f"defect count: {result.defect_count}")
    print(f"defect line count: {result.line_count}")
    print(f"defect line lengths: {[len(line) for line in result.defect_lines]}")


if __name__ == "__main__":
    main()
