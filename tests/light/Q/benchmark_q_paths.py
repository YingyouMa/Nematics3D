"""Benchmark lightweight and object Q defect-analysis paths."""

from __future__ import annotations

import gc
from pathlib import Path
import statistics
import sys
import time
from typing import Callable, NamedTuple

import numpy as np

from q_defect_light import analyze_Q


class BenchmarkResult(NamedTuple):
    """Small summary for one timed benchmark path."""

    label: str
    times: list[float]
    defect_count: int
    line_count: int
    line_lengths: list[int]


def load_example_Q() -> np.ndarray:
    """Load the same bundled Q field used by test.py."""
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "example" / "data"
    q_path = data_dir / "Q_example_workflow.npy"

    print(f"Loading Q from: {q_path}")
    Q = np.load(q_path)
    print(f"Q shape: {Q.shape}")
    return Q


def benchmark(
    label: str,
    runner: Callable[[], tuple[int, int, list[int]]],
    repeats: int = 5,
    warmups: int = 1,
) -> BenchmarkResult:
    """Run a callable repeatedly and return timing plus result metadata."""
    for _ in range(warmups):
        runner()

    times: list[float] = []
    defect_count = 0
    line_count = 0
    line_lengths: list[int] = []

    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        defect_count, line_count, line_lengths = runner()
        times.append(time.perf_counter() - start)

    return BenchmarkResult(
        label=label,
        times=times,
        defect_count=defect_count,
        line_count=line_count,
        line_lengths=line_lengths,
    )


def summarize(result: BenchmarkResult) -> None:
    """Print a compact timing summary."""
    print(f"\n{result.label}")
    print(f"  median: {statistics.median(result.times):.6f} s")
    print(f"  min:    {min(result.times):.6f} s")
    print(f"  max:    {max(result.times):.6f} s")
    print(f"  all:    {[round(t, 6) for t in result.times]}")
    print(f"  defects: {result.defect_count}")
    print(f"  lines:   {result.line_count}")
    print(f"  lengths: {result.line_lengths}")


def compare_light_and_object(Q: np.ndarray) -> None:
    """Compare the numerical fields returned by both analysis paths."""
    import nematics3d

    light_result = analyze_Q(Q)
    q_obj = nematics3d.QFieldObject(
        Q=Q,
        name="comparison_Q",
        is_detect_defects=True,
        is_classify_lines=True,
        log_mode="none",
    )

    s_abs_diff = np.abs(light_result.S - q_obj.raw_S)
    director_abs_dot = np.abs(np.einsum("...i,...i->...", light_result.n, q_obj.raw_n))
    director_abs_dot_error = np.abs(1 - director_abs_dot)
    min_dot_index = tuple(
        int(index)
        for index in np.unravel_index(
            np.argmin(director_abs_dot), director_abs_dot.shape
        )
    )

    print("\nNumerical comparison")
    print("  S absolute difference")
    print(f"    max:  {float(np.max(s_abs_diff)):.6e}")
    print(f"    mean: {float(np.mean(s_abs_diff)):.6e}")
    print(f"    rms:  {float(np.sqrt(np.mean(s_abs_diff**2))):.6e}")
    print("  abs(dot(n_light, n_object))")
    print(f"    min:  {float(np.min(director_abs_dot)):.6e}")
    print(f"    mean: {float(np.mean(director_abs_dot)):.6e}")
    print(f"    p001: {float(np.quantile(director_abs_dot, 0.001)):.6e}")
    print(f"    p01:  {float(np.quantile(director_abs_dot, 0.01)):.6e}")
    print(f"    max |1 - abs(dot)|: {float(np.max(director_abs_dot_error)):.6e}")
    print(f"    count < 0.999: {int(np.sum(director_abs_dot < 0.999))}")
    print(f"    count < 0.99:  {int(np.sum(director_abs_dot < 0.99))}")
    print(f"    count < 0.9:   {int(np.sum(director_abs_dot < 0.9))}")
    print("    worst index:")
    print(f"      {min_dot_index}")
    print("    S at worst index:")
    print(f"      light={float(light_result.S[min_dot_index]):.6e}")
    print(f"      object={float(q_obj.raw_S[min_dot_index]):.6e}")
    print("  defect indices equal:")
    print(
        f"    {np.array_equal(light_result.defect_indices, q_obj.calc_defect_indices)}"
    )
    print("  line lengths equal:")
    light_line_lengths = [len(line) for line in light_result.defect_lines]
    object_line_lengths = [line.calc_defect_num for line in q_obj.lines]
    print(f"    {light_line_lengths == object_line_lengths}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import nematics3d

    Q = load_example_Q()

    def run_light() -> tuple[int, int, list[int]]:
        result = analyze_Q(Q)
        return (
            result.defect_count,
            result.line_count,
            [len(line) for line in result.defect_lines],
        )

    def run_object_full() -> tuple[int, int, list[int]]:
        q_obj = nematics3d.QFieldObject(
            Q=Q,
            name="benchmark_Q",
            is_detect_defects=True,
            is_classify_lines=True,
            log_mode="none",
        )
        return (
            len(q_obj.calc_defect_indices),
            len(q_obj.lines),
            [line.calc_defect_num for line in q_obj.lines],
        )

    results = [
        benchmark("q_defect_light.analyze_Q", run_light),
        benchmark("QFieldObject full init + defect + lines", run_object_full),
    ]

    for result in results:
        summarize(result)

    light_time = statistics.median(results[0].times)
    object_time = statistics.median(results[1].times)
    print("\nSpeed ratio")
    print(f"  object_full / light: {object_time / light_time:.3f}x")

    compare_light_and_object(Q)


if __name__ == "__main__":
    main()
