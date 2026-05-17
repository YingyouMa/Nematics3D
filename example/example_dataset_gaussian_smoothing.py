import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nematics3d.classes.grid_field import GridFieldDataset, InputGridField


# This example shows how to smooth a shared-grid dataset field before taking
# spatial derivatives. The input here is synthetic so the script is standalone.

dataset = GridFieldDataset(
    inputValue=InputGridField(
        shape=(21, 21, 21),
        box_periodic_flag=(True, False, False),
        grid_transform=np.diag((1.0, 1.5, 2.0)),
    ),
    name="gaussian smoothing example dataset",
)

i, j, k = np.indices((21, 21, 21), dtype=float)
x = i
y = 1.5 * j
z = 2.0 * k

# A smooth background plus a localized peak gives the Gaussian smoother
# something visible to act on.
scalar = np.sin(2.0 * np.pi * x / 21.0) + 0.2 * np.cos(2.0 * np.pi * z / 42.0)
scalar[10, 10, 10] += 4.0

dataset.act_add_field("scalar", scalar)

smooth_result = dataset.act_gaussian_smooth(
    "scalar",
    sigma=2.0,
    coord="physical",
    is_result=True,
)
dataset.act_add_result_field("scalar_smooth", smooth_result)

grad_smooth = dataset.act_gradient("scalar_smooth", coord="physical")
lap_smooth = dataset.act_laplacian("scalar_smooth", coord="physical")

print("original field shape:", dataset["scalar"].raw_values.shape)
print("smoothed field shape:", dataset["scalar_smooth"].raw_values.shape)
print("gradient shape:", grad_smooth.shape)
print("laplacian shape:", lap_smooth.shape)
print("stored smoothing sigma:", smooth_result.raw_info.sigma)
print("stored smoothing sigma_index:", smooth_result.raw_info.sigma_index)
print("boundary modes:", smooth_result.raw_info.boundary)
print("center value before smoothing:", float(dataset["scalar"].raw_values[10, 10, 10]))
print(
    "center value after smoothing:",
    float(dataset["scalar_smooth"].raw_values[10, 10, 10]),
)
