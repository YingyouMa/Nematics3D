import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from elastic_test import get_deform_Q


def Plus(*args):
    result = 0
    for value in args:
        result = result + value
    return result


def Times(*args):
    result = 1
    for value in args:
        result = result * value
    return result


def Rational(a, b):
    return a / b


Sin = np.sin
Cos = np.cos
Power = np.power


N = int(os.environ.get("NEMATICS3D_ELASTIC_CHECK_N", "200"))
L = float(os.environ.get("NEMATICS3D_ELASTIC_CHECK_L", "2"))
SAMPLE_SIZE = int(os.environ.get("NEMATICS3D_ELASTIC_CHECK_SAMPLE_SIZE", "1000"))
SEED = int(os.environ.get("NEMATICS3D_ELASTIC_CHECK_SEED", "0"))

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

theta = 6 * X + 3 * Y**2 + Z**3
phi = X + Y + Z
n = np.array(
    [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]
).transpose((1, 2, 3, 0))

splay = Power(
    Plus(
        Times(6, Y, Cos(theta), Sin(phi)),
        Times(-1, Plus(Times(3, Power(Z, 2)), Sin(phi)), Sin(theta)),
        Times(Cos(phi), Plus(Times(6, Cos(theta)), Sin(theta))),
    ),
    2,
)[1:-1, 1:-1, 1:-1]

twist = Times(
    Rational(1, 4),
    Power(
        Plus(
            -1,
            Power(Cos(theta), 2),
            Times(12, Sin(phi)),
            Times(-1, Power(Sin(theta), 2)),
            Times(Sin(phi), Sin(Times(2, theta))),
            Times(Cos(phi), Plus(Times(-12, Y), Sin(Times(2, theta)))),
        ),
        2,
    ),
)[1:-1, 1:-1, 1:-1]

bend = Plus(
    Times(
        9,
        Power(Sin(theta), 2),
        Power(
            Plus(
                Times(Power(Z, 2), Cos(theta)),
                Times(2, Plus(Cos(phi), Times(Y, Sin(phi))), Sin(theta)),
            ),
            2,
        ),
    ),
    Power(
        Plus(
            Times(3, Power(Z, 2), Power(Cos(theta), 2), Sin(phi)),
            Times(
                Cos(theta),
                Plus(
                    Cos(phi),
                    Times(6, Cos(phi), Sin(phi)),
                    Times(6, Y, Power(Sin(phi), 2)),
                ),
                Sin(theta),
            ),
            Times(Cos(phi), Plus(Cos(phi), Sin(phi)), Power(Sin(theta), 2)),
        ),
        2,
    ),
    Power(
        Plus(
            Times(
                Sin(theta),
                Plus(
                    Times(Cos(theta), Plus(-6, Sin(phi), Times(6, Power(Sin(phi), 2)))),
                    Times(Power(Sin(phi), 2), Sin(theta)),
                ),
            ),
            Times(
                Cos(phi),
                Plus(
                    Times(-3, Power(Z, 2), Power(Cos(theta), 2)),
                    Times(-6, Y, Cos(theta), Sin(phi), Sin(theta)),
                    Times(Sin(phi), Power(Sin(theta), 2)),
                ),
            ),
        ),
        2,
    ),
)[1:-1, 1:-1, 1:-1]

deform_theory = {
    "splay": splay,
    "twist": twist,
    "bend": bend,
}

start = time.time()
print("analyzing with legacy get_deform_Q")
deform = get_deform_Q(n, 2 * L, 2)
elapsed = time.time() - start
print(f"done in {elapsed:.2f}s (N={N})")

rng = np.random.default_rng(SEED)
num_points = deform["splay"].size if isinstance(deform, dict) else deform[0].size
sample_size = min(SAMPLE_SIZE, num_points)
sample = rng.choice(num_points, size=sample_size, replace=False)

figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

for axis, (name, theory_values, numeric_values) in zip(
    axes,
    (
        ("splay", deform_theory["splay"], deform[0]),
        ("twist", deform_theory["twist"], deform[1]),
        ("bend", deform_theory["bend"], deform[2]),
    ),
):
    theory_sample = theory_values.reshape(-1)[sample]
    numeric_sample = numeric_values.reshape(-1)[sample]
    axis.plot(theory_sample, numeric_sample, "o", markersize=3, alpha=0.6)
    min_value = float(min(np.min(theory_sample), np.min(numeric_sample)))
    max_value = float(max(np.max(theory_sample), np.max(numeric_sample)))
    axis.plot([min_value, max_value], [min_value, max_value], "k-")
    axis.set_title(name)
    axis.set_xlabel("theory")
    axis.set_ylabel("legacy numeric")

    abs_err = np.abs(numeric_sample - theory_sample)
    print(
        f"{name}: "
        f"sample_max_abs_err={float(np.max(abs_err)):.6e}, "
        f"sample_mean_abs_err={float(np.mean(abs_err)):.6e}, "
        f"sample_rmse={float(np.sqrt(np.mean(abs_err**2))):.6e}"
    )

output_path = Path(__file__).with_name(f"legacy_scatter_N{N}.png")
figure.savefig(output_path, dpi=150)
print(f"saved scatter figure to {output_path}")
