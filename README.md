# Nematics3D

Tools for 3D nematic field analysis and visualization.


Nematics3D provides a Python interface for working with 3D nematic fields. It supports building `QFieldObject` instances from tensor or director-based inputs, detecting and classifying disclination defects, smoothing and interpolating line data, and generating 3D visualizations for analysis. The current release is a beta version intended for early use, feedback, and bug discovery.

## Installation

Nematics3D is currently tested with Python 3.12. A dedicated virtual environment is recommended for installation.


```bash
git clone https://github.com/YingyouMa/Nematics3D.git
cd Nematics3D
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install .
```

If you prefer conda, you can create an environment first and then install the package with pip:

```bash
conda create -n Nematics3D python=3.12
conda activate Nematics3D
pip install .
```

You can verify the installation with:

```python
import Nematics3D
```

## Main Features

- Build `QFieldObject` instances from `Q` tensors or from `S` and `n`.
- Detect defects and classify them into disclination lines.
- Smooth and interpolate disclination-line geometry.
- Visualize disclination lines, scalar fields, and director fields in 3D.
- Plot local director or scalar structure on planes and cross-sections.

## Quick Start

The following example loads the sample `S` and `n` fields from `example/data`, builds a `QFieldObject`, smooths the detected disclination lines, and visualizes them. The full script is available as [`example/example_quickstart.py`](/D:/Document/GitHub/Nematics3D/example/example_quickstart.py).

```python
import numpy as np
import Nematics3D

n = np.load("example/data/n_example_global.npy")
S = np.load("example/data/S_example_global.npy")

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True, name="testQ")
Q.act_lines_smooth()  # smooth the detected disclination lines
Q.act_visualize_disclination_lines()  # visualize the disclination lines in the system
```

Running this script produces a disclination-line visualization like the one below:

![Quick Start result](docs/example/quick_start/1.png)

During execution, you will see progress and info messages in the terminal. These messages report steps such as Q-field initialization, defect detection, line classification, smoothing, and visualization. They are normal and do not indicate an error.

The visualization opens in an interactive 3D figure window. The example above shows one typical view of that window after the disclination lines have been rendered.

## More Informative Examples

### Example 1: Lines and a Tilted Director Plane

The next example combines disclination lines with a director field on a plane inside a smaller region of the sample. This gives a more informative view of the local structure while still keeping the code compact. The full script is available as [`example/example_informative.py`](/D:/Document/GitHub/Nematics3D/example/example_informative.py).

```python
import numpy as np
import Nematics3D

n = np.load("example/data/n_example_global.npy")
S = np.load("example/data/S_example_global.npy")

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True, name="testQ")
Q.act_lines_smooth()  # smooth the detected disclination lines

bounds_max = 60
bounds = Nematics3D.as_bounds((0, bounds_max, 0, bounds_max, 0, bounds_max))  # focus on a smaller region
figure = Nematics3D.PlotFigure(
    name="lines and directors",
    is_off_screen=True,
)  # create one shared figure in off-screen mode for direct saving

Q.act_visualize_disclination_lines(
    figure=figure,
    bounds=bounds,
    line_color=(0.5, 0.5, 0.5),
    line_radius=0.3,
)  # draw the disclination lines inside the selected box

# Here the `grid_*` arguments control the geometry of the director plane,
# such as its orientation, position, size, and sampling spacing.
Q.act_visualize_n_plane(
    figure=figure,
    bounds=bounds,
    is_extent=False,  # do not draw another bounding box for this layer
    grid_normal=(1, 1, 1),
    grid_origin=(24, 24, 24),
    grid_size=100,
    grid_spacing=3,
    n_length=3,
)  # add the director field on a tilted plane

figure.act_commit(
    elevation=0,
    azimuth=90,
    distance=150,
)  # adjust the camera for a clearer view

figure.act_savefig("docs/example/informative/2.png")  # save the rendered figure
```

This example creates one shared figure, restricts the view to a `0` to `60` box, draws the disclination lines inside that box, adds a director field on a tilted plane through the same region, and then rotates the view for a clearer presentation.

With the default settings, directors near defects are highlighted by being fully opaque, while directors farther from defects remain semi-transparent. In this example, you can observe that the opaque directors surround the intersection between the disclination line and the plane.

During detailed tuning, you may sometimes see opaque directors without a visible disclination line. This is usually caused by visualization choices rather than by an inconsistency in the data, for example the minimum line length chosen for plotting or the smoothing settings used for the line. We explain these details in the corresponding function-level documentation.

One example output is shown below:

![Informative example result](docs/example/informative/2.png)
