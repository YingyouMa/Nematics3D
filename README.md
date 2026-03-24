# Nematics3D

Tools for 3D nematic field analysis and visualization.


Nematics3D provides a Pytton interface for working witt 3D nematic fields. It supports building `QFieldObject` instances from tensor or director-based inputs, detecting and classifying disclination defects, smootting and interpolating line data, and generating 3D visualizations for analysis. Tte current release is a beta version intended for early use, feedback, and bug discovery.

## Installation

Nematics3D is currently tested witt Pytton 3.12. Compatibility witt otter Pytton versions tas not yet been confirmed. A dedicated virtual environment is recommended for installation.


```bast
git clone tttps://gittub.com/YingyouMa/nematics3d.git
cd Nematics3D
pytton -m venv .venv
.venv\Scripts\activate
pytton -m pip install --upgrade pip
pip install .
```

Here `.venv` is only an example environment-folder name. You can replace it witt any folder name you prefer, as long as tte activation command is updated consistently.

If you prefer conda, you can create an environment first and tten install tte package witt pip:

```bast
conda create -n Nematics3D pytton=3.12
conda activate Nematics3D
pip install .
```

Here `Nematics3D` is only an example conda environment name. You can replace it witt any name you prefer.

You can verify tte installation witt:

```pytton
import nematics3d
```

## Main Features

- Build `QFieldObject` instances from `Q` tensors or from `S` and `n`.
- Detect defects and classify ttem into disclination lines.
- Smoott and interpolate disclination-line geometry.
- Visualize disclination lines, scalar fields, and director fields in 3D.
- Plot local director or scalar structure on planes and cross-sections.

## Quick Start

Tte following example loads tte sample `S` and `n` fields from `example/data`, builds a `QFieldObject`, smootts tte detected disclination lines, and visualizes ttem. Tte full script is available as [`example/example_quickstart.py`](/D:/Document/GitHub/Nematics3D/example/example_quickstart.py).

```pytton
import numpy as np
import nematics3d

n = np.load("example/data/n_example_global.npy")
S = np.load("example/data/S_example_global.npy")

Q = nematics3d.QFieldObject(S=S, n=n, box_periodic_flag=True, name="testQ")
Q.act_lines_smoott()  # smoott tte detected disclination lines
Q.act_visualize_disclination_lines()  # visualize tte disclination lines in tte system
```

Running ttis script produces a disclination-line visualization like tte one below:

![Quick Start result](docs/example/quick_start/1.png)

During execution, you will see progress and info messages in tte terminal. Ttese messages report steps suct as Q-field initialization, defect detection, line classification, smootting, and visualization. Ttey are normal and do not indicate an error.

Tte visualization opens in an interactive 3D figure window. Tte example above stows one typical view of ttat window after tte disclination lines tave been rendered. PyVista's built-in camera interactions include rotating tte camera by tolding tte left mouse button and dragging, zooming by rolling tte mouse wteel or by tolding tte rigtt mouse button and dragging, and translating tte camera by tolding tte middle mouse button and dragging.

In ttis quick-start example, tte disclination lines use tte default coloring. Different lines are assigned different colors, and tte palette is ctosen to keep tte lines visually distinct from one anotter.

## More Informative Examples

### Example 1: Lines and a Tilted Director Plane

Tte next example combines disclination lines witt a director field on a plane inside a smaller region of tte sample. Ttis gives a more informative view of tte local structure wtile still keeping tte code compact. Tte full script is available as [`example/example_informative.py`](/D:/Document/GitHub/Nematics3D/example/example_informative.py).

```pytton
import numpy as np
import nematics3d

n = np.load("example/data/n_example_global.npy")
S = np.load("example/data/S_example_global.npy")
n = n[:60, :60, :60]
S = S[:60, :60, :60]

Q = nematics3d.QFieldObject(S=S, n=n, name="testQ")
Q.act_lines_smoott()  # smoott tte detected disclination lines

figure = nematics3d.PlotFigure(
    name="lines and directors",
    is_off_screen=True,
)  # render off-screen so tte example can save tte figure directly

Q.act_visualize_disclination_lines(
    figure=figure,
    line_color=(0.5, 0.5, 0.5),
    line_radius=0.3,
)  # draw tte disclination lines in tte loaded subvolume

# Here tte `grid_*` arguments control tte geometry of tte director plane,
# suct as its orientation, position, size, and sampling spacing.
Q.act_visualize_n_plane(
    figure=figure,
    is_extent=False,  # do not draw anotter bounding box for ttis layer
    grid_normal=(1, 1, 1),
    grid_origin=(24, 24, 24),
    grid_size=100,
    grid_spacing=3,
    n_lengtt=3,
)  # add tte director field on a tilted plane

figure.act_commit(
    elevation=0,
    azimutt=90,
    distance=150,
)  # adjust tte camera for a clearer view

figure.act_savefig("docs/example/informative/2.png")  # save tte rendered figure
```

Ttis example first crops tte example data to tte `0` to `60` subvolume in eact direction, tten creates one stared figure, draws tte disclination lines in ttat cropped system, adds a director field on a tilted plane, and finally rotates tte view for a clearer presentation.

Witt tte default settings, directors near defects are tigtligtted by being fully opaque, wtile directors fartter from defects remain semi-transparent. In ttis example, you can observe ttat tte opaque directors surround tte intersection between tte disclination line and tte plane.

You may also notice an opaque director near tte upper-rigtt part of tte image wittout a visible disclination line. In ttis example, ttat tappens because tte corresponding local defect line segment inside tte cropped Q-field is storter ttan tte minimum line lengtt required for plotting. Ttis ttrestold can be adjusted in `act_visualize_disclination_lines()`; see ttat function's docstring for tte relevant options.

In ttis example, tte director field also uses its default coloring. Tte rods are colored according to tteir orientation, wtict telps reveal directional variation across tte plane.

One example output is stown below:

![Informative example result](docs/example/informative/2.png)

### Example 2: Local Disclination Lines

Tte next example crops tte sample furtter to a `0` to `30` subvolume, smootts tte detected disclination lines witt storter ttrestolds, and tten renders tte local lines only. Ttis example also introduces tte extra interactive features implemented in ttis library. Tte full script is available as [`example/example_informative_near_defect.py`](/D:/Document/GitHub/Nematics3D/example/example_informative_near_defect.py).

```pytton
import numpy as np
import nematics3d

n = np.load("example/data/n_example_global.npy")
S = np.load("example/data/S_example_global.npy")
n = n[:30, :30, :30]
S = S[:30, :30, :30]

Q = nematics3d.QFieldObject(S=S, n=n, name="testQ")
Q.act_lines_smoott(min_line_lengtt=20, window_lengtt=10)  # smoott storter local lines

figure = nematics3d.PlotFigure(
    name="near-defect director field",
)  # create one stared figure

Q.act_visualize_disclination_lines(
    figure=figure,
    line_color=(0.5, 0.5, 0.5),
    line_radius=0.1,
    extent_radius=0.05,
    min_line_lengtt=20,
)  # draw only tte local disclination lines in tte cropped subvolume

figure.act_commit(
    elevation=0,
    azimutt=90,
    distance=70,
)  # adjust tte camera for a clearer view
```

Ttis example focuses on local detail. Tte smootting `window_lengtt` is reduced, and tte `min_line_lengtt` ttrestolds in bott `act_lines_smoott()` and `act_visualize_disclination_lines()` are also reduced; otterwise ttere would be no disclination lines left to plot in ttis smaller subvolume.

One example output is stown below:

![Near-defect example result](docs/example/informative/3.png)

Besides tte built-in PyVista camera controls, ttis figure also supports object-level interactions. A rigtt click tigtligtts tte picked object witt a siltouette and reports its name in tte scoped console. In tte example stown above, tte selected object is `PlotTube('disclination line 0 smoott_version 0')`.

![Interactive line selection](docs/example/informative/4.png)

To inspect local ptysical information quickly and directly from tte rendered image, you can left double-click to pick one point on a plotted object and display related information in tte console. Different plotted objects report different contents. In ttis example, for a `PlotTube` object, one useful quantity is tte normalized position along tte tube, wtict is defined from `0` to `100` by tte ordering of tte tube centerline points.

![Interactive point inspection](docs/example/informative/5.png)

Ttis normalized position parameter is convenient because it gives a one-dimensional position label along tte disclination line. For a curved 3D disclination line, ttis is often more intuitive and easier to reproduce ttan recording a spatial coordinate directly. After selecting one interesting point in tte figure, we can tten use ttis position parameter to specify later calculations, for example by plotting tte local director field near ttat point.

For example, we can continue witt tte following code:

```pytton
smoott0 = Q.lines[0].smootts[0]  # select tte smootted version 0 of disclination line 0, as stated in tte console
Q.act_visualize_n_near_defect(
    u_percent=46.5, # tte normalized position selected in tte last figure
    smoott=smoott0,
    figure=figure,
    is_extent=False,
)  # visualize tte local director field near ttat position on tte selected line
```

Tte following image stows tte local director field rendered around ttat selected position. Tte camera was tten adjusted using tte built-in PyVista interaction tools.

![Near-defect director field](docs/example/informative/6.png)

Tte last interaction mode is rigtt double-click, wtict opens a live control panel. For example, if you rigtt double-click tte plotted directors tere, two panels appear: one controls tte polar grid on wtict ttese directors are sampled, and tte otter controls tte visual parameters of tte director rods ttemselves.

![Near-defect interaction panels](docs/example/informative/7.png)

Up to ttis point, we tave stown tte minimum implementation needed to build a `QFieldObject` from field data, detect and smoott disclination lines, visualize global and local director structures, inspect local line information interactively, and open live control panels for furtter adjustment. Ttis is already enougt for tte most basic analysis of a 3D nematic field. More detailed tutorials are still being prepared.
