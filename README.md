# Nematics3D

## Creator Information
Yingyou Ma, Brandeis, 2025  
https://github.com/YingyouMa
If you have any questions or suggestions, please contact:  yingyouma@outlook.com

## Introduction
The basic analysis of 3D uniaxial nematics.

### field.py
This provides the numerical analysis of the $Q$ field, including diagonalization, interpolation, handling periodic boundary conditions, visualization, and more.

### disclination.py
This provides the disclination analysis, including the detectation, topological analysis, visualization and more.

## Dependencies
This package relies on fundamental scientific computing libraries. I personally use the following versions:
 - **Numpy**:       2.3.2
 - **SciPy**:       1.16.0

For 3D visualization, it uses ```Mayavi```:
 - **mayavi**:      4.8.2

Since I use the unpack operator in subscript notation, ```Python > 3.11``` is required. My personal setup:
 - **Python**:      3.12.9 

## Installation
Currently, I manually place the package in a specific location and import it using a custom path.  
For example, in Windows, I'm applying:
```python
import sys
sys.path.insert(0, "WhereYouPutThePackage")
import Nematics3D
```
To install Mayavi using conda, run the following line on Anaconda Prompt
```
conda install -c conda-forge mayavi=4.8.2
```
A new environment specifically for ```Nematics3D```, or at least ```mayavi```, is highly recommended.

## Getting Started

### Q field
The most basic physical value of nematics system is the **tensor order parameter field** $Q$. In this package, this field is represented by the class ```QFieldObject```, which serves as the core data structure for storing and manipulating $Q$ in 3D space.   

Consider an object $Q$. ```Q()``` returns the numeric values of all components of $Q$ as a ```np.ndarray``` with shape ```(Nx, Ny, Nz, 5)```. The first three dimensions represent the lattice grid of the 3D field, while the last dimension stores the 5 independent components of $Q$: $Q_{xx}$, $Q_{xy}$, $Q_{xz}$, $Q_{yy}$, $Q_{yz}$, respectively. Similarly, all spatial fields in this package are represented as ```np.ndarray``` with shape ```(Nx, Ny, Nz, ...)```.

There are two ways to initialize an object of ```QFieldObject```.   

1. **Provide the $Q$ field directly**:
```python
Q = Nematics3D.QFieldObject(Q=Q_data)
```
The initialization automatically derive the scalar order parameter field $S$ and director field $n$, represented by ```Q.S``` and ```Q.n```. The input $Q$ data could either in the 5-component form ```(Nx, Ny, Nz, 5)``` or in tensorial structure ```(Nx, Ny, Nz, 9)```.    

2. **Provide $S$ and $n$**, in which case $Q$ is constructed as $Q = S,(nn-\frac{I}{3})$:
```python
Q = Nematics3D.QFieldObject(S=S_data, n=n_data)
```

Another important argument for ```QFieldObject``` is the periodic-boundary-condition flag. For example, ```box_periodic_flag=(True, False, False)``` indicates that only in $x$-direction there is periodic. This setting is crucial for disclination analysis: if PBC is specified incorrectly, a disclination line crossing a periodic boundary would be identified as multiple independent segments.

Finally, you may specify a transformation from lattice coordinates $g$ to real-space coordinates $r$ via $r = k g + b$, where $k$ is a linear transformation matrix (rotation/stretch) and $b$ is a translation vector as the offset. These are input via arguments ```grid_transform``` and ```grid_offset```. If they are not inputted, the real-space coordinates will directly use lattice coordinates as ```r=g```. The grid for the real-sapce coordinates is ```Q._calc_grid```.

### Defects detection and basic visualzation
All disclinations are automatically detected during initialization of ```QFieldObject```. You can further group them into individual smooth disclination lines via ```Q.act_lines_classify()``` and ```Q.act_lines_smooth()```, respectively. These lines are stored in ```Q.lines```, where each entry is a ```DisclinationLine``` object. For the smoothing options and parameters, see the docstrings of ```DisclinationLine.act_smooth()``` and ```QFieldObject.act_lines_smoothen()```.  Lines are sorted in descending order of length and named sequentially as ```line0```, ```line1```, and so on, according to their index.     

An example dataset of $S$ and $n$ field is provided under ```example/data```. The following code is an example of initialization of $Q$ field with disclination lines classified and smoothed:
```python
n = np.load( 'data/n_example_global.npy')
S = np.load( 'data/S_example_global.npy')

Q = Nematics3D.QFieldObject(S=S, n=n, box_periodic_flag=True)

Q.act_lines_classify()
Q.act_lines_smooth()
```

To visualize these disclination lines in $Q$ field, use ```Q.act_visualize_disclination_lines()```. The following is one example with the most important arguments input:
```python
Q.act_visualize_disclination_lines(min_line_length=20, lines_color_input_all=(1,0,0), radius=1, is_smooth=False)
```
* ```min_line_length``` ：the minimum length of disclinations to plot. This is set because tiny disclination loops are often below the coarse-grained resolution, making smoothing/visualization less meaningful.
* ```lines_color_input_all``` RGB color(s) in $[0,1]$ for the lines. . If this value is not set, the visualization will use the default colormap which tries to set those longest lines with distinct colors.
* ```radius```: tube radius used for rendering.
* ```is_smooth```: whether plot the smoothed line or the original line itself. ```is_smooth=False``` is helpful when you are adjusting the parameters of smoothing.
* Additional arguments are documented in the function’s docstring.

For instance, after setting up $Q$ as the example above, the following code    
```python
Q.act_visualize_disclination_lines()
Q.figs[0].save('figures/lines.png')
```
will produce the figure   
<p align="center">
  <img src="example/figures/lines.png" width="720">
</p>
and save it as ```figures/lines.png```.   

For the system with periodic boundary conditions, the disclination lines might cross boundaries or even the entire box. To comprehend this phenomena, you can disable line wrapping with flag ```is_wrap=False```:
```python
Q.act_visualize_disclination_lines(is_wrap=False)
Q.figs[1].save('figures/lines_unwrap.png')
```
<p align="center">
  <img src="example/figures/lines_unwrap.png" width="720">
</p>

### Simple introduction of figure structure and post-plot modification
All figures generated by a  ```QFieldObject``` are stored in ```Q.figs``` in chronological order. For example, the figures of wrapped and unwrapped disclination lines plotted in the previous section can be seperately addressed as ```Q.figs[0]``` and ```Q.figs[1]```. If we are going plot more figures, they could be derived by ```Q.figs[2]```, ```Q.figs[3]```, ...       

Within each figure with index ```idx```, the plotted elements are stored in the dictionary ```Q.figs[idex].objects['{item name}']```. In the current example of ```Q.figs[0]```, this includes ```Q.figs[0].objects['lines']``` and ```Q.figs[0].objects['extent']```, each of which is a list of the corresponding Mayavi objects. Besides, ```Q.figs[0].scene``` is the underlying ```figure``` object itself in ```mayavi``` . You can modify both the figure and its elements even after the plot has been created. For instance, the following example plots a new figure and then adjust properties
```python
Q.act_visualize_disclination_lines()
extent = Q.figs[2].objects['extent'][0]
extent.opacity = 0.5
extent.radius = 0.2
extent.color = [1,0,0]
scene = Q.figs[2].scene
scene.azimuth = 90
scene.elevation = 30
scene.roll = 30
scene.bgcolor = [0.5,0.5,0.5]
for line in Q.figs[2].objects['lines']:
    line.specular_power = 20
    line.specular_color = (1,0,0)
    line.radius = 2
    line.sides = 20
Q.figs[2].save('figures/lines_modified.png')
```
the figure correspondingly changes into
<p align="center">
  <img src="example/figures/lines_modified.png" width="720">
</p>

### Visualization of directors
It will be beneficial to plot disclination lines and directos in the same figure. You could visualize directors on a plane via ```Q.act_visualize_n_in_Q()```. Here we generate a new $Q$ field to focus on more local structure:
```python
index_max =  64
n = np.load( 'data/n_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
S = np.load( 'data/S_example_global.npy')[0:index_max, 0:index_max, 0:index_max]
Q = Nematics3D.QFieldObject(S=S, n=n)
Q.act_lines_classify()
Q.act_lines_smooth()

Q.act_visualize_disclination_lines(line_color=(0.5, 0.5, 0.5), extent_radius=0.1, line_radius=0.4)

n_length = 2.5
n_radius = 0.3
spacing = 2.5

Q.act_visualize_n_in_Q(plane_normal=(0,0,1), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), int(index_max)/2,0), 
                       n_length=2.5, n_radius=0.3, n_opacity=1, n_opacity_defect=1,
                       is_new=False, is_extent=False)
Q.figs[0].save('figures/PlotnPlaneZ.png')
```
<p align="center">
  <img src="example/figures/PlotnPlaneZ.png" width="720">
</p>

The following is the most significant parameters for ```Q.act_visualize_n_in_Q()```:   
* ```plane_normal```: the normal vector of your plane. Here "plane" stands for the plane you are going to plot directors on.    
* ```plane_spcaing```: the spcacing between neighboring directors    
* ```plane_size```: the side length (radius) of plane if it is square (circle)    
* ```plane_origin```: the position of center of the plane    
*  ```n_length```: the length of director glyph    
*  ```n_radius```: the radius of director glyph    
*  ```n_opacity```: the opacity of director glyph not closed to any defects
*  ```n_opacity_defect```: the opacity of director glyph closed to defects

Here we distinguish the directors by whether they are closed to a defect. The opacity of them is controlled by ```n_opacity``` and ```n_opacity_defect```. In our currrent example figure, we could not see the difference because they have the same opacity. The following is another example with different opacities:
```python
Q.act_visualize_disclination_lines(line_color=(0.5, 0.5, 0.5), extent_radius=0.1, line_radius=0.4)
Q.act_visualize_n_in_Q(plane_normal=(0,0,1), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), int(index_max)/2,0), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.act_visualize_n_in_Q(plane_normal=(0,1,0), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(int(index_max/2), 0, int(index_max)/2), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.act_visualize_n_in_Q(plane_normal=(1,0,0), plane_spacing=spacing, plane_size=0.95*index_max, plane_origin=(0, int(index_max/2), int(index_max)/2), 
                       n_length=2.5, n_opacity=0.2, n_radius=0.3, n_is_n_defect=True,
                       is_new=False, is_extent=False)
Q.figs[1].save('figures/PlotnPlaneXYZ.png')
```
<p align="center">
  <img src="example/figures/PlotnPlaneXYZ.png" width="720">
</p>
We descriminate directors 

### Logging function
Before moving on to additional features, it is helpful to introduce the logging mechanism provided in this package. Logging is enabled through the decorator ```logging_and_warning_decorator``` in ```logging_decorator.py```.    

Several functions in the package are wrapped with this decorator, which controls both the location and the verbosity of logging output. The key parameter is ```log_level```, which specifies the minimum severity of log messages to display. For example, the default setting ```log_level=logging.INFO``` will display only log messages at the ```info```, ```warning``` and ```error``` levels (note: this refers to manually defined ```warning``` calls within this package, not ```warning``` in existing external functions. You will need to ```logging``` before using this argument). Log messages at the ```debug``` level are ignored under this configuration.. 

As an example, the visualization function in Section **Defects detection and basic visualzation** produces the following log output when executed with its default settings
```python
[INFO]
        Start to defect defects
[INFO]
        Finished axis 0-direction in 0.2s
[INFO]
        Finished axis 1-direction in 0.13s
[INFO]
        Finished axis 2-direction in 0.14s
[INFO]
    No data of window_length is input for smoothening lines. 
    Use the default value 61
[INFO]
    No data of minimum line length is input for lines to be smoothened. 
    Use the default value 75
[INFO]
    No data of minimum line length is input for lines to be plotted. Use the default value 75
[INFO]
    No color data is input. Use the default color map, trying to set those longest lines with distinct colors
```
You can also include timestamps by setting ```show_timestamp=True```. This is helpful to further handle the time management. For example:
```python
@Nematics3D.logging_and_warning_decorator
def example_visualize(Q, logger=None):
    Q.update_defects(logger=logger)
    Q.update_lines_classify(logger=logger)
    Q.update_lines_smoothen(logger=logger)
    Q.visualize_disclination_lines(logger=logger)
    
example_visualize(Q, log_level=logging.DEBUG, show_timestamp=True)
```
will generate the logging information
```python
[DEBUG] - 2025-08-12 21:13:46
    Function `example_visualize` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:46
        Function `update_defects` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:46
            Function `defect_detect` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:46
            Threshold of the inner product between the first and last director is 0
[INFO] - 2025-08-12 21:13:46
            Start to defect defects
[DEBUG] - 2025-08-12 21:13:46
            Periodic boundary flags: [ True  True  True]
[INFO] - 2025-08-12 21:13:46
            Finished axis 0-direction in 0.19s
[INFO] - 2025-08-12 21:13:46
            Finished axis 1-direction in 0.13s
[INFO] - 2025-08-12 21:13:47
            Finished axis 2-direction in 0.13s
[DEBUG] - 2025-08-12 21:13:47
            Function `defect_detect` FINISHED in program `example_q.py`. Elapsed time: 0.470 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Function `update_defects` FINISHED in program `example_q.py`. Elapsed time: 0.472 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Function `update_lines_classify` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Function `defect_classify_into_lines` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            box_size_periodic: [128. 128. 128.]
[DEBUG] - 2025-08-12 21:13:47
            Start to find neighboring defects
[DEBUG] - 2025-08-12 21:13:47
            Start to perform Hierholzer algorithm
[DEBUG] - 2025-08-12 21:13:47
            Done!
[DEBUG] - 2025-08-12 21:13:47
            Function `defect_classify_into_lines` FINISHED in program `example_q.py`. Elapsed time: 0.400 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Function `update_lines_classify` FINISHED in program `example_q.py`. Elapsed time: 0.400 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Function `update_lines_smoothen` STARTED in program `example_q.py`
[INFO] - 2025-08-12 21:13:47
        No data of window_length is input for smoothening lines. 
        Use the default value 61
[INFO] - 2025-08-12 21:13:47
        No data of minimum line length is input for lines to be smoothened. 
        Use the default value 75
[DEBUG] - 2025-08-12 21:13:47
        Start to smoothen line0
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` FINISHED in program `example_q.py`. Elapsed time: 0.005 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Start to smoothen line1
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Start to smoothen line2
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Start to smoothen line3
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` FINISHED in program `example_q.py`. Elapsed time: 0.001 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Start to smoothen line4
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Start to smoothen line5
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Function `apply_smoothen` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Function `update_lines_smoothen` FINISHED in program `example_q.py`. Elapsed time: 0.007 seconds.
[DEBUG] - 2025-08-12 21:13:47
        Function `visualize_disclination_lines` STARTED in program `example_q.py`
[INFO] - 2025-08-12 21:13:47
        No data of minimum line length is input for lines to be plotted. Use the default value 75
[INFO] - 2025-08-12 21:13:47
        No color data is input. Use the default color map, trying to set those longest lines with distinct colors
[DEBUG] - 2025-08-12 21:13:47
        Start to draw disclination lines
[DEBUG] - 2025-08-12 21:13:47
            Function `visualize` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:47
            Start to visualize line0
[DEBUG] - 2025-08-12 21:13:47
                Function `PlotTube.__init__` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` FINISHED in program `example_q.py`. Elapsed time: 0.711 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` FINISHED in program `example_q.py`. Elapsed time: 0.714 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Start to visualize line1
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` FINISHED in program `example_q.py`. Elapsed time: 0.034 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` FINISHED in program `example_q.py`. Elapsed time: 0.034 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Start to visualize line2
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` FINISHED in program `example_q.py`. Elapsed time: 0.011 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` FINISHED in program `example_q.py`. Elapsed time: 0.012 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Start to visualize line3
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` FINISHED in program `example_q.py`. Elapsed time: 0.033 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` FINISHED in program `example_q.py`. Elapsed time: 0.033 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Start to visualize line4
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` FINISHED in program `example_q.py`. Elapsed time: 0.011 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` FINISHED in program `example_q.py`. Elapsed time: 0.012 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Start to visualize line5
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
                Function `PlotTube.__init__` FINISHED in program `example_q.py`. Elapsed time: 0.011 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `visualize` FINISHED in program `example_q.py`. Elapsed time: 0.011 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` STARTED in program `example_q.py`
[DEBUG] - 2025-08-12 21:13:48
            Function `add_object` FINISHED in program `example_q.py`. Elapsed time: 0.000 seconds.
[DEBUG] - 2025-08-12 21:13:48
        Function `visualize_disclination_lines` FINISHED in program `example_q.py`. Elapsed time: 1.375 seconds.
[DEBUG] - 2025-08-12 21:13:48
    Function `example_visualize` FINISHED in program `example_q.py`. Elapsed time: 2.254 seconds.
```
You may have noticed that the indentation of different log messages varies. This indentation is used to reflect the call hierarchy, making it easier to see when one function is invoked within another (its “parent” function).   

If the log output becomes too verbose, you can redirect it to a file by setting ```log_folder={AddressYouLike}, mode='file'```. The example of logging file from the aforementioned code is  ```example/log/example_visualize_20250812_211904_959843```, where the trailing numeric sequence encodes the exact date, time, and microsecond of file creation ```YYYYMMDD_HHMMSS_microseconds```. This ensures uniqueness timestamping. Finally, to disable all logging output entirely, set ```mode='none'```.
