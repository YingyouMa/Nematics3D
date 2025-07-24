# Nematics3D

## Creator Information
Yingyou Ma, Brandeis, 2025  
https://github.com/YingyouMa
If you have any questions or suggestions, please contact:  yingyouma@brandeis.edu

## Introduction
The basic analysis of 3D uniaxial nematics.

### field.py
This provides the numerical analysis of the $Q$ field, including diagonalization, interpolation, handling periodic boundary conditions, visualization, and more.

### disclination.py
This provides the disclination analysis, including the detectation, topological analysis, visualization and more.

### elastic.py
This calculates the Frank deformation energy.

### coarse.py
This provides the coarse-graining $Q$ and $\rho$ field of my particle-based simulation of nematics.
This is not necessary if you've already got your continuum $Q$ field.

### lammps.py
For my particle-based simulation of nematics, this reads the dump files of `LAMMPS`.
This is not necessary if you've already got your continuum $Q$ field.

### defect2D.py
Defect analysis for 2D nematics.

## Dependencies
This package relies on fundamental scientific computing libraries. I personally use the following versions:
 - **Numpy**:       2.2.3
 - **SciPy**:       1.15.1
 - **matplotlib**:  3.10.0 

For 3D visualization, it uses ```Mayavi```:
 - **mayavi**:      4.8.2

Since I use the unpack operator in subscript notation, ```Python > 3.11``` is required. My personal setup:
 - **Python**:      3.12.9 

In `coarse.py`, the output data of coarse-grained field is stored using package `h5py`. My personal setup:
 - **h5py**:        3.11.0

In `lammps.py`, the dump files of lammps is read using package `pandas`. My personal setup:
 - **pandas**:      2.2.2

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

