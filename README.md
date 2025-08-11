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

### elastic.py
This calculates the Frank deformation energy.

### defect2D.py
Defect analysis for 2D nematics.

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
The most basic physical value of nematics system is the tensor order parameter field $Q$. This is represented by ```QFieldObject``` in this package. Consider an object ```Q```, the digital value of each component of $Q$ could be extracted by ```Q()```. This is a ```np.ndarray``` with shape ```(Nx, Ny, Nz, 5)```. The first three dimensions represent the lattice grid of the 3D field, while the last one represent the 5 independent components of $Q$, $Q_{xx}$, $Q_{xy}$, $Q_{xz}$, $Q_{yy}$, $Q_{yz}$, seperately. Similarly, all space field variables in this package are ```np.ndarray``` with shape ```(Nx, Ny, Nz, ...)```. 

There are two ways to initialize an object of ```QFieldObject```. The first way is to input the $Q$ field directly as
```python
Q = Nematics3D.QFieldObject(Q=Q_data)
```
The other way is to input the scalar order parameter field $S$ and director field $n$, leading to $Q=S(nn-\frac{I}{3})$:
