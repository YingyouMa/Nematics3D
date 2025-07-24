# Read lammps dump file into pandas in Python.
# Return the information of each unit and simulation box.

from pathlib import Path

import numpy as np
import pandas as pd

import gzip


def read_lammps(lammps_file, columns='infer', dtype='infer'):
    fname = Path(lammps_file)
    if not fname.is_file():
        raise FileNotFoundError(f'Unable to open file {fname}')

    # a subset of the attributes that can be listed in a LAMMPS dump file
    dtype_defaults = {
        # identifiers
        'id': int,
        'mol': int,
        'type': int,
        'element': str,
        # properties
        'mass': float,
        'q': float,
        'radius': float,
        'diameter': float,
        # position
        'x': float,
        'y': float,
        'z': float,
        'xu': float,
        'yu': float,
        'zu': float,
        'xs': float,
        'ys': float,
        'zs': float,
        # velocity
        'vx': float,
        'vy': float,
        'vz': float,
        # force
        'fx': float,
        'fy': float,
        'fz': float,
        # dipole moment
        'mu': float,
        'mux': float,
        'muy': float,
        'muz': float,
        # torque
        'tqx': float,
        'tqy': float,
        'tqz': float,
        # angular velocity
        'omegax': float,
        'omegay': float,
        'omegaz': float,
        # angular momentum
        'angmomx': float,
        'angmomy': float,
        'angmomz': float,
    }

    box_bounds = {
        'x': None,
        'y': None,
        'z': None,
    }

    num_atoms = 0

    if fname.suffix == '.gz':
        def openfile(fin): return gzip.open(fin, "rt")
    else:
        def openfile(fin): return open(fin, "r")

    skip = 0
    with openfile(fname) as f:
        for line in f:
            line = line.strip()
            skip += 1

            if line.startswith('ITEM: NUMBER OF ATOMS'):
                num_atoms = int(f.readline().strip())
                skip += 1

            if line.startswith('ITEM: BOX BOUNDS'):
                for ax in 'xyz':
                    xlo, xhi = f.readline().strip().split()
                    box_bounds[ax] = [float(xlo), float(xhi)]
                    skip += 1

            if line.startswith('ITEM: ATOMS'):
                if columns == 'infer':
                    columns = line.split()[2:]
                if dtype == 'infer':
                    dtype = {col: dtype_defaults.get(col, object)
                             for col in columns}
                break

    data = pd.read_csv(fname, sep='\s+', names=columns, dtype=dtype,
                       skiprows=skip, header=None, index_col=False, 
                       nrows=num_atoms)

    return data, box_bounds
