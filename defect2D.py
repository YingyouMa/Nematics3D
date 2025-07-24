# ---------------------------------
# Line integration for defect identification in nematics via convolution
# % ------------------------------------------------------------
# Michael M. Norton, Physics @ Rochester Institute of Technology, 2021
# in collaboration w/ Grover Lab (Piyush Grover and Caleb Wagner, Mech. Eng. @ University of Nebraska-Lincoln)
# ---------------------------------

# -----------------
# numerical basics
import numpy as np
import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# -----------------
# image convolution

from scipy import signal
from scipy import misc

# -----------------
# image processing

# from skimage.io import imread, imshow
# from skimage.filters import gaussian, threshold_otsu
# from skimage.measure import label, regionprops
# from skimage import measure
import math

def func_unitcircle(r):
    # just makes a ring of ones
    # https://stackoverflow.com/questions/39073973/how-to-generate-a-matrix-with-circle-of-ones-in-numpy-scipy
    d = 2*r + 1
    rx, ry = d/2, d/2
    x_grid, y_grid = np.indices((d, d))
    ring_filter = (np.abs(np.hypot(rx - x_grid, ry - y_grid)-r)
                   < 0.5).astype(int)

    x_grid = x_grid-d/2
    y_grid = y_grid-d/2

    r_grid = (x_grid**2+y_grid**2)**0.5

    x_grid[ring_filter == 0] = 0
    y_grid[ring_filter == 0] = 0

    filter_x = -y_grid/r_grid
    filter_y = x_grid/r_grid

    return (x_grid, y_grid, filter_x, filter_y, ring_filter)


def func_defectfind(nx, ny, filter_radius, switchsign):
    """
    func_defectfind(x,y,x0,y0)
    
    Function returns scalar maps the size of the input arrays that identify regions of topological charge
    
    Parameters
    ----------
    nx,ny : components of director field
    filter_radius : radius of line integral region
    switchsign : flips identity of defects +/-  --> -/+ (needed for some data sets)
    
    Returns
    -------
    map : line integral map
    map_p : map for +1/2 defects
    map_m : map for -1/2 defects
    """

    x_grid, y_grid, filter_x, filter_y, ring_filter = func_unitcircle(
        filter_radius)

    Qxx = nx**2-1/2
    Qxy = nx*ny

    Qxx_x, Qxx_y = np.gradient(Qxx)
    Qxy_x, Qxy_y = np.gradient(Qxy)

    denom = (1+4*Qxx+4*Qxy**2+4*Qxx**2)
    dphidx_num = 2*(-2*Qxy*Qxx_x+(1+2*Qxx)*Qxy_x)
    dphidy_num = 2*(-2*Qxy*Qxx_y+(1+2*Qxx)*Qxy_y)

    dphidx = dphidx_num/denom
    dphidy = dphidy_num/denom

    eps_mine = 1E0

    #remove ~0/0 artifacts
    dphidx[(np.abs(denom) < eps_mine) & (np.abs(dphidx_num) < eps_mine)] = 0
    dphidy[(np.abs(denom) < eps_mine) & (np.abs(dphidy_num) < eps_mine)] = 0  #

    map = signal.convolve2d(dphidy, filter_y, boundary='symm', mode='same') + \
        signal.convolve2d(dphidx, filter_x, boundary='symm', mode='same')

    Nrows, Ncolumns = np.shape(map)

    map_m = np.zeros((Nrows, Ncolumns))
    map_p = np.zeros((Nrows, Ncolumns))
    #map_zeros= np.zeros((Nrows, Ncolumns))
    #map_zeros[(map > -0.2) & (map < 0.2)] = 1

    #map_p[map > threshold_otsu(map)] = 1

    #auto-threshold using threshold_otsu acts funny.. just hard-code a threshold

    if switchsign == 1:
        map_m[map > 1] = 1
        map_p[map < -1] = 1
    else:
        map_m[map < -1] = 1
        map_p[map > 1] = 1

    return(map, map_p, map_m)


def func_defectpos(binmap, areathresh):
    """
    func_defectpos(binmap, areathresh)
    
    Function identifies coordinates of defects
    
    Parameters
    ----------
    binmap : logical map of candidate defect regions
    areathresh : area threshold, keep regions greater than threshold 
    
    Returns
    -------
    [[x1, y1],
    [x2, y2], et..]= list of defect coordinates     
    """

    '''
    binmap_label = label(binmap)
    regions = regionprops(binmap_label)

    # centroid_list = np.array([])
    centroid_xs = []
    centroid_ys = []
    for props in regions:
        y0, x0 = props.centroid
        area = props.area
        if area > areathresh:
            centroid_xs.append(x0)
            centroid_ys.append(y0)
    '''

            # centroid_list = np.append(centroid_list, [x0, y0])

    # N = int(np.size(centroid_list)/2)
    # centroid_list_reshape = np.reshape(centroid_list, [N, 2])

    '''
    centroid_list_reshape = np.array([centroid_xs, centroid_ys]).T
    return(centroid_list_reshape)
    '''


def func_defectorient(centroids, nx, ny, filter_radius, type_str):
    """
    func_defectorient(centroids, nx, ny, filter_radius,type_str)
    
    Function identifies defect orientation
    
    Parameters
    ----------
    centroids : logical map of candidate defect regions
    nx,ny : director field
    filter_radius : 
    type_str : string that indicates defect type: "positive" or "negative"
    
    Returns
    -------
    [phi1,phi2...phiN] =  list of defect angles [0,2pi]     
    """

    x_grid, y_grid, filter_x, filter_y, ring_filter = func_unitcircle(
        filter_radius)

    pos = np.argwhere(ring_filter > 0)
    x_ring = np.ceil(pos[:, 0] - filter_radius)
    y_ring = np.ceil(pos[:, 1] - filter_radius)

    theta = np.arctan2(y_ring, x_ring)

    theta_sort = np.sort(theta)
    theta_argsort = np.argsort(theta)

    x_ring = x_ring[theta_argsort]
    y_ring = y_ring[theta_argsort]

    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]

    N_defects = np.size(centroids_x)

    phi = np.zeros((N_defects, 1))

    Nrows, Ncolumns = np.shape(nx)

    for ii in range(0, N_defects):

        x0 = centroids_x[ii]
        y0 = centroids_y[ii]

        x = x0 + x_ring
        y = y0 + y_ring

        x[x > (Ncolumns-1)] = Ncolumns-1
        x[x < 0] = 0
        y[y > (Nrows-1)] = Nrows-1
        y[y < 0] = 0

        x = x.astype(int)
        y = y.astype(int)

        # note that to find a coordinate (x,y), need to select (row, column)
        nx_local = nx[y, x]
        ny_local = ny[y, x]

        dotprod = np.abs(nx_local*np.cos(theta_sort) +
                         ny_local*np.sin(theta_sort))

        if np.char.equal(type_str, "positive") == 1:
            phi[ii] = theta_sort[np.argmax(dotprod)]+np.pi
        elif np.char.equal(type_str, "negative") == 1:
            phi[ii] = theta_sort[np.argmax(dotprod)]
        else:
            print("check type_str")

    return(phi)


def func_plotdefects(ax, centroids, phi, color_str, type_str, scale):

    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]

    N_defects = np.size(centroids_x)

    # Flatten phi for convenient indexing below
    phi = phi.flatten()

    for ii in range(0, N_defects):

        x0 = centroids_x[ii]
        y0 = centroids_y[ii]

        x1 = x0 + scale*np.cos(phi[ii])
        y1 = y0 + scale*np.sin(phi[ii])

        ax.scatter(x0, y0, color=color_str)

        if np.char.equal(type_str, "positive") == 1:
            ax.plot((x0, x1), (y0, y1), color_str, linewidth=2.5)
        else:
            for jj in range(0, 3):
                x1 = x0 + scale*np.cos(phi[ii]+(jj+1)*2*np.pi/3)
                y1 = y0 + scale*np.sin(phi[ii]+(jj+1)*2*np.pi/3)
                ax.plot((x0, x1), (y0, y1), color_str, linewidth=2.5)

    return


# 2021.08.17: func_wrap() and func_crop() added to help with periodic boundary conditions

def func_wrap(A, periodic_x, periodic_y, padamount):

    if periodic_x == 1:
        A_pad = A[:, 0:padamount]
        A = np.hstack((A, A_pad))
        if periodic_y == 1:
            A_pad = A[0:padamount, :]
            A = np.vstack((A, A_pad))

    elif periodic_y == 1:
        A_pad = A[0:padamount, :]
        A = np.vstack((A, A_pad))

    return(A)


def func_crop(A, periodic_x, periodic_y, padamount):

    if periodic_x == 1:
        A = A[:, 0:-padamount]
        if periodic_y == 1:
            A = A[0:-padamount, :]

    elif periodic_y == 1:
        A = A[0:-padamount, :]

    return(A)


__all__ = ["nematic_plot"]

def get_streamlines(x,
                 y,
                 u,
                 v,
                 ax=None,
                 density=1,
                 linewidth=None,
                 color=None,
                 cmap=None,
                 norm=None,
                 transform=None,
                 zorder=None,
                 start_points=None,
                 minlength=0.1,
                 maxlength=4.0,
                 integration_direction='both'):

    grid = Grid(x, y)
    mask = StreamMask(density)
    dmap = DomainMap(grid, mask)

    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']

    line_kw = {}

    if integration_direction not in ['both', 'forward', 'backward']:
        errstr = ("Integration direction '%s' not recognised. "
                  "Expected 'both', 'forward' or 'backward'." %
                  integration_direction)
        raise ValueError(errstr)

    if integration_direction == 'both':
        maxlength /= 2.

    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        if color.shape != grid.shape:
            raise ValueError(
                "If 'color' is given, must have the shape of 'Grid(x,y)'")
        line_colors = []
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError(
                "If 'linewidth' is given, must have the shape of 'Grid(x,y)'")
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder

    # Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must be of shape 'Grid(x,y)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = get_integrator(u, v, dmap, minlength, maxlength,
                               integration_direction)

    trajectories = []
    if start_points is None:
        for xm, ym in _gen_starting_points(mask.shape):
            if mask[ym, xm] == 0:
                xg, yg = dmap.mask2grid(xm, ym)
                t = integrate(xg, yg)
                if t is not None:
                    trajectories.append(t)
    else:
        sp2 = np.asanyarray(start_points, dtype=float).copy()

        # Check if start_points are outside the data boundaries
        for xs, ys in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                    and grid.y_origin <= ys <= grid.y_origin + grid.height):
                raise ValueError("Starting point ({}, {}) outside of data "
                                 "boundaries".format(xs, ys))

        # Convert start_points from data to array coords
        # Shift the seed points from the bottom left of the data so that
        # data2grid works properly.
        sp2[:, 0] -= grid.x_origin
        sp2[:, 1] -= grid.y_origin

        for xs, ys in sp2:
            xg, yg = dmap.data2grid(xs, ys)
            t = integrate(xg, yg)
            if t is not None:
                trajectories.append(t)

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        if cmap is None:
            cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
        else:
            cmap = cm.get_cmap(cmap)

    streamlines = []
    for t in trajectories:
        tgx = np.array(t[0])
        tgy = np.array(t[1])
        # Rescale from grid-coordinates to data-coordinates.
        tx, ty = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin

        points = np.transpose([tx, ty]).reshape(-1, 1, 2)
        streamlines.extend(np.hstack([points[:-1], points[1:]]))

        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
            line_kw['linewidth'].extend(line_widths)

        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy)[:-1]
            line_colors.append(color_values)

    return streamlines


def nematic_plot(x,
                 y,
                 u,
                 v,
                 ax=None,
                 density=1,
                 linewidth=None,
                 color=None,
                 cmap=None,
                 norm=None,
                 transform=None,
                 zorder=None,
                 start_points=None,
                 minlength=0.1,
                 maxlength=4.0,
                 integration_direction='both'):
    """
    Draw the streamlines of a nematic vector field.

    *x*, *y* : 1d arrays
        an *evenly spaced* grid.
    *u*, *v* : 2d arrays
        x- and y-components of the vector field. Number of rows should match
        length of y, and the number of columns should match x.
    *ax* : :class:`~matplotlib.axes.Axes`
        The Axes object in which to put the plot. Defaults to the current
        active axes.
    *density* : float or 2-tuple
        Controls the closeness of streamlines. When `density = 1`, the domain
        is divided into a 30x30 grid---*density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use [density_x, density_y].
    *linewidth* : numeric or 2d array
        vary linewidth when given a 2d array with the same shape as velocities.
    *color* : matplotlib color code, or 2d array
        Streamline color. When given an array with the same shape as
        velocities, *color* values are converted to colors using *cmap*.
    *cmap* : :class:`~matplotlib.colors.Colormap`
        Colormap used to plot streamlines and arrows. Only necessary when using
        an array input for *color*.
    *norm* : :class:`~matplotlib.colors.Normalize`
        Normalize object used to scale luminance data to 0, 1. If None, stretch
        (min, max) to (0, 1). Only necessary when *color* is an array.
    *start_points*: Nx2 array
        Coordinates of starting points for the streamlines.
        In data coordinates, the same as the ``x`` and ``y`` arrays.
    *zorder* : int
        any number
    *minlength* : float
        Minimum length of streamline in axes coordinates.
    *maxlength* : float
        Maximum length of streamline in axes coordinates.
    *integration_direction* : ['forward', 'backward', 'both']
        Integrate the streamline in forward, backward or both directions.

    Returns:

        *stream_container* : StreamplotSet
            Container object with attributes

                - lines: `matplotlib.collections.LineCollection` of streamlines

                - arrows: collection of `matplotlib.patches.FancyArrowPatch`
                  objects representing arrows half-way along stream
                  lines.

            This container will probably change in the future to allow changes
            to the colormap, alpha, etc. for both lines and arrows, but these
            changes should be backward compatible.

    """
    grid = Grid(x, y)
    mask = StreamMask(density)
    dmap = DomainMap(grid, mask)

    axes = ax or plt.gca()

    if zorder is None:
        zorder = mlines.Line2D.zorder

    # default to data coordinates
    if transform is None:
        transform = axes.transData

    if color is None:
        color = axes._get_lines.get_next_color()

    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']

    line_kw = {}

    if integration_direction not in ['both', 'forward', 'backward']:
        errstr = ("Integration direction '%s' not recognised. "
                  "Expected 'both', 'forward' or 'backward'." %
                  integration_direction)
        raise ValueError(errstr)

    if integration_direction == 'both':
        maxlength /= 2.

    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        if color.shape != grid.shape:
            raise ValueError(
                "If 'color' is given, must have the shape of 'Grid(x,y)'")
        line_colors = []
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError(
                "If 'linewidth' is given, must have the shape of 'Grid(x,y)'")
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder

    # Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must be of shape 'Grid(x,y)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)

    integrate = get_integrator(u, v, dmap, minlength, maxlength,
                               integration_direction)

    trajectories = []
    if start_points is None:
        for xm, ym in _gen_starting_points(mask.shape):
            if mask[ym, xm] == 0:
                xg, yg = dmap.mask2grid(xm, ym)
                t = integrate(xg, yg)
                if t is not None:
                    trajectories.append(t)
    else:
        sp2 = np.asanyarray(start_points, dtype=float).copy()

        # Check if start_points are outside the data boundaries
        for xs, ys in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                    and grid.y_origin <= ys <= grid.y_origin + grid.height):
                raise ValueError("Starting point ({}, {}) outside of data "
                                 "boundaries".format(xs, ys))

        # Convert start_points from data to array coords
        # Shift the seed points from the bottom left of the data so that
        # data2grid works properly.
        sp2[:, 0] -= grid.x_origin
        sp2[:, 1] -= grid.y_origin

        for xs, ys in sp2:
            xg, yg = dmap.data2grid(xs, ys)
            t = integrate(xg, yg)
            if t is not None:
                trajectories.append(t)

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        if cmap is None:
            cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
        else:
            cmap = cm.get_cmap(cmap)

    streamlines = []
    for t in trajectories:
        tgx = np.array(t[0])
        tgy = np.array(t[1])
        # Rescale from grid-coordinates to data-coordinates.
        tx, ty = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin

        points = np.transpose([tx, ty]).reshape(-1, 1, 2)
        streamlines.extend(np.hstack([points[:-1], points[1:]]))

        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
            line_kw['linewidth'].extend(line_widths)

        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy)[:-1]
            line_colors.append(color_values)

    lc = mcollections.LineCollection(streamlines,
                                     transform=transform,
                                     **line_kw)
    lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
    lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
    if use_multicolor_lines:
        lc.set_array(np.ma.hstack(line_colors))
        lc.set_cmap(cmap)
        lc.set_norm(norm)
    axes.add_collection(lc)
    axes.autoscale_view()

    stream_container = StreamplotSet(lc)
    return stream_container, streamlines


class StreamplotSet(object):
    def __init__(self, lines, **kwargs):
        self.lines = lines


# Coordinate definitions
# ========================


class DomainMap(object):
    """Map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """
    def __init__(self, grid, mask):
        self.grid = grid
        self.mask = mask
        # Constants for conversion between grid- and mask-coordinates
        self.x_grid2mask = (mask.nx - 1) / grid.nx
        self.y_grid2mask = (mask.ny - 1) / grid.ny

        self.x_mask2grid = 1. / self.x_grid2mask
        self.y_mask2grid = 1. / self.y_grid2mask

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy

    def grid2mask(self, xi, yi):
        """Return nearest space in mask-coords from given grid-coords."""
        return (int((xi * self.x_grid2mask) + 0.5),
                int((yi * self.y_grid2mask) + 0.5))

    def mask2grid(self, xm, ym):
        return xm * self.x_mask2grid, ym * self.y_mask2grid

    def data2grid(self, xd, yd):
        return xd * self.x_data2grid, yd * self.y_data2grid

    def grid2data(self, xg, yg):
        return xg / self.x_data2grid, yg / self.y_data2grid

    def start_trajectory(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._start_trajectory(xm, ym)

    def reset_start_point(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._current_xy = (xm, ym)

    def update_trajectory(self, xg, yg):
        if not self.grid.within_grid(xg, yg):
            raise InvalidIndexError
        xm, ym = self.grid2mask(xg, yg)
        self.mask._update_trajectory(xm, ym)

    def undo_trajectory(self):
        self.mask._undo_trajectory()


class Grid(object):
    """Grid of data."""
    def __init__(self, x, y):

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x_row = x[0, :]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y_col = y[:, 0]
            if not np.allclose(y_col, y.T):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_origin = x[0]
        self.y_origin = y[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

    @property
    def shape(self):
        return self.ny, self.nx

    def within_grid(self, xi, yi):
        """Return True if point is a valid index of grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since `xi` can be `self.nx - 1 < xi < self.nx`
        return xi >= 0 and xi <= self.nx - 1 and yi >= 0 and yi <= self.ny - 1


class StreamMask(object):
    """Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """
    def __init__(self, density):
        if np.isscalar(density):
            if density <= 0:
                raise ValueError("If a scalar, 'density' must be positive")
            self.nx = self.ny = int(30 * density)
        else:
            if len(density) != 2:
                raise ValueError("'density' can have at maximum 2 dimensions")
            self.nx = int(30 * density[0])
            self.ny = int(30 * density[1])
        self._mask = np.zeros((self.ny, self.nx))
        self.shape = self._mask.shape

        self._current_xy = None

    def __getitem__(self, *args):
        return self._mask.__getitem__(*args)

    def _start_trajectory(self, xm, ym):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask.__setitem__(t, 0)

    def _update_trajectory(self, xm, ym):
        """Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        if self._current_xy != (xm, ym):
            if self[ym, xm] == 0:
                self._traj.append((ym, xm))
                self._mask[ym, xm] = 1
                self._current_xy = (xm, ym)
            else:
                raise InvalidIndexError


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
# ======================


def get_integrator(u, v, dmap, minlength, maxlength, integration_direction):

    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)

    # speed (path length) will be in axes-coordinates
    u_ax = u / dmap.grid.nx
    v_ax = v / dmap.grid.ny
    speed = np.ma.sqrt(u_ax**2 + v_ax**2)

    def forward_time(xi, yi):
        ds_dt = interpgrid(speed, xi, yi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui, vi = interpgrid_vec(u, v, xi, yi)
        # vi = interpgrid_vec(v, xi, yi)

        return ui * dt_ds, vi * dt_ds

    def backward_time(xi, yi):
        dxi, dyi = forward_time(xi, yi)
        return -dxi, -dyi

    def integrate(x0, y0):
        """Return x, y grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, x_traj, y_traj = 0., [], []

        try:
            dmap.start_trajectory(x0, y0)
        except InvalidIndexError:
            return None
        if integration_direction in ['both', 'backward']:
            s, xt, yt = _integrate_rk12(x0, y0, dmap, backward_time, maxlength)
            stotal += s
            x_traj += xt[::-1]
            y_traj += yt[::-1]

        if integration_direction in ['both', 'forward']:
            dmap.reset_start_point(x0, y0)
            s, xt, yt = _integrate_rk12(x0, y0, dmap, forward_time, maxlength)
            if len(x_traj) > 0:
                xt = xt[1:]
                yt = yt[1:]
            stotal += s
            x_traj += xt
            y_traj += yt

        if stotal > minlength:
            return x_traj, y_traj
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return None

    return integrate


def _integrate_rk12(x0, y0, dmap, f, maxlength):
    """
    2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.003

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.1)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xf_traj = []
    yf_traj = []

    # previous step's derivatives
    tangent = (0, 0)

    while dmap.grid.within_grid(xi, yi):
        xf_traj.append(xi)
        yf_traj.append(yi)
        try:
            k1x, k1y = force_parallel(f(xi, yi), tangent)
            k2x, k2y = force_parallel(f(xi + ds * k1x, yi + ds * k1y), tangent)
        except IndexError:
            # Out of the domain on one of the intermediate integration steps.
            # Take an Euler step to the boundary to improve neatness.

            ds, xf_traj, yf_traj = _euler_step(xf_traj, yf_traj, dmap, f,
                                               tangent)
            stotal += ds
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)

        nx, ny = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.sqrt(((dx2 - dx1) / nx)**2 + ((dy2 - dy1) / ny)**2)

        # Only save step if within error tolerance
        if error < maxerror:
            # dx2, dy2 = force_parallel((dx2, dy2),

            xi += dx2
            yi += dy2

            tangent = (dx2, dy2)

            try:
                dmap.update_trajectory(xi, yi)
            except InvalidIndexError:
                break
            if stotal + ds > maxlength:
                break
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error)**0.5)

    return stotal, xf_traj, yf_traj


def _euler_step(xf_traj, yf_traj, dmap, f, tangent):
    """
    Simple Euler integration step that extends streamline to boundary.
    """

    ny, nx = dmap.grid.shape
    xi = xf_traj[-1]
    yi = yf_traj[-1]
    cx, cy = f(xi, yi)

    cx, cy = force_parallel((cx, cy), tangent)

    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx

    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    ds = min(dsx, dsy)
    xf_traj.append(xi + cx * ds)
    yf_traj.append(yi + cy * ds)
    return ds, xf_traj, yf_traj


# Utility functions
# ========================


def force_parallel(u, v):
    if u[0] * v[0] + u[1] * v[1] < 0:
        return (-u[0], -u[1])
    else:
        return u

    # u2 = u[0] * u[0] + u[1] * u[1]
    # v2 = v[0] * v[0] + v[1] * v[1]

    # if v2 == 0:
    #     cos = 1
    # elif u2 == 0:
    #     raise TerminateTrajectory
    # else:
    #     cos = (u[0] * v[0] + u[1] * v[1]) / np.sqrt(u2 * v2)

    # if cos > 0.1:
    #     return u
    # elif cos < -0.1:
    #     return (-u[0], -u[1])
    # else:
    #     raise TerminateTrajectory


def interpgrid(a, xi, yi):
    """
    Fast 2D, linear interpolation on an integer grid
    """

    Ny, Nx = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
    else:
        x = int(xi)
        y = int(yi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1

    a00 = a[y, x]
    a01 = a[y, xn]
    a10 = a[yn, x]
    a11 = a[yn, xn]
    xt = xi - x
    yt = yi - y
    a0 = a00 * (1 - xt) + a01 * xt
    a1 = a10 * (1 - xt) + a11 * xt
    ai = a0 * (1 - yt) + a1 * yt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ai):
            raise TerminateTrajectory

    return ai


def interpgrid_vec(u, v, xi, yi):
    """
    Fast 2D, linear interpolation of nematic vector on an integer grid
    """

    Ny, Nx = np.shape(u)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
    else:
        x = int(xi)
        y = int(yi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1

    xt = xi - x
    yt = yi - y

    u00 = u[y, x]
    u01 = u[y, xn]
    u10 = u[yn, x]
    u11 = u[yn, xn]

    v00 = v[y, x]
    v01 = v[y, xn]
    v10 = v[yn, x]
    v11 = v[yn, xn]

    u01, v01 = force_parallel((u01, v01), (u00, v00))
    u10, v10 = force_parallel((u10, v10), (u00, v00))
    u11, v11 = force_parallel((u11, v11), (u00, v00))

    u0 = u00 * (1 - xt) + u01 * xt
    u1 = u10 * (1 - xt) + u11 * xt
    ui = u0 * (1 - yt) + u1 * yt

    v0 = v00 * (1 - xt) + v01 * xt
    v1 = v10 * (1 - xt) + v11 * xt
    vi = v0 * (1 - yt) + v1 * yt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ui) or np.ma.is_masked(vi):
            raise TerminateTrajectory

    return (ui, vi)


def _gen_starting_points(shape):
    """Yield starting points for streamlines.

    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    ny, nx = shape
    xfirst = 0
    yfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    x, y = 0, 0
    i = 0
    direction = 'right'
    for i in range(nx * ny):

        yield x, y

        if direction == 'right':
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = 'up'
        elif direction == 'up':
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = 'left'
        elif direction == 'left':
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = 'down'
        elif direction == 'down':
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = 'right'

'''
height, activity, K = 10, 4.0, 7.0

data_dir = f"/scratch1/saptorshighosh/may17/images_{height}_{activity}_{K}"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
nopy = False

x = np.linspace(0, 100, 200)
y = np.linspace(0, 100, 200)

dx = x[1]-x[0]
dy = y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing="ij")

nx_t, ny_t, s_t = np.load(f"/scratch1/saptorshighosh/may17/find_defect_{activity}_{K}.npz")['nx'], np.load(f"/scratch1/saptorshighosh/may17/find_defect_{activity}_{K}.npz")['ny'], np.load(f"/scratch1/saptorshighosh/may17/find_defect_{activity}_{K}.npz")['S']

for i in range(35):

    nx, ny, S = nx_t[i], ny_t[i], s_t[i]
    # create charge density map
    _, map_p, map_m = func_defectfind(nx, ny, filter_radius=5, switchsign=0)

    # search map and identify circular regions of positive and negative charge
    centroids_p = func_defectpos(map_p, areathresh=60)
    centroids_m = func_defectpos(map_m, areathresh=60)

    # get the oriengation of defects
    phi_p = func_defectorient(centroids_p, nx, ny, filter_radius=5, type_str="positive")
    phi_m = func_defectorient(centroids_m, nx, ny, filter_radius=5, type_str="negative")

    fig, ax = plt.subplots()
    nematic_plot(x, y, nx, ny, density=2.0)
    #ax.imshow(fluorescence, cmap=plt.cm.gray)
    ax.set_aspect('equal', adjustable='box')
    color_p = 'magenta'
    color_m = 'cyan'
    defect_scale = 1

    cp = centroids_p*dx
    cm = centroids_m*dx
    func_plotdefects(ax, cp, phi_p, color_p, "positive", defect_scale)
    func_plotdefects(ax, cm, phi_m, color_m, "negative", defect_scale)

    c = plt.pcolor(x, y, S)
    plt.colorbar(c)
    nematic_plot(x, y, nx, ny, density=2, color = 'gray')
    plt.axhline(y = (50 + (height/2)), color = 'r', linestyle = '--')
    plt.axhline(y = (50 - (height/2)), color = 'r', linestyle = '--')
    plt.xlabel(r"X")
    plt.ylabel(r"Y")

    plt.tight_layout()
    plt.savefig(f"{data_dir}/defect_{i}.png", dpi = 200)
    plt.show()

src = mlab.pipeline.scalar_scatter(x, y, z)

# Connect them
src.mlab_source.dataset.lines = connections
src.update()

# The stripper filter cleans up connected lines
lines = mlab.pipeline.stripper(src)

# Finally, display the set of lines
mlab.pipeline.surface(lines, line_width=4, opacity=.4)
'''