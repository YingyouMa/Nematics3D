import numpy as np


class SmoothenedLine:
    '''
    Smoothen a line represented by coordinates using Savitzky-Golay filtering
    and cubic spline interpolation. Usually used for smoothening disclination lines.

    Parameters
    ----------
    line_coord : array_like, N x M
                 array containing the coordinates of the original line.
                 N is the number of coordinates, and M is the dimension.
    
    window_ratio : int, optional
                   Ratio to compute the Savitzky-Golay filter window length. 
                   window_length = number of coordinates (N) / window_ratio.
                   Default is 3.

    window_length: int, odd
                   If window_length is set directly, window_ratio would be ignored.
                   Default is None.

    order : int, optional
            Order of the Savitzky-Golay filter. 
            Default is 3.

    N_out_ratio : float, optional
                  Number of points in the output smoothened line compared to the original number of points. 
                  Default is 3.

    mode : str, optional
           Mode of extension for the Savitzky-Golay filter (usually 'interp' or 'wrap'). 
           Extension: Pad the signal with extension.
           Default is 'interp', with no extension.
           If 'wrap', the extension contains the values from the other end of the array.
           To smoothen loops (not crossing lines), "wrap' must be used.

    Returns
    -------
    self.output : numpy.ndarray, (N_out, M)
                  Array representing the smoothened line coordinates. 
                  N_out is the number of points in the smoothened line, and M is the dimension.

    All parameters are stored as private attributes of the returned instance.
                    
    Dependencies
    ------------
    - scipy: 1.7.3
    - numpy: 1.22.0   
    '''
    def __init__(self, line_coord,
                 window_ratio=3, window_length=None, order=3, N_out_ratio=3, mode='interp',
                 is_keep_origin=True):
        
        self._order = order
        self._N_out_ratio = N_out_ratio
        self._mode = mode
        self._N_init = len(line_coord)

        if window_length == None:
            self._window_length = int( self._N_init / window_ratio / 2 )*2 + 1
            self._window_ratio = self._N_init / self._window_length 
        else:
            self._window_length = window_length
            self._window_ratio = self._N_init / self._window_length

        self._N_out = int(self._N_init * N_out_ratio)

        
        if is_keep_origin:
            self._input = line_coord
        else:
            self._input = None

        # Apply Savitzky-Golay filter to smoothen the line
        from scipy.signal import savgol_filter
        line_points = savgol_filter(line_coord, self._window_length, order, axis=0, mode=mode)

        # Generate the parameter values for cubic spline interpolation
        uspline = np.arange(self._N_init) / self._N_init

        # Use cubic spline interpolation to obtain new coordinates
        from scipy.interpolate import splprep, splev
        tck = splprep(line_points.T, u=uspline, s=0)[0]
        self._output = np.array(splev(np.linspace(0,1,self._N_out), tck)).T

    def print_parameters(self):
        print(f'filter order: {self._order}')
        print(f'filter mode: {self._mode}')
        print(f'ratio between length of output and input: {self._N_out_ratio}')
        print(f'length of output: {self._N_out}')
        print(f'length of input: {self._N_init}')
        print(f'window length: {self._window_length}')
        print(f'ratio between length of input and window length: {self._window_ratio}')


