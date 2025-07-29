import numpy as np
import time
from typing import Tuple, Optional

from ..logging_decorator import logging_and_warning_decorator, Logger
from ..datatypes import Vect3D, QField, as_QField5, SField, nField, DimensionPeriodic, boundary_periodic_size_to_flag 
from ..field import diagonalizeQ, getQ, add_periodic_boundary, generate_coordinate_grid


class QFieldObject:
    """
    A data container and utility class for representing and manipulating a Q-tensor field in 3D space.

    The object can be initialized in two mutually exclusive ways:
    1. Provide `n` (director field), with optional `S` (scalar order parameter). The Q-tensor will be computed.
    2. Provide `Q` directly. If both `Q` and `n` are provided, `Q` will be ignored with a warning.

    Parameters
    ----------
    Q : QField, optional
        A precomputed Q-tensor field of shape (..., 5) or (..., 3, 3).
        Ignored if `n` is also provided.

    S : SField, optional
        Scalar order parameter field, shape (...,). Only used if `n` is provided.
        If omitted, a constant value of 1 is assumed.

    n : nField, optional
        Director field of shape (..., 3). If provided, the Q-tensor is constructed from it.

    box_periodic_size : DimensionPeriodic, optional
        Simulation box size or periodicity indicator. Defaults to [∞, ∞, ∞].

    origin : Vect3D, optional
        Origin of the spatial grid. Defaults to (0, 0, 0).

    space_index_ratio : DimensionInfo, optional
        Ratio between physical space and index space. Defaults to 1.

    logger : Logger, optional
        A logger instance used to report warnings or debug information. If None, logging is disabled.

    Raises
    ------
    NameError
        If neither `Q` nor `n` is provided. At least one form of input is required.
    """

    @logging_and_warning_decorator()
    def __init__(
        self,
        Q: QField = None,
        S: SField = None,
        n: nField = None,
        box_periodic_size: DimensionPeriodic = [np.inf, np.inf, np.inf],
        grid_offset: Optional[Vect3D] = None,
        grid_transform: Optional[np.ndarray] = None,
        is_diag: bool = True,
        logger: Logger = None
    ) -> None:
        
        start = time.time()
        logger.debug('Start to initialize Q field')
        if n is not None:
            self._n = n
            logger.debug('Initialize Q field with S and n')
            if S is not None:
                self._S = S
            else:
                logger.warning('No S input. Set to 1 everywhere.')
                self._S = np.zeros(np.shape(n)[:-1]) + 1.
            if Q is not None:
                logger.warning('Both Q and n are provided. Q will be IGNORED.')
            self._Q = as_QField5(getQ(n, S=S))
            logger.debug(f"Q field initialized in {time.time() - start:.2f} seconds.")
        else:
            if Q is not None:
                logger.debug('Initialize Q field with Q directly')
                self._Q = as_QField5(Q)
                logger.debug(f"Q field initialized in {time.time() - start:.2f} seconds.")
                if is_diag:
                    self._S, self._n = diagonalizeQ(self._Q, logger=logger)
            else:
                raise NameError('No data is input')
            
        self._box_periodic_size = box_periodic_size
        self._box_periodic_flag = boundary_periodic_size_to_flag(box_periodic_size)

        self._Q = add_periodic_boundary(self._Q, is_boundary_periodic=self._box_periodic_flag)
        if is_diag:
            self._n = add_periodic_boundary(self._n, is_boundary_periodic=self._box_periodic_flag)
            self._S = add_periodic_boundary(self._S, is_boundary_periodic=self._box_periodic_flag)

        grid_shape = np.shape(self._Q)[:3]
        self._grid = generate_coordinate_grid(grid_shape, grid_shape, grid_transform, grid_offset)

    
    def update_diag(self, logger=None):
        self._S, self._n = diagonalizeQ(self._Q, logger=logger)
        self._n = add_periodic_boundary(self._n, is_boundary_periodic=self._box_periodic_flag)
        self._S = add_periodic_boundary(self._S, is_boundary_periodic=self._box_periodic_flag)
            
        

        
