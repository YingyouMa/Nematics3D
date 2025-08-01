import numpy as np
import time
from typing import Tuple, Optional, List

from ..logging_decorator import logging_and_warning_decorator, Logger
from ..datatypes import (
    Vect3D,
    QField,
    as_QField5,
    SField,
    nField,
    DimensionPeriodicInput,
    boundary_periodic_size_to_flag,
)
from ..field import (
    diagonalizeQ,
    getQ,
    generate_coordinate_grid,
    apply_linear_transform,
)
from ..disclination import defect_detect, defect_classify_into_lines


class QFieldObject:
    """
    A data container and utility class for representing and manipulating a Q-tensor field in 3D space.

    The object can be initialized in two mutually exclusive ways:
    1. Provide `n` (director field), with optional `S` (scalar order parameter). 
       The Q-tensor will be computed.
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

    box_size_periodic : DimensionPeriodic, optional
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

    DEFAULT_SMOOTH_WINDOW_LENGTH = 61
    DEFAULT_MINIMUM_LINE_LENGTH = 75

    @logging_and_warning_decorator()
    def __init__(
        self,
        Q: QField = None,
        S: SField = None,
        n: nField = None,
        box_size_periodic: DimensionPeriodicInput = np.inf,
        grid_offset: Optional[Vect3D] = None,
        grid_transform: Optional[np.ndarray] = None,
        is_diag: bool = True,
        logger: Logger = None,
    ) -> None:

        start = time.time()
        logger.debug("Start to initialize Q field")
        if n is not None:
            self._n = n
            logger.debug("Initialize Q field with S and n")
            if S is not None:
                self._S = S
            else:
                logger.warning("No S input. Set to 1 everywhere.")
                self._S = np.zeros(np.shape(n)[:-1]) + 1.0
            if Q is not None:
                logger.warning("Both Q and n are provided. Q will be IGNORED.")
            self._Q = as_QField5(getQ(n, S=S))
            logger.debug(f"Q field initialized in {time.time() - start:.2f} seconds.")
        else:
            if Q is not None:
                logger.debug("Initialize Q field with Q directly")
                self._Q = as_QField5(Q)
                logger.debug(
                    f"Q field initialized in {time.time() - start:.2f} seconds."
                )
                if is_diag:
                    self._S, self._n = diagonalizeQ(self._Q, logger=logger)
            else:
                raise NameError("No data is input")

        self._box_size_periodic = box_size_periodic
        self._box_periodic_flag = boundary_periodic_size_to_flag(box_size_periodic)

        self._grid_transform = grid_transform
        self._grid_offset = grid_offset
        self.update_grid(grid_transform=grid_transform, grid_offset=grid_offset)

    @logging_and_warning_decorator()
    def update_diag(self, logger=None):
        self._S, self._n = diagonalizeQ(self._Q, logger=logger)

    @logging_and_warning_decorator()
    def update_defects(self, threshold=0, logger=None):

        self._defect_indices = defect_detect(
            self._n,
            threshold=threshold,
            is_boundary_periodic=self._box_periodic_flag,
            logger=logger,
        )

        self._defect_grid = apply_linear_transform(
            self._defect_indices,
            transform=self._grid_transform,
            offset=self._grid_offset,
        )

    def update_grid(
        self,
        grid_transform: Optional[np.ndarray] = None,
        grid_offset: Optional[np.ndarray] = None,
    ):
        """
        Generate the coordinates grid in the real space from the lattice indices through linear transform.
        See the document of apply_linear_transform()
        """

        if not hasattr(self, '_defect_indices'):
            grid_shape = np.shape(self._Q)[:3]
            self._grid = generate_coordinate_grid(grid_shape, grid_shape)

        self._grid_transform = grid_transform
        self._grid_offset = grid_offset
        self._grid = apply_linear_transform(
            self._grid, transform=self._grid_transform, offset=self._grid_offset
        )

        if hasattr(self, '_defect_indices'):
            self._defect_grid = apply_linear_transform(
                self._defect_indices,
                transform=self._grid_transform,
                offset=self._grid_offset,
            )

    @logging_and_warning_decorator()
    def update_lines_classify(self, logger=None):
        self._lines = defect_classify_into_lines(
            self._defect_indices, 
            box_size_periodic=self._box_size_periodic,
            offset=self._grid_offset,
            transform=self._grid_transform,
            logger=logger)
        return self._lines
    
    def update_lines_smoothen(
        self,
        window_ratio: Optional[int] = None,
        window_length: int = DEFAULT_SMOOTH_WINDOW_LENGTH,
        order: int = 3,
        N_out_ratio: float = 3.0,
        min_line_length: int = DEFAULT_MINIMUM_LINE_LENGTH
    ):
        for line in self._lines:
            if line._defect_num >= min_line_length:
                line.update_smoothen(
                    window_ratio=window_ratio,
                    window_length=window_length,
                    order=order,
                    N_out_ratio=N_out_ratio
                )

    def visualize_lines(
            self,
            is_new: bool = True,
            fig_size: Tuple[int, int] = (1920, 1360),
            bgcolor: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            is_wrap: bool = True,
            is_smooth: bool = True,
            color_input: Optional[Tuple[float, float, float]] = None,
            tube_radius: float = 0.5,
            tube_opacity: float = 1,
            tube_specular: float = 1,
            tube_specular_col: Tuple[float, float, float] = (1.0, 1.0, 1.0),
            tube_specular_pow: float = 11,
            is_outline: bool = True,
            outline_radius: float = 3
    ):
        
        from ..disclination import draw_multiple_disclination_lines


        draw_multiple_disclination_lines(
            self._lines,
            is_new=is_new,
            fig_size=fig_size,
            bgcolor=bgcolor,
            is_wrap=is_wrap,
            is_smooth=is_smooth,
            color_input=color_input,
            tube_radius=tube_radius,
            tube_opacity=tube_opacity,
            tube_specular=tube_specular,
            tube_specular_col=tube_specular_col,
            tube_specular_pow=tube_specular_pow,
        )

        from ..field import draw_box
        draw_box()


    def __call__(self) -> np.ndarray:
        return self._Q
