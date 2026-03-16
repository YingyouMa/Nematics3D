import numpy as np

from Nematics3D.field import apply_linear_transform


class Interpolator:

    def __init__(
        self,
        interpolator,
        owner_ref,
    ):

        self._interpolator = interpolator
        self._owner_ref = owner_ref
        
    
    @property
    def owner(self):
        ref = self._owner_ref
        return ref() if ref is not None else None


    def interpolate(self, points: np.ndarray, is_index=False):

        pts = np.asarray(points, dtype=float).copy()        

        if not is_index:
            grid_transform = self.owner._raw_grid_transform
            grid_offset = self.owner._raw_grid_offset
            points = apply_linear_transform(
                points,
                transform=np.linalg.inv(grid_transform),
                offset=-grid_offset,
            )
            
        shape = self.owner._raw_S.shape 
        periodic = self.owner._raw_box_periodic_flag
        
        for d in range(3):
            if periodic[d]:
                pts[:, d] = np.mod(pts[:, d], shape[d])
            else:
                pts[:, d] = np.clip(pts[:, d], 0, shape[d]-1)
                
        return self._interpolator(pts)

