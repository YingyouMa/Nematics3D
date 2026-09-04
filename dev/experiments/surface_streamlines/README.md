# Surface streamlines experiment

This experiment develops surface-director projection, line-field
interpolation, seed selection, and streamline integration in separate stages.

The first stage is `director_projection.py`. It assumes that one director is
already sampled at every input surface vertex. It does not interpolate a
volume field or integrate streamlines.

The second stage is `director_interpolation.py`. It locates arbitrary query
positions on the closest mesh triangles, sign-aligns the three vertex
directors as a nematic line field, and barycentrically interpolates them. A
reference direction may be supplied so a future streamline integrator can
choose a continuous sign from one step to the next.

The third stage is `surface_streamline.py`. It traces one nematic streamline
in both directions with projected midpoint steps, maps every accepted point
back to the triangle surface, and uses the previous step direction as the
interpolator reference so the line does not reverse under an equivalent
director-sign change.
