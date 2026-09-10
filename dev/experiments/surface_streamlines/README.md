# Surface streamlines experiment

This experiment develops surface-director projection, line-field
interpolation, seed selection, and streamline integration in separate stages.

The first stage is now the public
`nematics3d.analysis.project_surface_directors` function. It assumes that one
director is already sampled at every input surface vertex. It does not
interpolate a volume field or integrate streamlines.

The second stage is now the public
`nematics3d.analysis.interpolate_surface_directors` function. It locates
arbitrary query positions on the closest mesh triangles, sign-aligns the three
vertex directors as a nematic line field, and barycentrically interpolates
them. A reference direction may be supplied so a streamline integrator can
choose a continuous sign from one step to the next.

The third stage has also graduated into
`nematics3d.analysis.integrate_surface_streamline`. It traces a nematic line in
both directions with midpoint steps. Interpolated directors are explicitly
projected onto the local smooth tangent plane before every integration step,
and accepted positions are constrained back to the triangle surface. Closed
loops are returned once rather than integrating the same loop independently
in both directions. Closure distance and minimum closure length remain
explicit user-adjustable geometric stopping policies.
