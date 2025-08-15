# Changelog

## [0.1.3]
### Added
 - Added ```n_color_immerse()``` to color directors with immersion from RP^2 to R^3 (boys's surface).
### Changed
 - Replaced ```QField``` type alias in ```datatypes.py``` with a more explicit ```Union[QField5, QField9]``` for clearer type annotations.

## [0.1.2]
### Added
 - Added ```PlotPlaneGrid``` for creating a grid plane to visualize quantities.
 - Added ```PlotnPlane``` for visualizing director fields on a plane.

## [0.1.1]
### Added
 - Added ```as_Vect3D()``` and ```as_ColorRGB()``` in ```datatypes.py``` to validate input data.
 - Added smoothing functionality for both ends of disclination lines of type ```cross``` types.

 ### Fixed
- Fixed bugs in `as_QField5()` and `as_QField9()`.

## [0.1.0] 2025-08-13
 - Initial release.
