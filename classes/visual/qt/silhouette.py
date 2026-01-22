import pyvista as pv
import vtk

class SingleSilhouette:
    def __init__(self, 
                 color="black", 
                 line_width=6, 
                 opacity=1.0,
                 ):
        # self.pl = pl

        self._sil = vtk.vtkPolyDataSilhouette()

        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(self._sil.GetOutputPort())
        self._mapper.ScalarVisibilityOff()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self._mapper)
        self.actor.GetProperty().SetColor(pv.Color(color).float_rgb)
        self.actor.GetProperty().SetLineWidth(line_width)
        self.actor.GetProperty().SetOpacity(opacity)
        self.actor.GetProperty().LightingOff()
        self.actor.SetPickable(False)
        self.actor.SetVisibility(False)

        # pl.renderer.AddActor(self.actor)

    def show_for(self, polydata: pv.PolyData):
        surf = polydata.extract_surface().triangulate().clean()

        self._sil.SetCamera(self.pl.renderer.GetActiveCamera())
        self._sil.SetInputData(surf)
        self._sil.Update()

        self.actor.SetVisibility(True)
        # self.pl.render()

    def hide(self):
        self.actor.SetVisibility(False)
        # self.pl.render()
        
    @property
    def color(self):
        """RGB color of the silhouette (tuple of floats)."""
        return self.actor.GetProperty().GetColor()

    @color.setter
    def color(self, value):
        self.actor.GetProperty().SetColor(pv.Color(value).float_rgb)
        # self.pl.render()

    @property
    def line_width(self):
        """Line width of the silhouette."""
        return self.actor.GetProperty().GetLineWidth()

    @line_width.setter
    def line_width(self, value: float):
        self.actor.GetProperty().SetLineWidth(float(value))
        # self.pl.render()
        
    @property
    def opacity(self):
        """Opacity of the silhouette."""
        return self.actor.GetProperty().GetOpacity()

    @opacity.setter
    def opacity(self, value: float):
        self.actor.GetProperty().SetOpacity(float(value))
        # self.pl.render()