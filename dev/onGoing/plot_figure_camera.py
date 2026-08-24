from dataclasses import dataclass, field
import numpy as np
import weakref
from nematics3d.datatypes import Vect, as_Number, as_vector
from nematics3d.logging_decorator import logging_and_warning_decorator


@dataclass(slots=True)
class OptsCamera:
    azimuth: float = 0.0
    elevation: float = 0.0
    roll: float = 0.0
    distance: float = 10.0
    focal_point: Vect(3) = (0.0, 0.0, 0.0)

    _on_change: callable = field(default=None, repr=False, compare=False)

    __descriptions__ = {
        "azimuth": "The azimuthal angle (degrees) of the camera around the focal point.",
        "elevation": "The elevation angle (degrees) of the camera relative to the focal plane.",
        "roll": "The rotation (degrees) of the camera about the direction of projection.",
        "distance": "The distance from the camera position to the focal point.",
        "focal_point": "The point the camera is looking at (x, y, z).",
    }

    _validators = {
        "azimuth": lambda self, v: as_Number(
            v,
            name=self.__descriptions__["azimuth"],
            value_range=(0, 360),
            bounded=True,
        ),
        "elevation": lambda self, v: as_Number(
            v,
            name=self.__descriptions__["elevation"],
            value_range=(-90, 90),
            bounded=True,
        ),
        "roll": lambda self, v: as_Number(
            v,
            name=self.__descriptions__["roll"],
            value_range=(-180, 180),
            bounded=True,
        ),
        "distance": lambda self, v: as_Number(
            v,
            name=self.__descriptions__["distance"],
            value_range=(0, np.inf),
        ),
        "focal_point": lambda self, v: as_vector(
            v,
            name=self.__descriptions__["focal_point"],
            d=3,
        ),
    }

    def __setattr__(self, key, value):
        if key in self._validators:
            value = self._validators[key](self, value)

        old_value = getattr(self, key, None)
        object.__setattr__(self, key, value)

        if (
            not key.startswith("_")
            and old_value is not None
            and not np.allclose(old_value, value, atol=1e-7)
        ):
            if self._on_change:
                self._on_change(key, value)


class FigureCamera:
    def __init__(self, plotter):
        self._internal_plotter_ref = weakref.ref(plotter)

        self.opts = OptsCamera()
        self._helper_sync_from_cam()
        self.opts._on_change = self._helper_sync_from_opts

        weak_self = weakref.proxy(self)

        def _on_interaction_end(obj, event):
            try:
                weak_self._helper_sync_from_cam()
            except ReferenceError:
                pass

        plotter.iren.add_observer("EndInteractionEvent", _on_interaction_end)

    @property
    def _internal_plotter(self):
        return self._internal_plotter_ref()

    @property
    def _internal_cam(self):
        p = self._internal_plotter
        return p.camera if p else None

    @staticmethod
    def _helper_convert_pos_to_spherical(position, focal_point):
        pos = np.array(position)
        foc = np.array(focal_point)
        vec = pos - foc

        dist = np.linalg.norm(vec)

        if dist < 1e-9:
            return 0.0, 0.0, 0.0, focal_point

        elevation = np.degrees(np.arcsin(vec[2] / dist))

        az_rad = np.arctan2(vec[1], vec[0])
        azimuth = np.degrees(az_rad) % 360

        return azimuth, elevation, dist

    def _helper_sync_from_opts(self, key, value):
        cam = self._internal_cam
        if cam is not None:
            setattr(cam, key, value)
            self._internal_plotter.render()

    def _helper_sync_from_cam(self):
        if self._internal_cam is None:
            return

        cb = self.opts._on_change
        self.opts._on_change = None

        self.opts.roll = self._internal_cam.roll
        self.opts.focal_point = self._internal_cam.focal_point

        temp = self._helper_convert_pos_to_spherical(
            self._internal_cam.position, self._internal_cam.focal_point
        )

        self.opts.azimuth = temp[0]
        self.opts.elevation = temp[1]
        self.opts.distance = temp[2]

        self.opts._on_change = cb

    @property
    def position(self):
        return self._internal_cam.position

    @position.setter
    def position(self, v):
        self._internal_cam.position = as_vector(
            v, name="The position of the camera", d=3
        )
        self._internal_plotter.render()

    @property
    def view_up(self):
        return self._internal_cam.up

    @view_up.setter
    def view_up(self, v):
        self._internal_cam.up = as_vector(
            v, name="The upward direction of the camera", d=3
        )
        self._internal_plotter.render()

    def act_reset(self):
        self._internal_plotter.reset_camera()
        self._helper_sync_from_cam()

    def act_view_xy(self):
        self._internal_plotter.view_xy()
        self._helper_sync_from_cam()

    def act_view_xz(self):
        self._internal_plotter.view_xz()
        self._helper_sync_from_cam()

    def act_view_yz(self):
        self._internal_plotter.view_yz()
        self._helper_sync_from_cam()

    def act_view_isometric(self):
        self._internal_plotter.view_isometric()
        self._helper_sync_from_cam()

    @logging_and_warning_decorator(start_finish_level=5)
    def act_log_parameters(self, is_return: bool = False, logger=None) -> str:
        lines = []
        lines.append("-------------- FigureCamera Parameters --------------")

        lines.append("[Camera Options]:")
        for attr in self.opts.__slots__:
            if attr.startswith("_"):
                continue

            desc = self.opts.__descriptions__.get(attr, "(no description)")
            value = getattr(self.opts, attr, None)
            lines.append(f"  {attr}: {value!r}  # {desc}")

        lines.append("[Spatial States]:")
        lines.append(
            f"  position: {self.position!r}  # The absolute coordinate of the camera"
        )
        lines.append(
            f"  view_up : {self.view_up!r}  # The upward orientation of the camera"
        )

        lines.append("-----------------------------------------------------")

        msg = "\n".join(lines)

        if is_return:
            return msg
        else:
            logger.info(msg)

    def __str__(self) -> str:
        header = f"<{self.__class__.__name__} at {hex(id(self))}>"
        return header + "\n" + self.act_log_parameters(is_return=True)
