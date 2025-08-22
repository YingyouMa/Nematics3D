import numpy as np


class SceneWrapper:
    def __init__(self, scene):
        self._scene = scene
        self._cam = scene.camera

    # --- 工具函数 ---
    def _get_angles(self):
        pos = np.array(self._cam.position)
        focal = np.array(self._cam.focal_point)
        vec = pos - focal
        dist = np.linalg.norm(vec)

        az = np.degrees(np.arctan2(vec[1], vec[0]))
        el = np.degrees(np.arctan2(np.sqrt(vec[0] ** 2 + vec[1] ** 2), vec[2]))

        view_dir = -vec / dist
        up = np.array(self._cam.view_up)
        right = np.cross(view_dir, [0, 0, 1])
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(view_dir, [0, 1, 0])
        right /= np.linalg.norm(right)
        up_ref = np.cross(right, view_dir)
        roll = np.degrees(np.arctan2(np.dot(up, right), np.dot(up, up_ref)))

        return az, el, roll, dist, focal

    def _set_angles(self, az, el, roll, dist, focal):

        if dist is None:
            dist = self.distance

        if focal is None:
            focal = self.focal_point

        az = np.radians(az)
        el = np.radians(el)
        r = np.radians(roll)
        focal = np.asarray(focal, dtype=float)

        x = dist * np.sin(el) * np.cos(az)
        y = dist * np.sin(el) * np.sin(az)
        z = dist * np.cos(el)
        pos = focal + np.array([x, y, z])

        self._cam.position = pos.tolist()
        self._cam.focal_point = focal.tolist()

        view_dir = focal - pos
        view_dir = view_dir / np.linalg.norm(view_dir)
        up_candidate = np.array([0, 0, 1])
        up_proj = up_candidate - np.dot(up_candidate, view_dir) * view_dir
        if np.linalg.norm(up_proj) != 0:
            up = up_proj / np.linalg.norm(up_proj)
        else:
            up = np.array([0, 0, 1])

        if abs(r) > 1e-8:
            k = view_dir
            cos_r, sin_r = np.cos(r), np.sin(r)
            up = up * cos_r + np.cross(k, up) * sin_r + k * np.dot(k, up) * (1 - cos_r)

        self._cam.view_up = up.tolist()
        self._cam.compute_view_plane_normal()

    @property
    def azimuth(self):
        return self._get_angles()[0]

    @azimuth.setter
    def azimuth(self, value):
        _, el, roll, dist, focal = self._get_angles()
        self._set_angles(value, el, roll, dist, focal)

    @property
    def elevation(self):
        return self._get_angles()[1]

    @elevation.setter
    def elevation(self, value):
        az, _, roll, dist, focal = self._get_angles()
        self._set_angles(az, value, roll, dist, focal)

    @property
    def roll(self):
        return self._get_angles()[2]

    @roll.setter
    def roll(self, value):
        az, el, _, dist, focal = self._get_angles()
        self._set_angles(az, el, value, dist, focal)

    @property
    def distance(self):
        return self._get_angles()[3]

    @distance.setter
    def distance(self, value):
        az, el, roll, _, focal = self._get_angles()
        self._set_angles(az, el, roll, value, focal)

    @property
    def focal_point(self):
        return tuple(self._cam.focal_point)

    @focal_point.setter
    def focal_point(self, value):
        az, el, roll, dist, _ = self._get_angles()
        self._set_angles(az, el, roll, dist, value)

    @property
    def bgcolor(self):
        return self._scene.background

    @bgcolor.setter
    def bgcolor(self, value):
        self._scene.background = tuple(value)

    @property
    def fgcolor(self):
        return self._scene.foreground

    @fgcolor.setter
    def fgcolor(self, value):
        self._scene.foreground = tuple(value)
