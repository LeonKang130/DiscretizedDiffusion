from typing import Tuple

from pyrr import Vector3, Matrix44


class Camera(object):
    def __init__(self, origin: Vector3, look_at: Vector3, resolution: Tuple[int, int], fov: float) -> None:
        self.origin: Vector3 = origin
        self.look_at: Vector3 = look_at
        self.resolution = resolution
        self.fov: float = fov
        self.forward: Vector3 = Vector3(look_at - origin).normalised
        self.right = self.forward.cross(Vector3([0.0, 1.0, 0.0])).normalised
        self.up = self.right.cross(self.forward)
        self.near = 0.01
        self.far = 10.0

    @property
    def mvp(self):
        return (Matrix44.perspective_projection(self.fov, self.resolution[0] / self.resolution[1], self.near,
                                                self.far) * Matrix44.look_at(self.origin, self.look_at,
                                                                             self.up)).astype('f4')
