import numpy as np
from core.utils import dot, normalize

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = dot(ray.direction, ray.direction)
        b = 2.0 * dot(oc, ray.direction)
        c = dot(oc, oc) - self.radius**2

        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return t if t > 0 else None


class Plane:
    def __init__(self, point, normal, color):
        self.point = np.array(point)
        self.normal = normalize(np.array(normal))
        self.color = np.array(color)

    def intersect(self, ray):
        denom = dot(self.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None

        t = dot(self.point - ray.origin, self.normal) / denom
        return t if t > 0 else None