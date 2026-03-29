import numpy as np
from core.objects import Sphere, Plane

# Objects
spheres = [
    Sphere((0, 0, -5), 1, (1.0, 0.2, 0.2)),
    Sphere((2, 0, -6), 1, (0.2, 0.2, 1.0)),
]

plane = Plane((0, -1, 0), (0, 1, 0), (0.8, 0.8, 0.8))

# Light
light_pos = np.array([5, 5, 0])
light_intensity = 120.0

# Camera
camera_pos = np.array([0, 0, 0])
width, height = 400, 400
fov = np.pi / 3