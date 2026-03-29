import numpy as np
from PIL import Image
from core.ray import Ray
from scene import spheres, plane, camera_pos, width, height, fov

def find_hit(ray):
    closest_t = float('inf')
    hit_color = None

    for obj in spheres:
        t = obj.intersect(ray)
        if t and t < closest_t:
            closest_t = t
            hit_color = obj.color

    t = plane.intersect(ray)
    if t and t < closest_t:
        closest_t = t
        hit_color = plane.color

    return hit_color

image = np.zeros((height, width, 3))

for y in range(height):
    for x in range(width):
        px = (2*(x+0.5)/width - 1)*np.tan(fov/2)*width/height
        py = (1 - 2*(y+0.5)/height)*np.tan(fov/2)

        ray = Ray(camera_pos, np.array([px, py, -1]))
        color = find_hit(ray)

        if color is None:
            color = (0, 0, 0)

        image[y, x] = color

img = Image.fromarray((image * 255).astype(np.uint8))
img.save("basic.png")

print("Basic render saved as basic.png")