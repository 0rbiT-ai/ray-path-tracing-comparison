import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

from core.ray import Ray
from scene import camera_pos, width, height, fov
from path_tracing import path_trace



samples = 100
max_depth = 5



def render_row(y):
    row = np.zeros((width, 3))

    for x in range(width):
        pixel_color = np.array([0.0, 0.0, 0.0])

        for s in range(samples):
            px = (2 * (x + np.random.rand()) / width - 1) * np.tan(fov / 2) * width / height
            py = (1 - 2 * (y + np.random.rand()) / height) * np.tan(fov / 2)

            ray = Ray(camera_pos, np.array([px, py, -1]))
            pixel_color += path_trace(ray, max_depth)

        pixel_color /= samples
        pixel_color = np.clip(pixel_color, 0, 1)
        pixel_color=np.power(pixel_color,1.0/2.2)

        row[x] = pixel_color

    print(f"Row {y} done")
    return y, row



if __name__ == "__main__":

    image = np.zeros((height, width, 3))

    print(f"Using {cpu_count()} cores...")

    with Pool(cpu_count()) as p:
        results = p.map(render_row, range(height))

    for y, row in results:
        image[y] = row

    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save("path_traced.png")

    print("Saved as path_traced.png")