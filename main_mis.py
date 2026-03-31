"""
main_mis.py — Renders the scene using MIS path tracing.

Parallelized across CPU cores (one row per worker).
Outputs: mis.png
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image

from core.ray import Ray
from core.utils import normalize
from mis_tracing import mis_trace
from scene import spheres, plane, light_pos, light_intensity, camera_pos, width, height, fov


# ── Easy-to-tweak settings ──────────────────────────────────────────────
SAMPLES   = 100       # samples per pixel (increase for cleaner image)
MAX_DEPTH = 5         # max bounce depth  (increase for more indirect light)
# ─────────────────────────────────────────────────────────────────────────


# Build scene dict once (used by all workers)
SCENE = {
    "spheres":         spheres,
    "plane":           plane,
    "light_pos":       light_pos,
    "light_intensity": light_intensity,
    "camera_pos":      camera_pos,
    "width":           width,
    "height":          height,
    "fov":             fov,
}


def render_row(j):
    """
    Render a single row of pixels. Called in parallel by the worker pool.
    Returns (row_index, row_data) so we can reassemble the image.
    """
    w = SCENE["width"]
    h = SCENE["height"]
    cam = SCENE["camera_pos"]
    f = SCENE["fov"]
    aspect = w / h

    row = np.zeros((w, 3))

    for i in range(w):
        pixel_color = np.zeros(3)

        for s in range(SAMPLES):
            # Anti-aliasing: jitter the pixel position
            u = (i + np.random.rand()) / w
            v = (j + np.random.rand()) / h

            # Map to world-space direction
            x = (2.0 * u - 1.0) * np.tan(f / 2.0) * aspect
            y = (1.0 - 2.0 * v) * np.tan(f / 2.0)

            direction = normalize(np.array([x, y, -1.0]))
            ray = Ray(cam, direction)

            pixel_color += mis_trace(ray, SCENE, MAX_DEPTH)

        # Average over all samples
        pixel_color /= SAMPLES

        # Clamp and gamma correction (sRGB)
        pixel_color = np.clip(pixel_color, 0.0, 1.0)
        pixel_color = np.power(pixel_color, 1.0 / 2.2)

        row[i] = pixel_color

    print(f"  Row {j + 1}/{SCENE['height']} done")
    return j, row


def render():
    """Render the full image using multiprocessing."""
    image = np.zeros((height, width, 3))
    cores = cpu_count()

    print(f"MIS Path Tracing")
    print(f"  Resolution:  {width}×{height}")
    print(f"  Samples/px:  {SAMPLES}")
    print(f"  Max depth:   {MAX_DEPTH}")
    print(f"  CPU cores:   {cores}")
    print()

    with Pool(cores) as pool:
        results = pool.map(render_row, range(height))

    for j, row in results:
        image[j] = row

    save_image(image)


def save_image(image):
    """Convert to 8-bit and save as PNG."""
    img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    img.save("mis.png")
    print(f"\nDone! Saved to mis.png")


if __name__ == "__main__":
    render()