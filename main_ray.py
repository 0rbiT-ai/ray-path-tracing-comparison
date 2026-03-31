"""
main_ray.py — runs traceImage(scene) from the pseudocode:

    for each pixel (i, j) in image:
        A = pixelToWorld(i, j)
        d = (A - C) / ||A - C||
        I(i,j) = traceRay(C, d, scene)
"""
import numpy as np
from PIL import Image

from scene import spheres, plane, light_pos, light_intensity, camera_pos, width, height, fov
from ray_tracing import trace_ray


def pixel_to_world(i, j, width, height, fov, camera_pos):
    """
    A = pixelToWorld(i, j)
    Converts pixel (i=col, j=row) to a world-space point on the image plane.
    """
    aspect = width / height
    # Map pixel to [-1, 1] NDC, then to world via FOV
    px = (2 * (i + 0.5) / width - 1) * np.tan(fov / 2) * aspect
    py = (1 - 2 * (j + 0.5) / height) * np.tan(fov / 2)
    A = camera_pos + np.array([px, py, -1.0])
    return A


def trace_image(scene):
    """
    traceImage(scene) — outer loop from pseudocode.
    Returns an (H, W, 3) float array of pixel colors.
    """
    w, h = scene["width"], scene["height"]
    image = np.zeros((h, w, 3))
    C = scene["camera_pos"]

    for j in range(h):
        for i in range(w):
            # A = pixelToWorld(i, j)
            A = pixel_to_world(i, j, w, h, scene["fov"], C)

            # d = (A - C) / ||A - C||
            d = A - C
            d = d / np.linalg.norm(d)

            # I(i,j) = traceRay(C, d, scene)
            image[j, i] = trace_ray(C, d, scene, depth=3)

        if j % 40 == 0:
            print(f"  Row {j}/{h} done...")

    return image


if __name__ == "__main__":
    # Build scene dict (adds default material fields to objects)
    scene = {
        "spheres": spheres,
        "plane": plane,
        "light_pos": light_pos,
        "light_intensity": light_intensity,
        "camera_pos": camera_pos,
        "width": width,
        "height": height,
        "fov": fov,
    }

    print("Ray Tracing started...")
    img_array = trace_image(scene)

    # Convert to 8-bit and save
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    out = Image.fromarray(img_uint8)
    out.save("output_ray.png")
    print("Done! Saved to output_ray.png")