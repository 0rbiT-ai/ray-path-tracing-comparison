import numpy as np
from multiprocessing import Pool, cpu_count
from core.ray import Ray
from core.utils import normalize
from mis_tracing import mis_trace
from scene import camera_pos, width, height, fov
from PIL import Image

SAMPLES = 100
MAX_DEPTH = 5


# -------------------------
# Render ONE ROW (parallel)
# -------------------------
def render_row(j):
    aspect_ratio = width / height
    row = np.zeros((width, 3))

    for i in range(width):
        color = np.array([0.0, 0.0, 0.0])

        for s in range(SAMPLES):
            # Anti-aliasing (jitter)
            u = (i + np.random.rand()) / width
            v = (j + np.random.rand()) / height

            x = (2 * u - 1) * np.tan(fov / 2) * aspect_ratio
            y = (1 - 2 * v) * np.tan(fov / 2)

            direction = normalize(np.array([x, y, -1]))
            ray = Ray(camera_pos, direction)

            color += mis_trace(ray, MAX_DEPTH)

        # Average samples
        color /= SAMPLES

        # Clamp + Gamma correction
        color = np.clip(color, 0, 1)
        color = np.power(color, 1/2.2)

        row[i] = color

    print(f"Row {j+1}/{height} done")
    return j, row


# -------------------------
# Main Render Function
# -------------------------
def render():
    image = np.zeros((height, width, 3))

    print(f"Using {cpu_count()} CPU cores...")

    with Pool(cpu_count()) as pool:
        results = pool.map(render_row, range(height))

    # Reconstruct image
    for j, row in results:
        image[j] = row

    save_image(image)


# -------------------------
# Save Image as PNG
# -------------------------
def save_image(image):
    # Convert to 8-bit
    image_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    img = Image.fromarray(image_8bit)
    img.save("mis.png")

    print("Image saved as mis.png")


# -------------------------
# Entry Point (IMPORTANT)
# -------------------------
if __name__ == "__main__":
    render()