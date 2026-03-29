import numpy as np
from core.ray import Ray
from core.utils import dot, normalize
from scene import spheres, plane, light_pos, light_intensity

EPSILON = 1e-4
PI = np.pi

def uniform_sample_hemisphere(normal):
    # Random numbers
    r1 = np.random.rand()
    r2 = np.random.rand()

    phi = 2 * PI * r1
    z = r2
    sin_theta = np.sqrt(1 - z * z)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)

    # Local direction (around Z axis)
    local_dir = np.array([x, y, z])

    # -------------------------
    # Build orthonormal basis
    # -------------------------
    w = normal / np.linalg.norm(normal)

    if abs(w[0]) > 0.1:
        a = np.array([0, 1, 0])
    else:
        a = np.array([1, 0, 0])

    v = np.cross(w, a)
    v = v / np.linalg.norm(v)

    u = np.cross(v, w)

    # -------------------------
    # Transform to world space
    # -------------------------
    world_dir = (
        u * local_dir[0] +
        v * local_dir[1] +
        w * local_dir[2]
    )

    world_dir = world_dir / np.linalg.norm(world_dir)

    # Uniform hemisphere PDF
    pdf = 1 / (2 * PI)

    return world_dir, pdf

# -------------------------
# Scene Intersection Helper
# -------------------------
def intersect_scene(ray):
    closest_t = float('inf')
    hit_obj = None

    # Check spheres
    for obj in spheres:
        t = obj.intersect(ray)
        if t and t < closest_t:
            closest_t = t
            hit_obj = obj

    # Check plane
    t = plane.intersect(ray)
    if t and t < closest_t:
        closest_t = t
        hit_obj = plane

    if hit_obj is None:
        return None, None, None

    hit_point = ray.origin + closest_t * ray.direction

    if isinstance(hit_obj, type(plane)):
        normal = plane.normal
    else:
        normal = normalize(hit_point - hit_obj.center)

    return hit_point, normal, hit_obj


# -------------------------
# MIS TRACE FUNCTION
# -------------------------
def mis_trace(ray, depth):
    if depth <= 0:
        return np.array([0.0, 0.0, 0.0])

    hit_point, normal, obj = intersect_scene(ray)

    if obj is None:
        return np.array([0.0, 0.0, 0.0])

    color = obj.color

    # -------------------------
    # 🌞 DIRECT LIGHT
    # -------------------------
    to_light = light_pos - hit_point
    light_distance = np.linalg.norm(to_light)
    light_dir = normalize(to_light)

    shadow_ray = Ray(hit_point + normal * EPSILON, light_dir)
    shadow_hit_point, _, _ = intersect_scene(shadow_ray)

    visible = True
    if shadow_hit_point is not None:
        shadow_dist = np.linalg.norm(shadow_hit_point - hit_point)
        if shadow_dist > EPSILON and shadow_dist < light_distance:
            visible = False

    direct = np.array([0.0, 0.0, 0.0])

    if visible:
        lambert = max(dot(normal, light_dir), 0.0)
        distance2 = light_distance ** 2
        direct = (color / PI) * light_intensity * lambert / (distance2 + 1e-6)

    # -------------------------
    # 🔁 INDIRECT LIGHT
    # -------------------------
    sample_dir, pdf_brdf = uniform_sample_hemisphere(normal)

    new_ray = Ray(hit_point + normal * EPSILON, sample_dir)
    indirect = mis_trace(new_ray, depth - 1)

    cos_theta = max(dot(normal, sample_dir), 0.0)

    if pdf_brdf > 0:
        indirect = indirect * (color / PI) * cos_theta / pdf_brdf

    indirect*=1.5

    indirect = np.maximum(indirect, 0.0)

    # -------------------------
    # ⚖️ MIS WEIGHTS
    # -------------------------
    pdf_light = 1.0 / (light_distance**2 + 1e-6)

    denom = pdf_light + pdf_brdf + 1e-8

    w_light = pdf_light / denom
    w_brdf = pdf_brdf / denom

    # -------------------------
    # 🎨 FINAL COLOR
    # -------------------------
    ambient = 0.05*color
    return ambient + w_light * direct + w_brdf * indirect