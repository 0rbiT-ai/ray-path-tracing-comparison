import numpy as np
from core.ray import Ray
from core.utils import normalize, dot
from scene import spheres, plane, light_pos, light_intensity
from core.sampling import uniform_sample_hemisphere   


def find_hit(ray):
    closest_t = float('inf')
    hit_obj = None

    for obj in spheres:
        t = obj.intersect(ray)
        if t and t < closest_t:
            closest_t = t
            hit_obj = obj

    t = plane.intersect(ray)
    if t and t < closest_t:
        closest_t = t
        hit_obj = plane

    if hit_obj is None:
        return None, None, None

    hit_point = ray.origin + closest_t * ray.direction

    if hasattr(hit_obj, 'center'):
        normal = normalize(hit_point - hit_obj.center)
        color = hit_obj.color
    else:
        normal = hit_obj.normal
        color = hit_obj.color

    return hit_point, normal, color


def path_trace(ray, depth):
    if depth == 0:
        return np.array([0.0, 0.0, 0.0])

    hit_point, normal, color = find_hit(ray)

    if hit_point is None:
        return np.array([0.1, 0.1, 0.15])  # background

    # ✅ USE NEW SAMPLING FUNCTION
    direction, pdf = uniform_sample_hemisphere(normal)

    new_origin = hit_point + normal * 1e-4
    new_ray = Ray(new_origin, direction)

    indirect = path_trace(new_ray, depth - 1)

    # ✅ PDF + cosine correction
    cos_theta = max(dot(normal, direction), 0)

    if pdf > 0:
        indirect = indirect * cos_theta / pdf
    else:
        indirect = np.array([0.0, 0.0, 0.0])

    # -------- DIRECT LIGHT --------
    light_dir = normalize(light_pos - hit_point)
    distance = np.linalg.norm(light_pos - hit_point)

    shadow_origin = hit_point + normal * 1e-4
    shadow_ray = Ray(shadow_origin, light_dir)

    in_shadow = False

    for obj in spheres:
        t = obj.intersect(shadow_ray)
        if t and t < distance:
            in_shadow = True
            break

    if not in_shadow:
        t = plane.intersect(shadow_ray)
        if t and t < distance:
            in_shadow = True

    if in_shadow:
        direct = 0
    else:
        direct = max(dot(normal, light_dir), 0) * light_intensity / (distance * distance)

    
    return (color / np.pi) * (indirect + direct)