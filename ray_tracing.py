import numpy as np
from core.utils import normalize
from core.ray import Ray
from core.objects import Sphere


def trace_ray(ray_origin, ray_dir, scene, depth=3):
    if depth == 0:
        return np.zeros(3)

    hit_point, normal, material = intersect_scene(ray_origin, ray_dir, scene)

    if hit_point is None:
        return np.array([0.1, 0.1, 0.15])  # background

    d = ray_dir
    N = normal

    # r = -2(d·N)N + d  — reflection direction
    r = d - 2 * np.dot(d, N) * N
    r = normalize(r)

    # I = shade(Q, N, M, d, scene)
    color = shade(hit_point, N, material, d, scene)

    # + M.k_r * traceRay(Q, r, scene)
    k_r = material.get("k_r", 0.3)
    color += k_r * trace_ray(hit_point + N * 1e-4, r, scene, depth - 1)

    # + M.k_t * traceRay(Q, t, scene)
    k_t = material.get("k_t", 0.0)
    if k_t > 0:
        t_dir = refract(d, N, material.get("n_i", 1.0), material.get("n_t", 1.5))
        if t_dir is not None:
            color += k_t * trace_ray(hit_point - N * 1e-4, t_dir, scene, depth - 1)

    return np.clip(color, 0, 1)


def intersect_scene(ray_origin, ray_dir, scene):
    """
    Wraps origin+direction into a Ray object for each object's intersect(),
    then computes normal and material from the hit object.
    """
    ray = Ray(ray_origin, ray_dir)
    closest_t = np.inf
    hit_obj = None

    for obj in scene["spheres"]:
        t = obj.intersect(ray)
        if t is not None and t < closest_t:
            closest_t = t
            hit_obj = obj

    if scene["plane"] is not None:
        t = scene["plane"].intersect(ray)
        if t is not None and t < closest_t:
            closest_t = t
            hit_obj = scene["plane"]

    if hit_obj is None:
        return None, None, None

    hit_point = ray_origin + closest_t * ray_dir

    # Compute normal depending on object type
    if isinstance(hit_obj, Sphere):
        normal = normalize(hit_point - hit_obj.center)
    else:
        # Plane — flip to face the incoming ray
        normal = hit_obj.normal.copy()
        if np.dot(normal, ray_dir) > 0:
            normal = -normal

    # Build material dict from object color + default optical properties
    material = {
        "color":     hit_obj.color,
        "k_e":       0.0,
        "k_a":       0.05,
        "k_d":       0.8,
        "k_s":       0.5,
        "shininess": 32,
        "k_r":       0.2,
        "k_t":       0.0,
    }

    return hit_point, normal, material


def shade(hit_point, N, material, ray_dir, scene):
    light_pos       = scene["light_pos"]
    light_intensity = scene["light_intensity"]

    color     = material["color"]
    k_e       = material["k_e"]
    k_a       = material["k_a"]
    k_d       = material["k_d"]
    k_s       = material["k_s"]
    shininess = material["shininess"]

    # Emission + ambient
    I = k_e * color + k_a * color

    L_vec   = light_pos - hit_point
    dist_sq = np.dot(L_vec, L_vec)
    dist    = np.sqrt(dist_sq)
    L       = L_vec / dist

    dist_atten   = light_intensity / dist_sq
    shadow_atten = shadow_test(hit_point, L, dist, scene)
    atten        = dist_atten * shadow_atten

    # Diffuse (Lambertian)
    diff = max(np.dot(N, L), 0.0)

    # Specular (Blinn-Phong)
    V    = normalize(-ray_dir)
    H    = normalize(L + V)
    spec = max(np.dot(N, H), 0.0) ** shininess

    I += atten * color * (k_d * diff + k_s * spec)
    return I


def shadow_test(hit_point, L, light_dist, scene):
    shadow_ray = Ray(hit_point + L * 1e-4, L)

    for obj in scene["spheres"]:
        t = obj.intersect(shadow_ray)
        if t is not None and t < light_dist:
            return 0.0

    if scene["plane"] is not None:
        t = scene["plane"].intersect(shadow_ray)
        if t is not None and t < light_dist:
            return 0.0

    return 1.0


def refract(d, N, n_i, n_t):
    cos_i = -np.dot(d, N)
    if cos_i < 0:
        cos_i = -cos_i
        N     = -N
        n_i, n_t = n_t, n_i

    ratio  = n_i / n_t
    sin2_t = ratio**2 * (1 - cos_i**2)
    if sin2_t > 1.0:
        return None  # total internal reflection

    cos_t = np.sqrt(1 - sin2_t)
    return ratio * d + (ratio * cos_i - cos_t) * N