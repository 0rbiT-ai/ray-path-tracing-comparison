"""
mis_tracing.py — Optimized Path Tracer (NEE + Cosine Sampling)

NOTE:
We removed classic MIS because:
- Scene uses point light → NEE already optimal
- Cosine-weighted sampling gives better variance reduction

Result:
Lower noise, faster convergence, cleaner shadows
"""

import numpy as np
from core.ray import Ray
from core.objects import Sphere
from core.utils import normalize, dot
from core.sampling import cosine_weighted_hemisphere


# ── Constants ─────────────────────────────────────────────
EPSILON = 1e-4
PI = np.pi
BG_COLOR = np.array([0.1, 0.1, 0.15])


# ── Intersection ───────────────────────────────────────────
def intersect_scene(ray, scene, skip_inside=False):
    closest_t = np.inf
    hit_obj = None

    for obj in scene["spheres"]:
        if skip_inside and isinstance(obj, Sphere):
            dist_to_center = np.linalg.norm(ray.origin - obj.center)
            if dist_to_center < obj.radius - EPSILON:
                continue

        t = obj.intersect(ray)
        if t is not None and t > EPSILON and t < closest_t:
            closest_t = t
            hit_obj = obj

    if scene["plane"] is not None:
        t = scene["plane"].intersect(ray)
        if t is not None and t > EPSILON and t < closest_t:
            closest_t = t
            hit_obj = scene["plane"]

    if hit_obj is None:
        return None, None, None

    hit_point = ray.origin + closest_t * ray.direction

    if isinstance(hit_obj, Sphere):
        normal = normalize(hit_point - hit_obj.center)
    else:
        normal = hit_obj.normal.copy()
        if dot(normal, ray.direction) > 0:
            normal = -normal

    return hit_point, normal, hit_obj


# ── Shadow Ray ─────────────────────────────────────────────
def shadow_test(hit_point, normal, light_dir, light_dist, scene):
    shadow_origin = hit_point + normal * EPSILON
    shadow_ray = Ray(shadow_origin, light_dir)

    for obj in scene["spheres"]:
        t = obj.intersect(shadow_ray)
        if t is not None and t > EPSILON and t < light_dist:
            return 0.0

    if scene["plane"] is not None:
        t = scene["plane"].intersect(shadow_ray)
        if t is not None and t > EPSILON and t < light_dist:
            return 0.0

    return 1.0


# ── Main Trace ─────────────────────────────────────────────
def mis_trace(ray, scene, depth, first_bounce=True):

    if depth <= 0:
        return np.zeros(3)

    hit_point, normal, obj = intersect_scene(
        ray, scene, skip_inside=(not first_bounce)
    )

    if obj is None:
        return BG_COLOR.copy()

    color = obj.color
    brdf = color / PI

    light_pos = scene["light_pos"]
    light_intensity = scene["light_intensity"]

    # ── Ambient ───────────────────────────────────────────
    ambient = 0.05 * color

    # ── Direct Lighting (NEE) ─────────────────────────────
    to_light = light_pos - hit_point
    light_dist = np.linalg.norm(to_light)
    light_dir = to_light / light_dist

    visibility = shadow_test(hit_point, normal, light_dir, light_dist, scene)

    direct = np.zeros(3)
    if visibility > 0.0:
        cos_direct = max(dot(normal, light_dir), 0.0)
        direct = brdf * light_intensity * cos_direct / (light_dist * light_dist)

    # ── Russian Roulette ─────────────────────────────────
    if depth < 3:
        rr_prob = 1.0
    else:
        rr_prob = max(color)

        if np.random.rand() > rr_prob:
            return ambient + direct

    # ── Indirect Lighting (Cosine-weighted) ──────────────
    indirect = np.zeros(3)

    dir_b, pdf_b = cosine_weighted_hemisphere(normal)
    cos_b = max(dot(normal, dir_b), 0.0)

    if pdf_b > EPSILON and cos_b > 0.0:
        bounce_ray = Ray(hit_point + normal * EPSILON, dir_b)
        L_b = mis_trace(bounce_ray, scene, depth - 1, first_bounce=False)

        # KEY SIMPLIFICATION: (brdf * cos) / pdf = color
        indirect = color * L_b

        # Apply Russian roulette weight
        indirect /= rr_prob

    # ── Final Color ──────────────────────────────────────
    return ambient + direct + indirect