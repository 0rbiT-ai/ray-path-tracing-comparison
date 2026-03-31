"""
mis_tracing.py — Multiple Importance Sampling (MIS) path tracer.

Uses one-sample MIS with the balance heuristic:
  - Strategy 1: importance_sample_light (biased toward light)
  - Strategy 2: uniform_sample_hemisphere (unbiased)

At each hit point:
  1. Direct illumination via explicit shadow ray (NEE)
  2. Indirect illumination via one-sample MIS
     → randomly pick one strategy, weight by the mixture PDF
"""

import numpy as np
from core.ray import Ray
from core.objects import Sphere
from core.utils import normalize, dot

# ── Try to import teammate's sampling functions ──────────────────────────
# If core/sampling.py is not yet implemented, fall back to built-in stubs.
# Delete this try/except block once the real sampling.py is ready.
try:
    from core.sampling import (
        uniform_sample_hemisphere,
        importance_sample_light,
        importance_sample_light_pdf,
    )
except (ImportError, AttributeError):
    # ── Temporary stubs ──────────────────────────────────────────────────
    LIGHT_LOBE_EXPONENT = 20

    def _build_basis(w):
        """Build orthonormal basis (u, v, w) where w is the given unit vector."""
        if abs(w[0]) > 0.1:
            a = np.array([0.0, 1.0, 0.0])
        else:
            a = np.array([1.0, 0.0, 0.0])
        v = np.cross(w, a)
        v = v / np.linalg.norm(v)
        u = np.cross(v, w)
        return u, v, w

    def uniform_sample_hemisphere(normal):
        r1 = np.random.rand()
        r2 = np.random.rand()
        phi = 2.0 * np.pi * r1
        cos_theta = r2
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        local = np.array([sin_theta * np.cos(phi),
                          sin_theta * np.sin(phi),
                          cos_theta])
        w = normal / np.linalg.norm(normal)
        u, v, w = _build_basis(w)
        direction = u * local[0] + v * local[1] + w * local[2]
        direction = direction / np.linalg.norm(direction)
        pdf = 1.0 / (2.0 * np.pi)
        return direction, pdf

    def importance_sample_light(normal, light_dir):
        n = LIGHT_LOBE_EXPONENT
        r1 = np.random.rand()
        r2 = np.random.rand()
        cos_theta = r1 ** (1.0 / (n + 1))
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        phi = 2.0 * np.pi * r2
        local = np.array([sin_theta * np.cos(phi),
                          sin_theta * np.sin(phi),
                          cos_theta])
        w = light_dir / np.linalg.norm(light_dir)
        u, v, w = _build_basis(w)
        direction = u * local[0] + v * local[1] + w * local[2]
        # Ensure direction is in the hemisphere of the surface normal
        if np.dot(direction, normal) < 0:
            direction = direction - 2.0 * np.dot(direction, normal) * normal
        direction = direction / np.linalg.norm(direction)
        cos_alpha = max(np.dot(direction, light_dir), 0.0)
        pdf = (n + 1) / (2.0 * np.pi) * (cos_alpha ** n)
        pdf = max(pdf, 1e-8)
        return direction, pdf

    def importance_sample_light_pdf(direction, light_dir):
        n = LIGHT_LOBE_EXPONENT
        cos_alpha = max(np.dot(direction, light_dir), 0.0)
        pdf = (n + 1) / (2.0 * np.pi) * (cos_alpha ** n)
        return max(pdf, 1e-8)
# ── End of temporary stubs ───────────────────────────────────────────────


# ── Constants ────────────────────────────────────────────────────────────
EPSILON = 1e-4
PI = np.pi
BG_COLOR = np.array([0.1, 0.1, 0.15])


def intersect_scene(ray, scene):
    """
    Find the closest intersection of `ray` with any object in the scene.

    Returns:
        (hit_point, normal, obj)  on hit
        (None, None, None)        on miss
    """
    closest_t = np.inf
    hit_obj = None

    for obj in scene["spheres"]:
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

    # Compute surface normal
    if isinstance(hit_obj, Sphere):
        normal = normalize(hit_point - hit_obj.center)
    else:
        # Plane — flip normal to face the incoming ray
        normal = hit_obj.normal.copy()
        if dot(normal, ray.direction) > 0:
            normal = -normal

    return hit_point, normal, hit_obj


def shadow_test(hit_point, normal, light_dir, light_dist, scene):
    """
    Returns 1.0 if the point light is visible from `hit_point`, 0.0 if blocked.
    """
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


def mis_trace(ray, scene, depth):
    """
    One-sample MIS path tracer.

    At each surface hit:
      1. Ambient term
      2. Direct illumination via explicit shadow ray (NEE)
      3. Indirect illumination via one-sample MIS:
         - 50% chance: sample direction biased toward light
         - 50% chance: sample direction uniformly over hemisphere
         - Weight by mixture PDF (balance heuristic)
    """
    if depth <= 0:
        return np.zeros(3)

    hit_point, normal, obj = intersect_scene(ray, scene)

    if obj is None:
        return BG_COLOR.copy()

    color = obj.color
    brdf = color / PI  # Lambertian diffuse BRDF

    light_pos = scene["light_pos"]
    light_intensity = scene["light_intensity"]

    # ── 1. Ambient ───────────────────────────────────────────────────────
    ambient = 0.05 * color

    # ── 2. Direct illumination (NEE — shadow ray to point light) ─────────
    to_light = light_pos - hit_point
    light_dist = np.linalg.norm(to_light)
    light_dir = to_light / light_dist

    visibility = shadow_test(hit_point, normal, light_dir, light_dist, scene)

    direct = np.zeros(3)
    if visibility > 0.0:
        cos_direct = max(dot(normal, light_dir), 0.0)
        direct = brdf * light_intensity * cos_direct / (light_dist * light_dist)

    # ── 3. Indirect illumination (one-sample MIS) ────────────────────────
    #
    # Randomly choose ONE of two strategies:
    #   Strategy A — importance_sample_light: biased toward light direction
    #   Strategy B — uniform_sample_hemisphere: unbiased random
    #
    # Mixture PDF = 0.5 × pdf_A(ω) + 0.5 × pdf_B(ω)
    # This is the balance heuristic — provably near-optimal (Veach 1995).

    if np.random.rand() < 0.5:
        # Strategy A: light-biased sampling
        sample_dir, pdf_sample = importance_sample_light(normal, light_dir)
        pdf_other = 1.0 / (2.0 * PI)  # uniform hemisphere PDF
    else:
        # Strategy B: uniform hemisphere sampling
        sample_dir, pdf_sample = uniform_sample_hemisphere(normal)
        pdf_other = importance_sample_light_pdf(sample_dir, light_dir)

    # Mixture PDF (balance heuristic with equal selection probabilities)
    mixture_pdf = 0.5 * pdf_sample + 0.5 * pdf_other

    # Trace the bounce ray
    bounce_origin = hit_point + normal * EPSILON
    bounce_ray = Ray(bounce_origin, sample_dir)
    L_incoming = mis_trace(bounce_ray, scene, depth - 1)

    cos_theta = max(dot(normal, sample_dir), 0.0)

    indirect = np.zeros(3)
    if mixture_pdf > EPSILON:
        indirect = brdf * L_incoming * cos_theta / mixture_pdf

    # Clamp negative values (can arise from numerical issues)
    indirect = np.maximum(indirect, 0.0)

    # ── 4. Combine ───────────────────────────────────────────────────────
    return ambient + direct + indirect