"""
mis_tracing.py — Multiple Importance Sampling (MIS) path tracer.

Uses the multi-sample balance heuristic (Veach 1995):
  - Strategy 1: importance_sample_light  (biased toward light source)
  - Strategy 2: uniform_sample_hemisphere (unbiased, uniform coverage)

First bounce:   two-sample MIS — one sample from each strategy, both
                recurse. Balance heuristic weights combine them so
                neither strategy can hurt the other.
Deeper bounces: single uniform sample (same as vanilla path tracing).

Key optimisation: rays that start inside a sphere (contact artifacts)
skip that sphere during intersection to prevent light tunnelling.
"""

import numpy as np
from core.ray import Ray
from core.objects import Sphere
from core.utils import normalize, dot
from core.sampling import (
    uniform_sample_hemisphere,
    importance_sample_light,
    importance_sample_light_pdf,
)


# ── Constants ────────────────────────────────────────────────────────────
EPSILON = 1e-4
PI = np.pi
BG_COLOR = np.array([0.1, 0.1, 0.15])


def intersect_scene(ray, scene, skip_inside=False):
    """
    Find the closest intersection of `ray` with any object in the scene.

    If skip_inside=True, any sphere whose interior contains the ray origin
    is excluded from intersection testing. This prevents bounce rays that
    originate at a sphere–plane contact point from tunnelling through the
    sphere and picking up light from the far side.

    Returns:
        (hit_point, normal, obj)  on hit
        (None, None, None)        on miss
    """
    closest_t = np.inf
    hit_obj = None

    for obj in scene["spheres"]:
        # ── Through-ball fix ─────────────────────────────────────────
        # If the ray starts inside this sphere, skip it entirely.
        # This happens at sphere–ground contact points where the
        # bounce origin (hit + normal*ε) falls just inside the sphere.
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


def mis_trace(ray, scene, depth, first_bounce=True):
    """
    Two-sample MIS path tracer (first bounce) + path tracer (deeper bounces).

    At each surface hit:
      1. Ambient term (small constant to prevent pure-black areas)
      2. Direct illumination via explicit shadow ray (NEE)
      3. Indirect illumination:

         FIRST BOUNCE — Two-sample balance heuristic:
           Sample A: importance_sample_light direction → full recursion
           Sample B: uniform_sample_hemisphere direction → full recursion

           Each sample's contribution:
             contrib = f(ω) / [ pdf_light(ω) + pdf_uniform(ω) ]

           Key property: for uniform samples pointing AWAY from light,
           pdf_light ≈ 0 → denominator ≈ pdf_uniform → contribution
           matches path tracing exactly (no amplification).

           The light-biased sample ADDS extra information about bright
           directions on top. Net result: less noise.

         DEEPER BOUNCES — Single uniform sample:
           Identical to vanilla path tracing. No MIS overhead.
    """
    if depth <= 0:
        return np.zeros(3)

    # ── Primary ray uses normal intersection; bounce rays skip-inside ────
    hit_point, normal, obj = intersect_scene(ray, scene,
                                             skip_inside=(not first_bounce))

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

    # ── 3. Indirect illumination ─────────────────────────────────────────

    indirect = np.zeros(3)

    if first_bounce:
        # ── TWO-SAMPLE BALANCE HEURISTIC (first bounce only) ─────────────
        #
        # Both samples estimate the SAME integral via full recursion.
        #
        # Balance heuristic weight simplifies both to:
        #   contrib(ω) = brdf × L(ω) × cos(θ) / [ pdf_light(ω) + pdf_uniform(ω) ]
        #
        # For uniform samples in dark directions: pdf_light ≈ 0
        #   → denominator ≈ pdf_uniform → same as path tracing (no amplification!)
        #
        # For light samples in bright directions: pdf_light is large
        #   → denominator is large → small but well-estimated contribution
        #
        # Net: uniform sample gives path-tracer floor, light sample adds on top.
        # Two samples at first bounce = visible noise reduction.
        #
        # Cost: 2 rays here, 1 ray at deeper levels ≈ 2× total.

        # Sample A: Light-biased direction
        dir_a, pdf_a = importance_sample_light(normal, light_dir)
        cos_a = max(dot(normal, dir_a), 0.0)

        contrib_a = np.zeros(3)
        if pdf_a > EPSILON and cos_a > 0.0:
            bounce_ray_a = Ray(hit_point + normal * EPSILON, dir_a)
            L_a = mis_trace(bounce_ray_a, scene, depth - 1, first_bounce=False)

            # Balance heuristic denominator
            pdf_a_uniform = 1.0 / (2.0 * PI)
            denom_a = pdf_a + pdf_a_uniform

            if denom_a > EPSILON:
                contrib_a = brdf * L_a * cos_a / denom_a
                contrib_a = np.maximum(contrib_a, 0.0)

        # Sample B: Uniform hemisphere direction
        dir_b, pdf_b = uniform_sample_hemisphere(normal)
        cos_b = max(dot(normal, dir_b), 0.0)

        contrib_b = np.zeros(3)
        if pdf_b > EPSILON and cos_b > 0.0:
            bounce_ray_b = Ray(hit_point + normal * EPSILON, dir_b)
            L_b = mis_trace(bounce_ray_b, scene, depth - 1, first_bounce=False)

            # Balance heuristic denominator
            pdf_b_light = importance_sample_light_pdf(dir_b, light_dir)
            denom_b = pdf_b + pdf_b_light

            if denom_b > EPSILON:
                contrib_b = brdf * L_b * cos_b / denom_b
                contrib_b = np.maximum(contrib_b, 0.0)

        # Sum of both MIS contributions
        indirect = contrib_a + contrib_b

    else:
        # ── SINGLE UNIFORM SAMPLE (deeper bounces — plain path tracing) ──
        dir_b, pdf_b = uniform_sample_hemisphere(normal)
        cos_b = max(dot(normal, dir_b), 0.0)

        if pdf_b > EPSILON and cos_b > 0.0:
            bounce_ray_b = Ray(hit_point + normal * EPSILON, dir_b)
            L_b = mis_trace(bounce_ray_b, scene, depth - 1, first_bounce=False)

            contrib_b = brdf * L_b * cos_b / pdf_b
            indirect = np.maximum(contrib_b, 0.0)

    # ── 4. Combine ───────────────────────────────────────────────────────
    return ambient + direct + indirect