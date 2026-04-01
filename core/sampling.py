# Sampling functions will be implemented here

import numpy as np
from core.utils import normalize, dot

LIGHT_LOBE_EXPONENT = 20


def build_basis(w):
    if abs(w[0]) > 0.1:
        a = np.array([0, 1, 0])
    else:
        a = np.array([1, 0, 0])

    v = normalize(np.cross(w, a))
    u = np.cross(v, w)
    return u, v, w


def uniform_sample_hemisphere(normal):
    r1 = np.random.rand()
    r2 = np.random.rand()

    phi = 2 * np.pi * r1
    cos_theta = r2
    sin_theta = np.sqrt(1 - cos_theta**2)

    local = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

    u, v, w = build_basis(normal)

    direction = u * local[0] + v * local[1] + w * local[2]
    direction = normalize(direction)

    pdf = 1 / (2 * np.pi)

    return direction, pdf


def importance_sample_light(normal, light_dir):
    light_dir = normalize(light_dir)

    r1 = np.random.rand()
    r2 = np.random.rand()

    n = LIGHT_LOBE_EXPONENT

    cos_theta = r1 ** (1 / (n + 1))
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * r2

    local = np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

    u, v, w = build_basis(light_dir)

    direction = u * local[0] + v * local[1] + w * local[2]
    direction = normalize(direction)

    # ensure direction is above surface
    if dot(direction, normal) < 0:
        direction = direction - 2 * dot(direction, normal) * normal
        direction = normalize(direction)

    cos_alpha = max(dot(direction, light_dir), 0)
    pdf = (n + 1) / (2 * np.pi) * (cos_alpha ** n)

    return direction, max(pdf, 1e-8)


def importance_sample_light_pdf(direction, light_dir):
    direction = normalize(direction)
    light_dir = normalize(light_dir)

    n = LIGHT_LOBE_EXPONENT

    cos_alpha = max(dot(direction, light_dir), 0)
    pdf = (n + 1) / (2 * np.pi) * (cos_alpha ** n)

    return max(pdf, 1e-8)