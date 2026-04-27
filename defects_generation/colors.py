from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def sample_background_color(
    image: np.ndarray,
    occupied_mask: np.ndarray,
) -> Tuple[int, int, int]:
    h, w = image.shape[:2]
    mask_resized = cv2.resize(occupied_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    bg_mask = (mask_resized == 0)

    if not np.any(bg_mask):
        return (55, 140, 55)  # default green solder mask BGR

    bg_pixels = image[bg_mask]
    mean_color = np.mean(bg_pixels, axis=0).astype(int)
    return tuple(mean_color.tolist())


def sample_trace_color(
    image: np.ndarray,
    trace_mask: np.ndarray,
) -> Tuple[int, int, int]:
    h, w = image.shape[:2]
    mask_resized = cv2.resize(trace_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    trace_pixels_mask = (mask_resized > 0)

    if not np.any(trace_pixels_mask):
        return (160, 160, 160)  # default copper BGR

    trace_pixels = image[trace_pixels_mask]
    mean_color = np.mean(trace_pixels, axis=0).astype(int)
    return tuple(mean_color.tolist())


def darken_color(color: Tuple[int, int, int], factor: float = 0.6) -> Tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color)


def add_noise_to_color(
    color: Tuple[int, int, int],
    sigma: float = 8.0,
    rng: np.random.Generator = None,
) -> Tuple[int, int, int]:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, sigma, 3)
    noisy = np.clip(np.array(color) + noise, 0, 255).astype(int)
    return tuple(noisy.tolist())


def generate_copper_texture(
    shape: Tuple[int, int],
    base_color: Tuple[int, int, int],
    noise_sigma: float = 8.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    h, w = shape
    texture = np.zeros((h, w, 3), dtype=np.uint8)
    texture[:] = base_color

    noise = rng.normal(0, noise_sigma, (h, w, 3))
    texture = np.clip(texture.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return texture
