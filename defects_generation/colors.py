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


def sample_dark_background_color(
    image: np.ndarray,
    occupied_mask: np.ndarray,
    percentile: float = 20.0,
) -> Tuple[int, int, int]:
    h, w = image.shape[:2]
    mask_resized = cv2.resize(occupied_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    bg_mask = (mask_resized == 0)

    if not np.any(bg_mask):
        return (30, 100, 20)  # dark green default

    bg_pixels = image[bg_mask]
    # Compute brightness (sum of BGR channels)
    brightness = bg_pixels.sum(axis=1)
    # Find the threshold for darkest percentile
    threshold = np.percentile(brightness, percentile)
    dark_pixels = bg_pixels[brightness <= threshold]

    if len(dark_pixels) == 0:
        dark_pixels = bg_pixels

    mean_color = np.mean(dark_pixels, axis=0).astype(int)
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


def sample_midtone_trace_color(
    image: np.ndarray,
    trace_mask: np.ndarray,
    low_pct: float = 30.0,
    high_pct: float = 70.0,
) -> Tuple[int, int, int]:
    """Sample color from mid-brightness trace pixels — avoids dark edges and white highlights."""
    h, w = image.shape[:2]
    mask_resized = cv2.resize(trace_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    trace_pixels_mask = (mask_resized > 0)

    if not np.any(trace_pixels_mask):
        return (160, 160, 160)

    trace_pixels = image[trace_pixels_mask]
    brightness = trace_pixels.sum(axis=1)
    lo = np.percentile(brightness, low_pct)
    hi = np.percentile(brightness, high_pct)
    mid_pixels = trace_pixels[(brightness >= lo) & (brightness <= hi)]

    if len(mid_pixels) == 0:
        mid_pixels = trace_pixels

    mean_color = np.mean(mid_pixels, axis=0).astype(int)
    return tuple(mean_color.tolist())


def sample_trace_center_color(
    image: np.ndarray,
    trace_mask: np.ndarray,
    skeleton: np.ndarray,
    center_x: int = None,
    center_y: int = None,
    radius: int = 8,
) -> Tuple[int, int, int]:
    """Sample color from the center (skeleton) area of a trace.

    If center_x, center_y provided, samples from a small region around that point.
    Otherwise samples from the entire skeleton.
    """
    h, w = image.shape[:2]
    mask_h, mask_w = trace_mask.shape
    scale_x = w / mask_w
    scale_y = h / mask_h

    if center_x is not None and center_y is not None:
        # Sample from a small circular region around the center point
        cx_img = int(center_x * scale_x)
        cy_img = int(center_y * scale_y)

        # Create a small circular mask around the point
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_sq = (x_coords - cx_img) ** 2 + (y_coords - cy_img) ** 2
        sample_mask = dist_sq <= (radius ** 2)

        # Also require it to be within the trace
        trace_mask_scaled = cv2.resize(trace_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        sample_mask = sample_mask & (trace_mask_scaled > 0)

        if np.any(sample_mask):
            center_pixels = image[sample_mask]
            mean_color = np.mean(center_pixels, axis=0).astype(int)
            return tuple(mean_color.tolist())

    # Fallback: sample from skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    center_mask = cv2.dilate(skeleton, kernel)
    center_mask = center_mask & (trace_mask > 0)

    center_mask_scaled = cv2.resize(center_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if not np.any(center_mask_scaled > 0):
        return sample_trace_color(image, trace_mask)

    center_pixels = image[center_mask_scaled > 0]
    mean_color = np.mean(center_pixels, axis=0).astype(int)
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
