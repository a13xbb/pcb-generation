from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_midtone_trace_color, add_noise_to_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def _rounded_rect_mask(
    cx: int,
    cy: int,
    rw: int,
    rh: int,
    corner_r: int,
    angle_deg: float,
    img_shape: tuple,
) -> np.ndarray:
    corner_r = max(2, min(corner_r, rw // 2 - 1, rh // 2 - 1))
    hw = rw / 2
    hh = rh / 2

    # Corner centers in local space and their arc start angles
    corners = [
        ( hw - corner_r,  hh - corner_r,   0),
        (-hw + corner_r,  hh - corner_r,  90),
        (-hw + corner_r, -hh + corner_r, 180),
        ( hw - corner_r, -hh + corner_r, 270),
    ]

    n_per_corner = 10
    points = []
    for ccx, ccy, start_a in corners:
        for i in range(n_per_corner):
            a = np.radians(start_a + i * 90 / (n_per_corner - 1))
            points.append([ccx + corner_r * np.cos(a), ccy + corner_r * np.sin(a)])

    pts = np.array(points, dtype=np.float32)
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = (pts @ rot.T + [cx, cy]).astype(np.int32)

    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def generate_spurious_copper(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    min_w: int = 10,
    max_w: int = 80,
    clearance: int = 12,
) -> DefectResult:
    traces = canvas.trace_instances
    if not traces:
        return DefectResult(success=False)

    # Sample trace color - use median for stable, typical trace color
    combined_trace_mask = np.zeros_like(canvas.occupied_mask)
    for t in traces:
        combined_trace_mask = np.maximum(combined_trace_mask, t.mask_world)

    h, w = image.shape[:2]
    mask_resized = cv2.resize(combined_trace_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    trace_pixels = image[mask_resized > 0]

    if len(trace_pixels) > 0:
        median_color = np.median(trace_pixels, axis=0).astype(int)
        # Boost green, reduce blue/red for richer green
        b, g, r = median_color
        b = int(b * 0.85)
        g = int(min(255, g * 1.05))
        r = int(r * 0.85)
        trace_color = (b, g, r)
    else:
        trace_color = (45, 140, 30)  # fallback BGR green

    trace_color = add_noise_to_color(trace_color, sigma=1.5, rng=rng)

    h, w = image.shape[:2]
    mask_h, mask_w = canvas.occupied_mask.shape

    # Shape dimensions: long side 3–7× the short side (line-like)
    long_side = rng.integers(min_w, max_w + 1)
    ratio = rng.uniform(3.0, 7.0)
    short_side = max(4, min(12, int(long_side / ratio)))
    rw, rh = long_side, short_side
    half_diag = int(np.sqrt(rw ** 2 + rh ** 2) / 2) + 2

    # Build exclusion zone: dilate occupied mask by shape half-diagonal + clearance
    occupied_scaled = cv2.resize(canvas.occupied_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    pad = half_diag + clearance
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    exclusion = cv2.dilate(occupied_scaled, kernel)

    # Mask out near-white image border areas (brightness > 200 across all channels)
    # These are the white padding regions at the edges of generated PCB images
    gray = image.min(axis=2)  # min of BGR — white pixels have high min
    white_border = (gray > 200).astype(np.uint8)
    exclusion = np.maximum(exclusion, white_border)

    # Valid center positions: not in exclusion zone, at least 20px from image edge
    border = half_diag + 20
    if border >= h // 2 or border >= w // 2:
        border = half_diag + 2
    valid_ys, valid_xs = np.where(exclusion[border:h - border, border:w - border] == 0)
    if len(valid_xs) == 0:
        return DefectResult(success=False)

    valid_xs = valid_xs + border
    valid_ys = valid_ys + border

    idx = rng.integers(len(valid_xs))
    cx, cy = int(valid_xs[idx]), int(valid_ys[idx])

    corner_r = int(min(rw, rh) * rng.uniform(0.2, 0.5))
    angle_deg = rng.uniform(0, 360)

    shape_mask = _rounded_rect_mask(cx, cy, rw, rh, corner_r, angle_deg, (h, w))

    # Verify shape doesn't touch any occupied area
    if np.any((shape_mask > 0) & (occupied_scaled > 0)):
        return DefectResult(success=False)

    spur_pixels = np.sum(shape_mask > 0)
    if spur_pixels < 20:
        return DefectResult(success=False)

    # Textured fill
    noise = rng.normal(0, 3.0, (h, w, 3))
    textured = np.clip(np.array(trace_color) + noise, 0, 255).astype(np.uint8)
    image[shape_mask > 0] = textured[shape_mask > 0]

    ys, xs = np.where(shape_mask > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["spurious_copper"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
