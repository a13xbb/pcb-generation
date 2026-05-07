from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "layout_generator"))

from utils import skeletonize, find_endpoints

from .colors import sample_dark_background_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def _create_break_shape(
    rng: np.random.Generator,
    center_x: int,
    center_y: int,
    trace_angle: float,
    trace_width: int,
    break_width: int,
    img_shape: tuple,
) -> np.ndarray:
    h, w = img_shape
    break_mask = np.zeros((h, w), dtype=np.uint8)

    shape_type = rng.choice(["wedge", "irregular", "straight"])

    perp_angle = trace_angle + np.pi / 2
    cos_perp = np.cos(perp_angle)
    sin_perp = np.sin(perp_angle)
    cos_dir = np.cos(trace_angle)
    sin_dir = np.sin(trace_angle)

    half_width = trace_width // 2 + 3
    half_break = break_width // 2

    if shape_type == "wedge":
        wide_side = rng.choice([-1, 1])
        wide_hw = int(half_break * rng.uniform(1.2, 1.8))
        narrow_hw = int(half_break * rng.uniform(0.3, 0.6))

        if wide_side == -1:
            hw1, hw2 = wide_hw, narrow_hw
        else:
            hw1, hw2 = narrow_hw, wide_hw

        points = [
            (int(center_x - half_width * cos_perp - hw1 * cos_dir),
             int(center_y - half_width * sin_perp - hw1 * sin_dir)),
            (int(center_x - half_width * cos_perp + hw1 * cos_dir),
             int(center_y - half_width * sin_perp + hw1 * sin_dir)),
            (int(center_x + half_width * cos_perp + hw2 * cos_dir),
             int(center_y + half_width * sin_perp + hw2 * sin_dir)),
            (int(center_x + half_width * cos_perp - hw2 * cos_dir),
             int(center_y + half_width * sin_perp - hw2 * sin_dir)),
        ]
        pts = np.array(points, np.int32)
        cv2.fillPoly(break_mask, [pts], 255)

    elif shape_type == "irregular":
        n_points_per_side = rng.integers(3, 6)
        points = []

        for side in [-1, 1]:
            base_x = center_x + side * half_width * cos_perp
            base_y = center_y + side * half_width * sin_perp

            for i in range(n_points_per_side):
                t = -1 + 2 * i / (n_points_per_side - 1)
                jitter = rng.uniform(-0.3, 0.3) * half_break
                px = int(base_x + (t * half_break + jitter) * cos_dir)
                py = int(base_y + (t * half_break + jitter) * sin_dir)
                points.append([px, py])

        points = np.array(points, np.int32)
        hull = cv2.convexHull(points)
        cv2.fillPoly(break_mask, [hull], 255)

    else:  # straight
        hw_variation = rng.uniform(0.8, 1.2)
        actual_hw = int(half_break * hw_variation)

        points = [
            (int(center_x - half_width * cos_perp - actual_hw * cos_dir),
             int(center_y - half_width * sin_perp - actual_hw * sin_dir)),
            (int(center_x - half_width * cos_perp + actual_hw * cos_dir),
             int(center_y - half_width * sin_perp + actual_hw * sin_dir)),
            (int(center_x + half_width * cos_perp + actual_hw * cos_dir),
             int(center_y + half_width * sin_perp + actual_hw * sin_dir)),
            (int(center_x + half_width * cos_perp - actual_hw * cos_dir),
             int(center_y + half_width * sin_perp - actual_hw * sin_dir)),
        ]
        pts = np.array(points, np.int32)
        cv2.fillPoly(break_mask, [pts], 255)

    return break_mask


def generate_open_circuit(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    min_break_width: int = 6,
    max_break_width: int = 16,
) -> DefectResult:
    long_traces = filter_long_traces(canvas.trace_instances)
    if not long_traces:
        return DefectResult(success=False)

    trace = rng.choice(long_traces)
    mask = trace.mask_world

    skel = skeletonize(mask)
    skel_pts = np.argwhere(skel > 0)

    if len(skel_pts) < 10:
        return DefectResult(success=False)

    endpoints = find_endpoints(mask, skel)
    if len(endpoints) < 2:
        return DefectResult(success=False)

    ep_set = set(endpoints)

    # Build pad exclusion zone
    pad_mask = np.zeros_like(mask)
    for pad in canvas.pad_instances:
        pad_mask = np.maximum(pad_mask, pad.mask_world)
    pad_buffer = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_buffer * 2 + 1, pad_buffer * 2 + 1))
    pad_zone = cv2.dilate(pad_mask, kernel)

    valid_pts = []
    for y, x in skel_pts:
        # Skip points near endpoints
        min_dist = min(np.sqrt((x - ex) ** 2 + (y - ey) ** 2) for ex, ey in ep_set)
        if min_dist <= 15:
            continue
        # Skip points near pads
        if pad_zone[y, x] > 0:
            continue
        valid_pts.append((x, y))

    if len(valid_pts) < 3:
        return DefectResult(success=False)

    break_x, break_y = valid_pts[rng.integers(len(valid_pts))]

    neighborhood = 5
    ys, xs = np.where(skel[max(0, break_y - neighborhood):break_y + neighborhood + 1,
                           max(0, break_x - neighborhood):break_x + neighborhood + 1] > 0)
    if len(xs) < 2:
        trace_angle = 0.0
    else:
        pts = np.column_stack([xs, ys])
        if len(pts) > 2:
            _, _, vt = np.linalg.svd(pts - pts.mean(axis=0))
            trace_angle = np.arctan2(vt[0, 1], vt[0, 0])
        else:
            trace_angle = 0.0

    perp_angle = trace_angle + np.pi / 2

    h_m, w_m = mask.shape
    trace_width = 0
    for sign in [1, -1]:
        for dist in range(1, 30):
            nx = int(break_x + sign * dist * np.cos(perp_angle))
            ny = int(break_y + sign * dist * np.sin(perp_angle))
            if not (0 <= nx < w_m and 0 <= ny < h_m):
                break
            if mask[ny, nx] == 0:
                break
            trace_width += 1

    if trace_width < 4:
        return DefectResult(success=False)

    h, w = image.shape[:2]
    scale_x = w / w_m
    scale_y = h / h_m
    avg_scale = (scale_x + scale_y) / 2

    bx_img = int(break_x * scale_x)
    by_img = int(break_y * scale_y)
    trace_width_img = int(trace_width * avg_scale)
    break_width_img = rng.integers(min_break_width, max_break_width + 1)

    break_mask = _create_break_shape(
        rng, bx_img, by_img, trace_angle, trace_width_img, break_width_img, (h, w)
    )

    trace_mask_scaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    trace_mask_dilated = cv2.dilate(trace_mask_scaled, kernel)
    break_mask = ((break_mask > 0) & (trace_mask_dilated > 0)).astype(np.uint8) * 255

    overlap_with_trace = np.sum((break_mask > 0) & (trace_mask_scaled > 0))
    if overlap_with_trace < 10:
        return DefectResult(success=False)

    dark_bg_color = sample_dark_background_color(image, canvas.occupied_mask, percentile=20.0)

    noise = rng.normal(0, 5.0, (h, w, 3))
    textured = np.clip(np.array(dark_bg_color) + noise, 0, 255).astype(np.uint8)

    image[break_mask > 0] = textured[break_mask > 0]

    break_ys, break_xs = np.where(break_mask > 0)
    if len(break_xs) == 0:
        return DefectResult(success=False)

    x1 = int(break_xs.min())
    y1 = int(break_ys.min())
    x2 = int(break_xs.max())
    y2 = int(break_ys.max())

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["open_circuit"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
