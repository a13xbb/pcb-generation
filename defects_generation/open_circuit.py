from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "layout_generator"))

from utils import skeletonize, find_endpoints

from .colors import sample_background_color, add_noise_to_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def generate_open_circuit(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    gap_width: int = 4,
    gap_extra_ratio: float = 1.3,
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

    valid_pts = []
    for y, x in skel_pts:
        min_dist = min(np.sqrt((x - ex) ** 2 + (y - ey) ** 2) for ex, ey in ep_set)
        if min_dist > 15:
            valid_pts.append((x, y))

    if len(valid_pts) < 3:
        return DefectResult(success=False)

    break_x, break_y = valid_pts[rng.integers(len(valid_pts))]

    neighborhood = 5
    ys, xs = np.where(skel[max(0, break_y - neighborhood):break_y + neighborhood + 1,
                           max(0, break_x - neighborhood):break_x + neighborhood + 1] > 0)
    if len(xs) < 2:
        direction = 0.0
    else:
        pts = np.column_stack([xs, ys])
        if len(pts) > 2:
            _, _, vt = np.linalg.svd(pts - pts.mean(axis=0))
            direction = np.arctan2(vt[0, 1], vt[0, 0])
        else:
            direction = 0.0

    perp_angle = direction + np.pi / 2

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

    gap_length = int(trace_width * gap_extra_ratio)
    gap_length = max(gap_length, 8)

    h, w = image.shape[:2]
    scale_x = w / w_m
    scale_y = h / h_m

    bx_img = int(break_x * scale_x)
    by_img = int(break_y * scale_y)
    gap_length_img = int(gap_length * max(scale_x, scale_y))
    gap_width_img = int(gap_width * max(scale_x, scale_y))

    bg_color = sample_background_color(image, canvas.occupied_mask)
    bg_color = add_noise_to_color(bg_color, sigma=5.0, rng=rng)

    dx = int(gap_length_img * 0.5 * np.cos(perp_angle))
    dy = int(gap_length_img * 0.5 * np.sin(perp_angle))
    pt1 = (bx_img - dx, by_img - dy)
    pt2 = (bx_img + dx, by_img + dy)

    cv2.line(image, pt1, pt2, bg_color, gap_width_img)

    x1 = min(pt1[0], pt2[0]) - gap_width_img
    y1 = min(pt1[1], pt2[1]) - gap_width_img
    x2 = max(pt1[0], pt2[0]) + gap_width_img
    y2 = max(pt1[1], pt2[1]) + gap_width_img

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["open_circuit"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
