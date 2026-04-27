from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_trace_color, add_noise_to_color, generate_copper_texture
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces, find_nearby_trace_pairs

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def generate_short(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    bridge_width: int = 4,
    max_distance: int = 60,
) -> DefectResult:
    long_traces = filter_long_traces(canvas.trace_instances)
    if len(long_traces) < 2:
        return DefectResult(success=False)

    pairs = find_nearby_trace_pairs(long_traces, max_distance=max_distance)

    if not pairs:
        return DefectResult(success=False)

    idx = rng.integers(len(pairs))
    i, j, pt_i, pt_j = pairs[idx]

    h, w = image.shape[:2]
    mask_h, mask_w = canvas.occupied_mask.shape
    scale_x = w / mask_w
    scale_y = h / mask_h

    pt1 = (int(pt_i[0] * scale_x), int(pt_i[1] * scale_y))
    pt2 = (int(pt_j[0] * scale_x), int(pt_j[1] * scale_y))

    trace_color = sample_trace_color(image, long_traces[i].mask_world)
    trace_color = add_noise_to_color(trace_color, sigma=5.0, rng=rng)

    bridge_width_img = max(2, int(bridge_width * max(scale_x, scale_y)))

    cv2.line(image, pt1, pt2, trace_color, bridge_width_img)

    x1 = min(pt1[0], pt2[0]) - bridge_width_img
    y1 = min(pt1[1], pt2[1]) - bridge_width_img
    x2 = max(pt1[0], pt2[0]) + bridge_width_img
    y2 = max(pt1[1], pt2[1]) + bridge_width_img

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["short"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
