from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_trace_color, add_noise_to_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces, find_trace_edge_points

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def generate_spur(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    min_spur_length: int = 10,
    max_spur_length: int = 25,
    spur_width: int = 3,
) -> DefectResult:
    long_traces = filter_long_traces(canvas.trace_instances)
    if not long_traces:
        return DefectResult(success=False)

    trace = rng.choice(long_traces)

    edge_points = find_trace_edge_points(trace.mask_world, sample_count=30)
    if not edge_points:
        return DefectResult(success=False)

    valid_points = []
    h_m, w_m = trace.mask_world.shape
    occupied = canvas.occupied_mask

    for ex, ey, normal_angle in edge_points:
        check_dist = 20
        check_x = int(ex + check_dist * np.cos(normal_angle))
        check_y = int(ey + check_dist * np.sin(normal_angle))

        if not (0 <= check_x < w_m and 0 <= check_y < h_m):
            continue

        if occupied[check_y, check_x] == 0:
            valid_points.append((ex, ey, normal_angle))

    if not valid_points:
        return DefectResult(success=False)

    ex, ey, normal_angle = valid_points[rng.integers(len(valid_points))]

    h, w = image.shape[:2]
    scale_x = w / w_m
    scale_y = h / h_m

    start_x = int(ex * scale_x)
    start_y = int(ey * scale_y)

    spur_length = rng.integers(min_spur_length, max_spur_length + 1)
    spur_length_img = int(spur_length * max(scale_x, scale_y))

    end_x = int(start_x + spur_length_img * np.cos(normal_angle))
    end_y = int(start_y + spur_length_img * np.sin(normal_angle))

    trace_color = sample_trace_color(image, trace.mask_world)
    trace_color = add_noise_to_color(trace_color, sigma=5.0, rng=rng)

    spur_width_img = max(2, int(spur_width * max(scale_x, scale_y)))

    cv2.line(image, (start_x, start_y), (end_x, end_y), trace_color, spur_width_img)

    x1 = min(start_x, end_x) - spur_width_img
    y1 = min(start_y, end_y) - spur_width_img
    x2 = max(start_x, end_x) + spur_width_img
    y2 = max(start_y, end_y) + spur_width_img

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["spur"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
