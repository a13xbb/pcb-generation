from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_dark_background_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces, find_trace_edge_points_away_from_pads, get_trace_width_at_point

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def _create_bite_shape(
    rng: np.random.Generator,
    edge_x: int,
    edge_y: int,
    inward_angle: float,
    bite_depth: int,
    bite_width: int,
    img_shape: tuple,
) -> np.ndarray:
    h, w = img_shape
    bite_mask = np.zeros((h, w), dtype=np.uint8)

    shape_type = rng.choice(["semi_ellipse", "triangle", "irregular"])

    cos_a = np.cos(inward_angle)
    sin_a = np.sin(inward_angle)
    perp_cos = np.cos(inward_angle + np.pi / 2)
    perp_sin = np.sin(inward_angle + np.pi / 2)

    # Start bite from OUTSIDE the canvas edge to account for visual trace
    # being wider than canvas mask (due to diffusion). The bite will extend
    # inward from this point.
    outward_offset = 2  # pixels outside canvas edge
    base_x = int(edge_x - outward_offset * cos_a)
    base_y = int(edge_y - outward_offset * sin_a)

    # Add extra depth to reach into the visual trace from outside
    effective_depth = bite_depth + 4

    if shape_type == "semi_ellipse":
        width_variation = rng.uniform(0.6, 1.2)
        depth_variation = rng.uniform(0.8, 1.0)

        center_x = int(base_x + effective_depth * 0.5 * cos_a)
        center_y = int(base_y + effective_depth * 0.5 * sin_a)

        axes = (
            max(3, int(bite_width * width_variation / 2)),
            max(3, int(effective_depth * depth_variation / 2))
        )
        angle_deg = int(np.degrees(inward_angle))
        cv2.ellipse(bite_mask, (center_x, center_y), axes, angle_deg, 0, 360, 255, -1)

    elif shape_type == "triangle":
        tip_depth = effective_depth * rng.uniform(0.7, 0.95)
        base_width_left = bite_width * rng.uniform(0.25, 0.5)
        base_width_right = bite_width * rng.uniform(0.25, 0.5)

        tip_x = int(base_x + tip_depth * cos_a)
        tip_y = int(base_y + tip_depth * sin_a)

        left_x = int(base_x + base_width_left * perp_cos)
        left_y = int(base_y + base_width_left * perp_sin)
        right_x = int(base_x - base_width_right * perp_cos)
        right_y = int(base_y - base_width_right * perp_sin)

        pts = np.array([[tip_x, tip_y], [left_x, left_y], [right_x, right_y]], np.int32)
        cv2.fillPoly(bite_mask, [pts], 255)

    else:  # irregular
        n_points = rng.integers(4, 7)
        points = []

        # Base points along trace edge (outside)
        for i in range(3):
            t = (i - 1) * 0.3
            px = int(base_x + t * bite_width * perp_cos)
            py = int(base_y + t * bite_width * perp_sin)
            points.append([px, py])

        # Points extending inward
        for _ in range(n_points - 3):
            t = rng.uniform(-0.3, 0.3)
            depth_here = effective_depth * rng.uniform(0.5, 0.9)
            px = int(base_x + depth_here * cos_a + t * bite_width * perp_cos)
            py = int(base_y + depth_here * sin_a + t * bite_width * perp_sin)
            points.append([px, py])

        points = np.array(points, np.int32)
        hull = cv2.convexHull(points)
        cv2.fillPoly(bite_mask, [hull], 255)

    return bite_mask


def generate_mouse_bite(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    min_bite_width: int = 10,
    max_bite_width: int = 22,
) -> DefectResult:
    long_traces = filter_long_traces(canvas.trace_instances)
    if not long_traces:
        return DefectResult(success=False)

    trace = rng.choice(long_traces)

    edge_points = find_trace_edge_points_away_from_pads(trace.mask_world, canvas)
    if not edge_points:
        return DefectResult(success=False)

    ex, ey, normal_angle = edge_points[rng.integers(len(edge_points))]

    trace_width = get_trace_width_at_point(trace.mask_world, (ex, ey), normal_angle)
    if trace_width < 5:
        trace_width = 10

    h, w = image.shape[:2]
    mask_h, mask_w = canvas.occupied_mask.shape
    scale_x = w / mask_w
    scale_y = h / mask_h

    # Bite depth: 10-25% of trace width with hard cap at 6 pixels
    # (visual trace is often thinner than canvas mask due to diffusion)
    bite_depth_ratio = rng.uniform(0.10, 0.25)
    bite_depth = int(trace_width * bite_depth_ratio)
    bite_depth = max(2, min(bite_depth, 6))

    ex_img = int(ex * scale_x)
    ey_img = int(ey * scale_y)
    bite_depth_img = max(4, int(bite_depth * scale_y))
    bite_width_img = rng.integers(min_bite_width, max_bite_width + 1)

    inward_angle = normal_angle + np.pi

    # Create varied bite shape
    bite_mask = _create_bite_shape(
        rng, ex_img, ey_img, inward_angle, bite_depth_img, bite_width_img, (h, w)
    )

    # Scale trace mask and dilate to account for visual trace being wider
    trace_mask_scaled = cv2.resize(trace.mask_world, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    trace_mask_dilated = cv2.dilate(trace_mask_scaled, kernel)
    bite_mask = ((bite_mask > 0) & (trace_mask_dilated > 0)).astype(np.uint8) * 255

    if bite_mask.sum() == 0:
        return DefectResult(success=False)

    # Sample dark background color (darkest 20% of background pixels)
    dark_bg_color = sample_dark_background_color(image, canvas.occupied_mask, percentile=20.0)

    # Create textured fill with slight noise matching dark background
    noise = rng.normal(0, 5.0, (h, w, 3))
    textured = np.clip(np.array(dark_bg_color) + noise, 0, 255).astype(np.uint8)

    image[bite_mask > 0] = textured[bite_mask > 0]

    # Compute bounding box
    bite_ys, bite_xs = np.where(bite_mask > 0)
    x1 = int(bite_xs.min())
    y1 = int(bite_ys.min())
    x2 = int(bite_xs.max())
    y2 = int(bite_ys.max())

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["mouse_bite"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
