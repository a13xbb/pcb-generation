from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_dark_background_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces, find_trace_edge_points_away_from_pads, measure_trace_width_along_normal

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

    # Start bite from the edge point itself (edge detection places points
    # slightly outside trace, which is already accounted for)
    base_x = edge_x
    base_y = edge_y

    # Use bite_depth directly (already in image pixels)
    effective_depth = bite_depth

    if shape_type == "semi_ellipse":
        width_variation = rng.uniform(0.6, 1.2)
        depth_variation = rng.uniform(0.8, 1.0)

        half_width = max(3, int(bite_width * width_variation / 2))
        depth = max(3, int(effective_depth * depth_variation))

        # Create half-ellipse that only extends INWARD from base
        # Generate points along the arc from one side to the other
        n_arc_points = 15
        points = []
        for i in range(n_arc_points):
            # t goes from -1 to 1 (left edge to right edge of ellipse)
            t = -1 + 2 * i / (n_arc_points - 1)
            # Ellipse: at position t along width, depth is sqrt(1 - t^2) * max_depth
            arc_depth = depth * np.sqrt(max(0, 1 - t * t))
            arc_width = t * half_width
            # Convert to image coordinates
            px = int(base_x + arc_depth * cos_a + arc_width * perp_cos)
            py = int(base_y + arc_depth * sin_a + arc_width * perp_sin)
            points.append([px, py])

        pts = np.array(points, np.int32)
        cv2.fillPoly(bite_mask, [pts], 255)

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

    h, w = image.shape[:2]
    mask_h, mask_w = canvas.occupied_mask.shape
    scale_x = w / mask_w
    scale_y = h / mask_h

    # Try edge points until we find one where we can measure width
    indices = rng.permutation(len(edge_points))
    selected_point = None
    trace_width_canvas = None

    for idx in indices:
        ex, ey, normal_angle = edge_points[idx]
        inward_angle = normal_angle + np.pi
        width = measure_trace_width_along_normal(
            trace.mask_world, ex, ey, inward_angle
        )
        if width is not None and width >= 6:
            selected_point = (ex, ey, normal_angle)
            trace_width_canvas = width
            break

    if selected_point is None:
        return DefectResult(success=False)

    ex, ey, normal_angle = selected_point
    inward_angle = normal_angle + np.pi

    # Bite depth is 30-70% of measured trace width
    depth_fraction = rng.uniform(0.3, 0.7)
    bite_depth_canvas = int(trace_width_canvas * depth_fraction)

    # Scale to image pixels
    avg_scale = (scale_x + scale_y) / 2
    bite_depth_img = max(3, int(bite_depth_canvas * avg_scale))
    bite_width_img = rng.integers(min_bite_width, max_bite_width + 1)

    ex_img = int(ex * scale_x)
    ey_img = int(ey * scale_y)

    # Create varied bite shape
    bite_mask = _create_bite_shape(
        rng, ex_img, ey_img, inward_angle, bite_depth_img, bite_width_img, (h, w)
    )

    # Scale trace mask and dilate slightly to account for visual trace being wider
    trace_mask_scaled = cv2.resize(trace.mask_world, (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    trace_mask_dilated = cv2.dilate(trace_mask_scaled, kernel)
    bite_mask = ((bite_mask > 0) & (trace_mask_dilated > 0)).astype(np.uint8) * 255

    # Verify bite actually overlaps with original trace (not just dilated buffer)
    overlap_with_trace = np.sum((bite_mask > 0) & (trace_mask_scaled > 0))
    if overlap_with_trace < 5:
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
        CLASS_IDS["mouse_bite"], x1, y1, x2, y2, w, h,
        metadata={"depth_pct": int(depth_fraction * 100)}
    )

    return DefectResult(success=True, annotation=annotation)
