from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "layout_generator"))

from utils import skeletonize, find_endpoints

from .colors import sample_trace_center_color, add_noise_to_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def _create_spur_shape(
    rng: np.random.Generator,
    center_x: int,
    center_y: int,
    outward_angle: float,
    spur_length: int,
    spur_width: int,
    img_shape: tuple,
) -> np.ndarray:
    h, w = img_shape
    spur_mask = np.zeros((h, w), dtype=np.uint8)

    shape_type = rng.choice(["semicircle", "trapezoid", "blob"])

    cos_a = np.cos(outward_angle)
    sin_a = np.sin(outward_angle)
    perp_cos = np.cos(outward_angle + np.pi / 2)
    perp_sin = np.sin(outward_angle + np.pi / 2)

    base_x = center_x
    base_y = center_y

    if shape_type == "semicircle":
        # Half-circle extending outward
        radius = max(3, int(spur_length * rng.uniform(0.7, 1.0)))
        half_width = max(3, int(spur_width * rng.uniform(0.4, 0.6)))

        # Center of the semicircle is at the base
        n_arc_points = 20
        points = []

        # Draw arc from one side to the other
        for i in range(n_arc_points):
            theta = -np.pi / 2 + np.pi * i / (n_arc_points - 1)
            # theta goes from -90 to +90 degrees (half circle)
            r = radius
            arc_x = r * np.cos(theta)  # 0 to r to 0
            arc_y = r * np.sin(theta)  # -r to 0 to +r

            # Transform: arc_x is along outward direction, arc_y is perpendicular
            px = int(base_x + arc_x * cos_a + arc_y * perp_cos * (half_width / radius))
            py = int(base_y + arc_x * sin_a + arc_y * perp_sin * (half_width / radius))
            points.append([px, py])

        pts = np.array(points, np.int32)
        cv2.fillPoly(spur_mask, [pts], 255)

    elif shape_type == "trapezoid":
        # Trapezoid: wider at base, narrower at tip
        length = max(3, int(spur_length * rng.uniform(0.8, 1.0)))
        base_half_width = max(3, int(spur_width * rng.uniform(0.4, 0.6)))
        tip_half_width = max(2, int(base_half_width * rng.uniform(0.3, 0.6)))

        # Four corners of trapezoid
        points = [
            # Base left
            (int(base_x + base_half_width * perp_cos),
             int(base_y + base_half_width * perp_sin)),
            # Base right
            (int(base_x - base_half_width * perp_cos),
             int(base_y - base_half_width * perp_sin)),
            # Tip right
            (int(base_x + length * cos_a - tip_half_width * perp_cos),
             int(base_y + length * sin_a - tip_half_width * perp_sin)),
            # Tip left
            (int(base_x + length * cos_a + tip_half_width * perp_cos),
             int(base_y + length * sin_a + tip_half_width * perp_sin)),
        ]

        pts = np.array(points, np.int32)
        cv2.fillPoly(spur_mask, [pts], 255)

    else:  # blob - organic irregular shape
        length = max(3, int(spur_length * rng.uniform(0.7, 1.0)))
        half_width = max(3, int(spur_width * rng.uniform(0.3, 0.5)))

        # Generate blob using multiple overlapping ellipses
        n_ellipses = rng.integers(2, 4)
        for i in range(n_ellipses):
            # Position along the spur direction
            t = rng.uniform(0.2, 0.8)
            cx = int(base_x + t * length * cos_a + rng.uniform(-0.2, 0.2) * half_width * perp_cos)
            cy = int(base_y + t * length * sin_a + rng.uniform(-0.2, 0.2) * half_width * perp_sin)

            # Random ellipse size
            axes = (
                max(3, int(half_width * rng.uniform(0.6, 1.2))),
                max(3, int(length * 0.3 * rng.uniform(0.5, 1.0)))
            )

            # Angle aligned with spur direction
            angle_deg = np.degrees(outward_angle) + rng.uniform(-15, 15)

            cv2.ellipse(spur_mask, (cx, cy), axes, angle_deg, 0, 360, 255, -1)

        # Also draw base ellipse to connect to trace
        base_axes = (max(3, int(half_width * 0.8)), max(2, int(length * 0.2)))
        cv2.ellipse(spur_mask, (base_x, base_y), base_axes,
                    np.degrees(outward_angle), 0, 360, 255, -1)

    return spur_mask


def generate_spur(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
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

    # Check for space to extend outward (not blocked by other traces)
    occupied = canvas.occupied_mask

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

    # Try multiple points to find one with space for spur
    indices = rng.permutation(len(valid_pts))
    selected_point = None
    half_width_canvas = None
    outward_angle = None

    h_m, w_m = mask.shape

    for idx in indices[:20]:
        sx, sy = valid_pts[idx]

        # Compute trace direction at this point
        neighborhood = 5
        y1 = max(0, sy - neighborhood)
        y2 = sy + neighborhood + 1
        x1 = max(0, sx - neighborhood)
        x2 = sx + neighborhood + 1
        ys, xs = np.where(skel[y1:y2, x1:x2] > 0)

        if len(xs) < 2:
            trace_angle = 0.0
        else:
            pts = np.column_stack([xs, ys])
            if len(pts) > 2:
                _, _, vt = np.linalg.svd(pts - pts.mean(axis=0))
                trace_angle = np.arctan2(vt[0, 1], vt[0, 0])
            else:
                trace_angle = 0.0

        # Try both perpendicular directions
        for sign in [1, -1]:
            perp_angle = trace_angle + sign * np.pi / 2

            # Measure half-width in this direction
            hw = 0
            for dist in range(1, 50):
                nx = int(sx + dist * np.cos(perp_angle))
                ny = int(sy + dist * np.sin(perp_angle))
                if not (0 <= nx < w_m and 0 <= ny < h_m):
                    break
                if mask[ny, nx] == 0:
                    break
                hw += 1

            if hw < 3:
                continue

            # Check if there's space beyond trace edge for at least minimum extension
            check_dist = hw + 20
            check_x = int(sx + check_dist * np.cos(perp_angle))
            check_y = int(sy + check_dist * np.sin(perp_angle))

            if not (0 <= check_x < w_m and 0 <= check_y < h_m):
                continue

            if occupied[check_y, check_x] > 0:
                continue

            selected_point = (sx, sy)
            half_width_canvas = hw
            outward_angle = perp_angle
            break

        if selected_point is not None:
            break

    if selected_point is None:
        return DefectResult(success=False)

    sx, sy = selected_point
    trace_width_canvas = 2 * half_width_canvas

    # Extension beyond trace edge: 50-100% of trace width, minimum 15 canvas px for visibility
    extension_canvas = max(15, int(trace_width_canvas * rng.uniform(0.5, 1.0)))

    h, w = image.shape[:2]
    scale_x = w / w_m
    scale_y = h / h_m
    avg_scale = (scale_x + scale_y) / 2

    # Anchor spur at the trace EDGE (not skeleton center) so it is always visually attached
    edge_x_img = int((sx + half_width_canvas * np.cos(outward_angle)) * scale_x)
    edge_y_img = int((sy + half_width_canvas * np.sin(outward_angle)) * scale_y)

    spur_length_img = max(9, int(extension_canvas * avg_scale))
    trace_width_img = max(1, int(trace_width_canvas * avg_scale))
    # Width along trace edge: varies from narrow (trace width) to wide (2.5x trace width)
    spur_width_img = max(12, min(45, int(trace_width_img * rng.uniform(0.8, 2.5))))

    # Create spur shape starting at trace edge, extending outward
    spur_mask = _create_spur_shape(
        rng, edge_x_img, edge_y_img, outward_angle, spur_length_img, spur_width_img, (h, w)
    )

    # Mask out areas that overlap with other traces (except the source trace)
    trace_mask_scaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    occupied_scaled = cv2.resize(occupied, (w, h), interpolation=cv2.INTER_NEAREST)
    other_traces_mask = (occupied_scaled > 0) & (trace_mask_scaled == 0)

    # Spur should not overlap other traces
    spur_mask = spur_mask & (~other_traces_mask).astype(np.uint8) * 255

    # Check spur has reasonable size
    spur_pixels = np.sum(spur_mask > 0)
    if spur_pixels < 20:
        return DefectResult(success=False)

    # Sample color from the exact center point on this trace
    trace_color = sample_trace_center_color(image, mask, skel, center_x=sx, center_y=sy)
    trace_color = add_noise_to_color(trace_color, sigma=3.0, rng=rng)

    # Create textured fill
    noise = rng.normal(0, 5.0, (h, w, 3))
    textured = np.clip(np.array(trace_color) + noise, 0, 255).astype(np.uint8)

    image[spur_mask > 0] = textured[spur_mask > 0]

    # Compute bounding box
    spur_ys, spur_xs = np.where(spur_mask > 0)
    if len(spur_xs) == 0:
        return DefectResult(success=False)

    x1 = int(spur_xs.min())
    y1 = int(spur_ys.min())
    x2 = int(spur_xs.max())
    y2 = int(spur_ys.max())

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["spur"], x1, y1, x2, y2, w, h,
        metadata={"extension_px": extension_canvas}
    )

    return DefectResult(success=True, annotation=annotation)
