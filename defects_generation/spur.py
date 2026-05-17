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
    """Create spur shape extending only outward from trace center.

    Shapes: semi-sphere, semi-oval, rectangle, trapezoid
    All extend from (center_x, center_y) in the outward_angle direction only.
    """
    h, w = img_shape
    spur_mask = np.zeros((h, w), dtype=np.uint8)

    shape_type = rng.choice(["semi_sphere", "semi_oval", "rectangle", "trapezoid"])

    cos_a = np.cos(outward_angle)
    sin_a = np.sin(outward_angle)
    perp_cos = np.cos(outward_angle + np.pi / 2)
    perp_sin = np.sin(outward_angle + np.pi / 2)

    base_x = center_x
    base_y = center_y
    half_width = spur_width // 2

    if shape_type == "semi_sphere":
        # Half circle extending outward, radius = spur_length
        radius = max(3, spur_length)
        n_points = 30
        points = []

        # Arc from -90 to +90 degrees in outward direction
        for i in range(n_points + 1):
            theta = -np.pi / 2 + np.pi * i / n_points
            # Point on semicircle
            r_x = radius * np.cos(theta)  # 0 -> radius -> 0
            r_y = radius * np.sin(theta)  # -radius -> 0 -> +radius

            # Transform to world coordinates
            px = int(base_x + r_x * cos_a + r_y * perp_cos)
            py = int(base_y + r_x * sin_a + r_y * perp_sin)
            points.append([px, py])

        pts = np.array(points, np.int32)
        cv2.fillPoly(spur_mask, [pts], 255)

    elif shape_type == "semi_oval":
        # Half ellipse: length outward, width along trace
        length = max(3, spur_length)
        n_points = 30
        points = []

        for i in range(n_points + 1):
            theta = -np.pi / 2 + np.pi * i / n_points
            # Ellipse point
            r_x = length * np.cos(theta)  # outward extent
            r_y = half_width * np.sin(theta)  # along-trace extent

            px = int(base_x + r_x * cos_a + r_y * perp_cos)
            py = int(base_y + r_x * sin_a + r_y * perp_sin)
            points.append([px, py])

        pts = np.array(points, np.int32)
        cv2.fillPoly(spur_mask, [pts], 255)

    elif shape_type == "rectangle":
        # Rectangle: length outward, width along trace
        length = max(3, spur_length)
        use_rounded = rng.random() < 0.5

        if use_rounded:
            # Rounded rectangle using multiple points
            corner_r = max(2, min(length // 3, half_width // 3))
            points = []
            n_corner = 5

            # Four corners: base-left, base-right, tip-right, tip-left
            corners = [
                (0, -half_width, np.pi, 3*np.pi/2),      # base-left
                (0, half_width, np.pi/2, np.pi),         # base-right
                (length, half_width, 0, np.pi/2),        # tip-right
                (length, -half_width, 3*np.pi/2, 2*np.pi) # tip-left
            ]

            for (cx_local, cy_local, start_a, end_a) in corners:
                # Adjust corner center inward
                cx_adj = cx_local + (corner_r if cx_local == 0 else -corner_r)
                cy_adj = cy_local + (corner_r if cy_local < 0 else -corner_r)

                for j in range(n_corner):
                    a = start_a + (end_a - start_a) * j / (n_corner - 1)
                    lx = cx_adj + corner_r * np.cos(a)
                    ly = cy_adj + corner_r * np.sin(a)
                    px = int(base_x + lx * cos_a + ly * perp_cos)
                    py = int(base_y + lx * sin_a + ly * perp_sin)
                    points.append([px, py])

            pts = np.array(points, np.int32)
            cv2.fillPoly(spur_mask, [pts], 255)
        else:
            # Sharp rectangle
            points = [
                (int(base_x - half_width * perp_cos), int(base_y - half_width * perp_sin)),
                (int(base_x + half_width * perp_cos), int(base_y + half_width * perp_sin)),
                (int(base_x + length * cos_a + half_width * perp_cos),
                 int(base_y + length * sin_a + half_width * perp_sin)),
                (int(base_x + length * cos_a - half_width * perp_cos),
                 int(base_y + length * sin_a - half_width * perp_sin)),
            ]
            pts = np.array(points, np.int32)
            cv2.fillPoly(spur_mask, [pts], 255)

    else:  # trapezoid
        # Trapezoid: wider at base, narrower at tip
        length = max(3, spur_length)
        tip_ratio = rng.uniform(0.3, 0.7)  # tip is 30-70% of base width
        tip_half_width = max(2, int(half_width * tip_ratio))
        use_rounded = rng.random() < 0.5

        if use_rounded:
            # Rounded trapezoid
            corner_r = max(2, min(length // 4, tip_half_width // 2))
            points = []
            n_corner = 5

            # Base corners (wider)
            # Base-left corner
            for j in range(n_corner):
                a = np.pi + (np.pi/2) * j / (n_corner - 1)
                lx = corner_r + corner_r * np.cos(a)
                ly = -half_width + corner_r + corner_r * np.sin(a)
                px = int(base_x + lx * cos_a + ly * perp_cos)
                py = int(base_y + lx * sin_a + ly * perp_sin)
                points.append([px, py])

            # Base-right corner
            for j in range(n_corner):
                a = np.pi/2 + (np.pi/2) * j / (n_corner - 1)
                lx = corner_r + corner_r * np.cos(a)
                ly = half_width - corner_r + corner_r * np.sin(a)
                px = int(base_x + lx * cos_a + ly * perp_cos)
                py = int(base_y + lx * sin_a + ly * perp_sin)
                points.append([px, py])

            # Tip-right corner
            for j in range(n_corner):
                a = 0 + (np.pi/2) * j / (n_corner - 1)
                lx = length - corner_r + corner_r * np.cos(a)
                ly = tip_half_width - corner_r + corner_r * np.sin(a)
                px = int(base_x + lx * cos_a + ly * perp_cos)
                py = int(base_y + lx * sin_a + ly * perp_sin)
                points.append([px, py])

            # Tip-left corner
            for j in range(n_corner):
                a = 3*np.pi/2 + (np.pi/2) * j / (n_corner - 1)
                lx = length - corner_r + corner_r * np.cos(a)
                ly = -tip_half_width + corner_r + corner_r * np.sin(a)
                px = int(base_x + lx * cos_a + ly * perp_cos)
                py = int(base_y + lx * sin_a + ly * perp_sin)
                points.append([px, py])

            pts = np.array(points, np.int32)
            cv2.fillPoly(spur_mask, [pts], 255)
        else:
            # Sharp trapezoid
            points = [
                (int(base_x - half_width * perp_cos), int(base_y - half_width * perp_sin)),
                (int(base_x + half_width * perp_cos), int(base_y + half_width * perp_sin)),
                (int(base_x + length * cos_a + tip_half_width * perp_cos),
                 int(base_y + length * sin_a + tip_half_width * perp_sin)),
                (int(base_x + length * cos_a - tip_half_width * perp_cos),
                 int(base_y + length * sin_a - tip_half_width * perp_sin)),
            ]
            pts = np.array(points, np.int32)
            cv2.fillPoly(spur_mask, [pts], 255)

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

    h, w = image.shape[:2]
    scale_x = w / w_m
    scale_y = h / h_m
    avg_scale = (scale_x + scale_y) / 2
    trace_width_img = max(1, int(trace_width_canvas * avg_scale))

    # Anchor spur inside the trace (at skeleton center) so it grows outward through the edge
    anchor_x_img = int(sx * scale_x)
    anchor_y_img = int(sy * scale_y)

    # Spur length = half trace width (to reach edge) + outward extension (20-60% of trace width)
    outward_extension = trace_width_canvas * rng.uniform(0.2, 0.6)
    total_length_canvas = half_width_canvas + outward_extension
    spur_length_img = max(10, int(total_length_canvas * avg_scale))

    # Vary width along trace edge: 4-10x trace width (long along trace)
    spur_width_img = max(25, min(120, int(trace_width_img * rng.uniform(4.0, 10.0))))

    # Create spur shape extending outward from trace center (will pass through edge)
    spur_mask = _create_spur_shape(
        rng, anchor_x_img, anchor_y_img, outward_angle, spur_length_img, spur_width_img, (h, w)
    )

    # Create half-plane mask: only keep pixels on the outward side of the anchor
    # This prevents the spur from crossing to the opposite side of the trace
    yy, xx = np.mgrid[0:h, 0:w]
    # Vector from anchor to each pixel
    dx = xx - anchor_x_img
    dy = yy - anchor_y_img
    # Dot product with outward direction - positive means on outward side
    outward_dot = dx * np.cos(outward_angle) + dy * np.sin(outward_angle)
    # Allow a small margin into the trace for attachment (a few pixels)
    outward_side_mask = (outward_dot > -3).astype(np.uint8)
    spur_mask = spur_mask & (outward_side_mask * 255)

    # Mask out areas that overlap with other traces or pads
    trace_mask_scaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    occupied_scaled = cv2.resize(occupied, (w, h), interpolation=cv2.INTER_NEAREST)
    pad_zone_scaled = cv2.resize(pad_zone, (w, h), interpolation=cv2.INTER_NEAREST)

    # Spur should not overlap: other traces or pads
    other_traces_mask = (occupied_scaled > 0) & (trace_mask_scaled == 0)
    exclusion_mask = (other_traces_mask) | (pad_zone_scaled > 0)
    spur_mask = spur_mask & (~exclusion_mask).astype(np.uint8) * 255

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
        metadata={"length_px": spur_length_img}
    )

    return DefectResult(success=True, annotation=annotation)
