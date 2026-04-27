from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_trace_color, add_noise_to_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import find_background_region

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def generate_spurious_copper(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    min_blob_size: int = 15,
    max_blob_size: int = 40,
) -> DefectResult:
    region = find_background_region(canvas.occupied_mask, min_area=500, erosion_size=20)

    if region is None:
        return DefectResult(success=False)

    rx1, ry1, rx2, ry2 = region

    h, w = image.shape[:2]
    mask_h, mask_w = canvas.occupied_mask.shape
    scale_x = w / mask_w
    scale_y = h / mask_h

    rx1_img = int(rx1 * scale_x)
    ry1_img = int(ry1 * scale_y)
    rx2_img = int(rx2 * scale_x)
    ry2_img = int(ry2 * scale_y)

    margin = max_blob_size
    cx = rng.integers(rx1_img + margin, max(rx1_img + margin + 1, rx2_img - margin))
    cy = rng.integers(ry1_img + margin, max(ry1_img + margin + 1, ry2_img - margin))

    trace_color = sample_trace_color(image, canvas.occupied_mask)
    trace_color = add_noise_to_color(trace_color, sigma=8.0, rng=rng)

    shape_type = rng.choice(["ellipse", "rect", "irregular"])

    if shape_type == "ellipse":
        axis1 = rng.integers(min_blob_size // 2, max_blob_size // 2 + 1)
        axis2 = rng.integers(min_blob_size // 2, max_blob_size // 2 + 1)
        angle = rng.integers(0, 180)
        cv2.ellipse(image, (cx, cy), (axis1, axis2), angle, 0, 360, trace_color, -1)
        half_size = max(axis1, axis2)
        x1, y1 = cx - half_size, cy - half_size
        x2, y2 = cx + half_size, cy + half_size

    elif shape_type == "rect":
        rw = rng.integers(min_blob_size, max_blob_size + 1)
        rh = rng.integers(min_blob_size // 2, max_blob_size // 2 + 1)
        angle = rng.uniform(0, 360)

        rect_pts = np.array([
            [-rw / 2, -rh / 2],
            [rw / 2, -rh / 2],
            [rw / 2, rh / 2],
            [-rw / 2, rh / 2],
        ])

        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rect_pts = rect_pts @ rot_matrix.T
        rect_pts = rect_pts + [cx, cy]
        rect_pts = rect_pts.astype(np.int32)

        cv2.fillPoly(image, [rect_pts], trace_color)

        x1 = rect_pts[:, 0].min()
        y1 = rect_pts[:, 1].min()
        x2 = rect_pts[:, 0].max()
        y2 = rect_pts[:, 1].max()

    else:
        n_pts = rng.integers(4, 8)
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
        radii = rng.uniform(min_blob_size / 2, max_blob_size / 2, n_pts)

        pts = np.zeros((n_pts, 2), dtype=np.int32)
        for i, (a, r) in enumerate(zip(angles, radii)):
            pts[i, 0] = int(cx + r * np.cos(a))
            pts[i, 1] = int(cy + r * np.sin(a))

        cv2.fillPoly(image, [pts], trace_color)

        x1 = pts[:, 0].min()
        y1 = pts[:, 1].min()
        x2 = pts[:, 0].max()
        y2 = pts[:, 1].max()

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["spurious_copper"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
