from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_background_color, add_noise_to_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def generate_missing_hole(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    min_radius: int = 5,
    max_radius: int = 15,
) -> DefectResult:
    if not canvas.pad_instances:
        return DefectResult(success=False)

    pad = rng.choice(canvas.pad_instances)

    ys, xs = np.where(pad.mask_world > 0)
    if len(xs) == 0:
        return DefectResult(success=False)

    cx = int(xs.mean())
    cy = int(ys.mean())

    h, w = image.shape[:2]
    mask_h, mask_w = canvas.occupied_mask.shape
    scale_x = w / mask_w
    scale_y = h / mask_h
    cx_img = int(cx * scale_x)
    cy_img = int(cy * scale_y)

    bg_color = sample_background_color(image, canvas.occupied_mask)
    bg_color = add_noise_to_color(bg_color, sigma=5.0, rng=rng)

    radius = rng.integers(min_radius, max_radius + 1)

    defect_type = rng.choice(["circle", "line"])

    if defect_type == "circle":
        cv2.circle(image, (cx_img, cy_img), radius, bg_color, -1)
        x1 = cx_img - radius
        y1 = cy_img - radius
        x2 = cx_img + radius
        y2 = cy_img + radius
    else:
        angle = rng.uniform(0, np.pi)
        dx = int(radius * np.cos(angle))
        dy = int(radius * np.sin(angle))
        pt1 = (cx_img - dx, cy_img - dy)
        pt2 = (cx_img + dx, cy_img + dy)
        thickness = rng.integers(2, 5)
        cv2.line(image, pt1, pt2, bg_color, thickness)
        x1 = min(pt1[0], pt2[0]) - thickness
        y1 = min(pt1[1], pt2[1]) - thickness
        x2 = max(pt1[0], pt2[0]) + thickness
        y2 = max(pt1[1], pt2[1]) + thickness

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["missing_hole"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
