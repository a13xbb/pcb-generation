from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_background_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def generate_missing_hole(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    min_radius: int = 5,
    max_radius: int = 12,
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

    # Add slight offset from exact center (±2 pixels)
    offset_x = rng.integers(-2, 3)
    offset_y = rng.integers(-2, 3)
    cx_img += offset_x
    cy_img += offset_y

    radius = rng.integers(min_radius, max_radius + 1)

    # Irregular ellipse: random axis variation (±20%) and rotation
    axis_var = rng.uniform(0.8, 1.2, 2)
    axes = (int(radius * axis_var[0]), int(radius * axis_var[1]))
    angle = rng.uniform(0, 360)

    # Create defect mask
    defect_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(defect_mask, (cx_img, cy_img), axes, angle, 0, 360, 255, -1)

    # Generate noisy background texture
    bg_color = sample_background_color(image, canvas.occupied_mask)
    noise = rng.normal(0, 6.0, (h, w, 3))
    textured = np.clip(np.array(bg_color) + noise, 0, 255).astype(np.uint8)

    # Blur the texture for a more natural look
    textured = cv2.GaussianBlur(textured, (5, 5), 0)

    # Apply texture where defect mask is set
    image[defect_mask > 0] = textured[defect_mask > 0]

    # Compute bounding box from defect mask
    defect_ys, defect_xs = np.where(defect_mask > 0)
    if len(defect_xs) == 0:
        return DefectResult(success=False)

    x1 = int(defect_xs.min())
    y1 = int(defect_ys.min())
    x2 = int(defect_xs.max())
    y2 = int(defect_ys.max())

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["missing_hole"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
