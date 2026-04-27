from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}


@dataclass
class DefectAnnotation:
    class_id: int
    center_x: float  # normalized [0,1]
    center_y: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.center_x:.4f} {self.center_y:.4f} {self.width:.4f} {self.height:.4f}"

    @staticmethod
    def from_pixel_bbox(
        class_id: int,
        x1: int, y1: int, x2: int, y2: int,
        img_w: int, img_h: int,
    ) -> "DefectAnnotation":
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return DefectAnnotation(class_id, cx, cy, w, h)


@dataclass
class DefectResult:
    success: bool
    annotation: Optional[DefectAnnotation] = None
    affected_mask: Optional[np.ndarray] = None
