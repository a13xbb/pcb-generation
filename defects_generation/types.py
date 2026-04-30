from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np


CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "open_circuit",
    # 4: "spurious_copper",
    # 5: "short",
}

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}


@dataclass
class DefectAnnotation:
    class_id: int
    center_x: float  # normalized [0,1]
    center_y: float
    width: float
    height: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.center_x:.4f} {self.center_y:.4f} {self.width:.4f} {self.height:.4f}"

    @staticmethod
    def from_pixel_bbox(
        class_id: int,
        x1: int, y1: int, x2: int, y2: int,
        img_w: int, img_h: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DefectAnnotation":
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return DefectAnnotation(class_id, cx, cy, w, h, metadata or {})


@dataclass
class DefectResult:
    success: bool
    annotation: Optional[DefectAnnotation] = None
    affected_mask: Optional[np.ndarray] = None
