from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np


CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "open_circuit",
    4: "spurious_copper",
    # 5: "short",
}

CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

# Bbox padding to match real-dataset annotation style (real annotators add margin).
# Values: (width_pad, height_pad) per side, derived from (real_median - synth_median) / 2
CLASS_BBOX_PAD = {
    0: (0.0134, 0.0192),  # mouse_bite
    1: (0.0158, 0.0142),  # spur
    2: (0.0092, 0.0100),  # missing_hole
    3: (0.0075, 0.0067),  # open_circuit
    4: (0.0075, 0.0067),  # spurious_copper
}


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

    def padded(self) -> "DefectAnnotation":
        """Return a copy with bbox expanded to match real-dataset annotation style."""
        pw, ph = CLASS_BBOX_PAD.get(self.class_id, (0.0, 0.0))
        new_w = min(self.width + 2 * pw, 1.0)
        new_h = min(self.height + 2 * ph, 1.0)
        new_cx = max(new_w / 2, min(self.center_x, 1.0 - new_w / 2))
        new_cy = max(new_h / 2, min(self.center_y, 1.0 - new_h / 2))
        return DefectAnnotation(self.class_id, new_cx, new_cy, new_w, new_h, self.metadata.copy())

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
