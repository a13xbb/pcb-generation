from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .types import DefectAnnotation, DefectResult, CLASS_NAMES
from .missing_hole import generate_missing_hole
from .mouse_bite import generate_mouse_bite
from .open_circuit import generate_open_circuit
# from .short import generate_short
from .spur import generate_spur
from .spurious_copper import generate_spurious_copper


def _overlaps_existing(
    new_ann: DefectAnnotation,
    existing: List[DefectAnnotation],
    padding: float = 0.01,
) -> bool:
    """Check if new annotation bbox overlaps with any existing one (with padding)."""
    for ann in existing:
        dx = abs(new_ann.center_x - ann.center_x)
        dy = abs(new_ann.center_y - ann.center_y)
        min_x_gap = (new_ann.width + ann.width) / 2 + padding
        min_y_gap = (new_ann.height + ann.height) / 2 + padding
        if dx < min_x_gap and dy < min_y_gap:
            return True
    return False

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


DEFECT_GENERATORS = {
    0: generate_mouse_bite,
    1: generate_spur,
    2: generate_missing_hole,
    3: generate_open_circuit,
    4: generate_spurious_copper,
    # 5: generate_short,
}


def add_defects(
    image: np.ndarray,
    canvas: "CanvasState",
    n_defects: int = 3,
    defect_weights: Optional[Dict[int, float]] = None,
    seed: int = 42,
    max_attempts_per_defect: int = 5,
) -> Tuple[np.ndarray, List[DefectAnnotation]]:
    rng = np.random.default_rng(seed)

    if defect_weights is None:
        defect_weights = {i: 1.0 for i in DEFECT_GENERATORS}

    class_ids = list(defect_weights.keys())
    weights = np.array([defect_weights[i] for i in class_ids])
    weights = weights / weights.sum()

    annotations: List[DefectAnnotation] = []

    for _ in range(n_defects):
        class_id = rng.choice(class_ids, p=weights)
        generator = DEFECT_GENERATORS[class_id]

        for attempt in range(max_attempts_per_defect):
            # Work on a copy so we can discard if overlap detected
            image_copy = image.copy()
            result = generator(image_copy, canvas, rng)
            if result.success and result.annotation is not None:
                if not _overlaps_existing(result.annotation, annotations):
                    # Commit changes to actual image
                    np.copyto(image, image_copy)
                    annotations.append(result.annotation)
                    break
                # else: discard image_copy, defect not committed

    return image, annotations


def save_yolo_labels(
    annotations: List[DefectAnnotation],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ann in annotations:
            f.write(ann.to_yolo_line() + "\n")
