"""Canvas-corrected binarization with local defect handling.

Approach:
1. Old binarization (baseline) - simple global threshold
2. Canvas correction - fill in missed copper using ground truth
3. Local defect regions - adaptive binarization to capture defect modifications
"""

import gzip
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# Add layout_generator to path for unpickling CanvasState
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "layout_generator") not in sys.path:
    sys.path.insert(0, str(_ROOT / "layout_generator"))


def extract_layout_id(img_path: str) -> int:
    """Extract layout number from filename like 'layout_42_rep3.png'."""
    match = re.search(r'layout_(\d+)(?:_rep|\.)', str(img_path))
    return int(match.group(1)) if match else -1


def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO format labels as list of (cls, cx, cy, w, h)."""
    labels = []
    path = Path(label_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append((
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4])
                    ))
    return labels


def get_label_path(img_path: str) -> str:
    """Convert image path to label path."""
    return str(img_path).replace("/images/", "/labels/").replace(".png", ".txt").replace(".jpg", ".txt")


def find_dataset_root(img_path: str) -> Path:
    """Auto-detect dataset root from image path."""
    img_path = Path(img_path)
    # img_path is like .../defects/images/layout_N_repR.png
    # dataset_root is .../
    if "defects" in img_path.parts:
        idx = img_path.parts.index("defects")
        return Path(*img_path.parts[:idx])
    return img_path.parent.parent


def load_canvas_mask(layout_id: int, target_size: Tuple[int, int], dataset_root: Path) -> np.ndarray:
    """Load and resize canvas occupied_mask.

    Args:
        layout_id: Layout number
        target_size: (width, height) to resize to
        dataset_root: Root of dataset containing canvases/

    Returns:
        Binary mask where copper=255, background=0
    """
    canvas_path = dataset_root / "canvases" / f"layout_{layout_id}.pkl.gz"
    if not canvas_path.exists():
        raise FileNotFoundError(f"Canvas not found: {canvas_path}")

    with gzip.open(canvas_path, 'rb') as f:
        canvas = pickle.load(f)

    # occupied_mask is at canvas resolution (typically 1000x1000)
    mask = (canvas.occupied_mask > 0).astype(np.uint8) * 255
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def binarize_synth_old(img: np.ndarray) -> np.ndarray:
    """Original simple binarization (conservative).

    Same as binarize_synth in augmentations.py.
    Returns grayscale binary: copper=0/black, bg=255/white
    """
    from collections import Counter

    h, w = img.shape[:2]

    # Find background color
    flat = img.reshape(-1, 3)
    colors_tuple = [tuple(c) for c in flat]
    color_counts = Counter(colors_tuple)
    bg_color = color_counts.most_common(1)[0][0]
    bg_gray = int(0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0])

    # Median filter for noise reduction
    filtered = cv2.medianBlur(img, 5)

    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, sat, _ = cv2.split(hsv)

    # Pads: low saturation (metallic silver)
    pad_mask = (sat < 70).astype(np.uint8) * 255

    # Traces: brighter than background
    trace_mask = (gray > bg_gray + 15).astype(np.uint8) * 255

    # Combine
    copper = cv2.bitwise_or(pad_mask, trace_mask)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    copper = cv2.morphologyEx(copper, cv2.MORPH_CLOSE, kernel)

    # Remove small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(copper, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            copper[labels == i] = 0

    # Mask edges
    border = 10
    edge_mask = np.zeros_like(copper)
    edge_mask[border:h-border, border:w-border] = 255
    copper = cv2.bitwise_and(copper, edge_mask)

    # DeepPCB style: invert (copper=black, background=white)
    result = cv2.bitwise_not(copper)

    # Clean border
    result[:border, :] = 255
    result[h-border:, :] = 255
    result[:, :border] = 255
    result[:, w-border:] = 255

    return result


def local_adaptive_binarize(region: np.ndarray) -> np.ndarray:
    """Adaptive binarization for small defect regions.

    Args:
        region: BGR color image region

    Returns:
        Grayscale binary: copper=0/black, bg=255/white
    """
    if region.size == 0:
        return np.ones(region.shape[:2], dtype=np.uint8) * 255

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # Pads: low saturation (metallic)
    pad_mask = (sat < 70).astype(np.uint8) * 255

    # Traces: adaptive threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    trace_mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=-5
    )

    copper = cv2.bitwise_or(pad_mask, trace_mask)

    # Light cleanup
    kernel = np.ones((2, 2), np.uint8)
    copper = cv2.morphologyEx(copper, cv2.MORPH_CLOSE, kernel)

    # Invert to DeepPCB style (copper=black, bg=white)
    return cv2.bitwise_not(copper)


def perfect_binarize(
    img_path: str,
    dataset_root: Optional[Path] = None,
    defect_pad: int = 10
) -> np.ndarray:
    """Canvas-corrected binarization with local defect handling.

    1. Old binarization (baseline)
    2. Canvas correction (fill missed copper)
    3. Local defect region processing

    Args:
        img_path: Path to defected image like "defects/images/layout_N_repR.png"
        dataset_root: Root of dataset (auto-detected if None)
        defect_pad: Padding around defect bboxes

    Returns:
        Grayscale binary image: copper=0/black, bg=255/white
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]

    # Auto-detect dataset root
    if dataset_root is None:
        dataset_root = find_dataset_root(str(img_path))
    dataset_root = Path(dataset_root)

    # Step 1: Old binarization (baseline)
    old_binary = binarize_synth_old(img)  # copper=0, bg=255

    # Step 2: Canvas correction
    layout_id = extract_layout_id(img_path.name)
    if layout_id < 0:
        # Can't load canvas, return old binarization
        return old_binary

    try:
        canvas_copper = load_canvas_mask(layout_id, (w, h), dataset_root)
        # canvas_copper: copper=255, bg=0

        corrected = old_binary.copy()
        # Fill missed copper: where canvas says copper (255) but old_binary says bg (255)
        missed_copper = (canvas_copper == 255) & (old_binary == 255)
        corrected[missed_copper] = 0  # Set to copper (black)
    except FileNotFoundError:
        # No canvas available, use old binarization
        corrected = old_binary

    # Step 3: Local defect regions
    label_path = get_label_path(str(img_path))
    defect_bboxes = load_yolo_labels(label_path)

    for cls_id, cx, cy, bw, bh in defect_bboxes:
        # Bbox with padding
        x1 = max(0, int((cx - bw / 2) * w) - defect_pad)
        y1 = max(0, int((cy - bh / 2) * h) - defect_pad)
        x2 = min(w, int((cx + bw / 2) * w) + defect_pad)
        y2 = min(h, int((cy + bh / 2) * h) + defect_pad)

        if x2 <= x1 or y2 <= y1:
            continue

        # Local binarization on original color image
        region = img[y1:y2, x1:x2]
        local_binary = local_adaptive_binarize(region)

        # Replace region in corrected image
        corrected[y1:y2, x1:x2] = local_binary

    return corrected


def perfect_binarize_bgr(
    img_path: str,
    dataset_root: Optional[Path] = None,
    defect_pad: int = 10
) -> np.ndarray:
    """Same as perfect_binarize but returns 3-channel BGR for YOLO."""
    binary = perfect_binarize(img_path, dataset_root, defect_pad)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def binarize_canvas_except_defects(
    img: np.ndarray,
    img_path: str,
    dataset_root: Path,
    labels: Optional[List] = None,
    defect_bbox_scale: float = 1.5
) -> np.ndarray:
    """Canvas-refined binarization EXCEPT defect regions.

    1. Baseline binarization
    2. Canvas refinement EXCEPT defect bboxes (those keep baseline)

    Args:
        img: BGR image
        img_path: Path to image (to extract layout_id)
        dataset_root: Root of dataset containing canvases/
        labels: Optional list of [cls, cx, cy, w, h] labels (if None, loads from file)
        defect_bbox_scale: Scale factor for defect bbox exclusion zones

    Returns:
        3-channel BGR binary image (for YOLO compatibility)
    """
    from augmentations import binarize_synth

    h, w = img.shape[:2]

    # Step 1: Baseline binarization
    baseline = binarize_synth(img)  # BGR
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

    # Step 2: Load canvas
    layout_id = extract_layout_id(str(img_path))
    if layout_id < 0:
        return baseline

    try:
        canvas_copper = load_canvas_mask(layout_id, (w, h), dataset_root)
    except FileNotFoundError:
        return baseline

    if canvas_copper is None:
        return baseline

    # Step 3: Create defect exclusion mask
    defect_mask = np.zeros((h, w), dtype=np.uint8)

    if labels is None:
        label_path = get_label_path(str(img_path))
        labels = load_yolo_labels(label_path)

    for label in labels:
        if len(label) >= 5:
            cx, cy, bw, bh = label[1], label[2], label[3], label[4]
        else:
            continue

        # Extend bbox
        bw_ext = bw * defect_bbox_scale
        bh_ext = bh * defect_bbox_scale

        x1 = max(0, int((cx - bw_ext / 2) * w))
        y1 = max(0, int((cy - bh_ext / 2) * h))
        x2 = min(w, int((cx + bw_ext / 2) * w))
        y2 = min(h, int((cy + bh_ext / 2) * h))

        defect_mask[y1:y2, x1:x2] = 255

    # Step 4: Canvas refinement EXCEPT defect regions
    result = baseline_gray.copy()

    # Where canvas says copper (255) AND baseline missed it (255) AND NOT in defect region
    missed_copper = (canvas_copper == 255) & (baseline_gray == 255) & (defect_mask == 0)
    result[missed_copper] = 0  # Set to copper (black)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
