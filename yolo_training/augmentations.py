"""Custom augmentations for grayscale PCB defect training."""

from collections import Counter
from typing import Tuple, List, Optional

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale (3-channel for YOLO compatibility)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def find_background_color(img: np.ndarray) -> Tuple[int, int, int]:
    """Find the most common color (background/solder mask)."""
    flat = img.reshape(-1, 3)
    colors_tuple = [tuple(c) for c in flat]
    color_counts = Counter(colors_tuple)
    return color_counts.most_common(1)[0][0]


def binarize_real(img: np.ndarray) -> np.ndarray:
    """Binarize a real PCB image to DeepPCB-style binary (copper=black, background=white).

    CLAHE on grayscale enhances local contrast so Otsu reliably separates copper
    from the solder-mask substrate regardless of lighting variation across the board.
    Run on the full image before cropping — the global threshold is stable;
    per-patch thresholds are not.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(blurred)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def binarize_real_edges(img: np.ndarray) -> np.ndarray:
    """Binarize a real PCB image using edge detection + flood fill.

    1. Canny edges (auto thresholds from image median).
    2. Dilate to seal small gaps between copper outline segments.
    3. Flood-fill background from a padded border so the background is
       connected regardless of image corners.
    4. Pixels not reached by flood fill (enclosed by edges) = copper → black.

    Returns 3-channel (copper=black, background=white), same convention as
    binarize_real / binarize_synth.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    med = float(np.median(blurred))
    edges = cv2.Canny(blurred, max(0.0, 0.5 * med), min(255.0, 1.5 * med))

    # Dilate to close small gaps in copper outlines
    kernel = np.ones((5, 5), np.uint8)
    thick = cv2.dilate(edges, kernel, iterations=1)

    # Build canvas: non-edge pixels are 255 (traversable), edge pixels are 0 (walls)
    canvas = np.where(thick > 0, 0, 255).astype(np.uint8)

    # Add a 1-pixel white border so the flood fill is guaranteed to start in background
    h, w = canvas.shape
    padded = cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, mask, (0, 0), 128)   # background → 128
    interior = padded[1:h + 1, 1:w + 1]        # remove padding

    # 128 = background → white; everything else (edges + enclosed copper) → black
    result = np.where(interior == 128, 255, 0).astype(np.uint8)

    # Remove specks smaller than a small pad area
    k_open = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k_open)

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def binarize_synth(img: np.ndarray) -> np.ndarray:
    """Convert synthetic PCB image to DeepPCB-style binary.

    Adapted from scripts/synth_to_deeppcb.py:convert_to_deeppcb_v8
    """
    h, w = img.shape[:2]

    # Find background color
    bg_color = find_background_color(img)
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

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def random_crop_zoom(
    img: np.ndarray,
    labels: np.ndarray,
    scale_range: Tuple[float, float] = (0.3, 0.7)
) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop (zoom-in) with bounding box adjustment.

    Args:
        img: BGR image (H, W, 3)
        labels: YOLO format labels (N, 5) - [class_id, cx, cy, w, h] normalized
        scale_range: (min_scale, max_scale) for crop size relative to image

    Returns:
        (cropped_img, adjusted_labels)
    """
    h, w = img.shape[:2]

    # Random crop scale
    scale = np.random.uniform(scale_range[0], scale_range[1])
    crop_h, crop_w = int(h * scale), int(w * scale)

    # Random top-left corner
    y = np.random.randint(0, h - crop_h + 1)
    x = np.random.randint(0, w - crop_w + 1)

    # Crop image
    cropped = img[y:y+crop_h, x:x+crop_w]

    # Resize back to original size
    cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # Adjust bounding boxes
    if len(labels) == 0:
        return cropped, labels

    adjusted_labels = []
    for label in labels:
        cls_id, cx, cy, bw, bh = label

        # Convert to absolute coordinates
        abs_cx = cx * w
        abs_cy = cy * h
        abs_w = bw * w
        abs_h = bh * h

        # Calculate box corners
        x1 = abs_cx - abs_w / 2
        y1 = abs_cy - abs_h / 2
        x2 = abs_cx + abs_w / 2
        y2 = abs_cy + abs_h / 2

        # Clip to crop region
        x1_crop = max(x1, x) - x
        y1_crop = max(y1, y) - y
        x2_crop = min(x2, x + crop_w) - x
        y2_crop = min(y2, y + crop_h) - y

        # Check if box is still valid (at least partially in crop)
        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            continue

        # Check that MOST of the original defect is visible (at least 70%)
        new_w = x2_crop - x1_crop
        new_h = y2_crop - y1_crop
        original_area = abs_w * abs_h
        visible_area = new_w * new_h
        if visible_area < original_area * 0.7:
            continue

        # Also ensure the center of original bbox is inside the crop
        # (avoids edge cases where corner is visible but defect isn't)
        if not (x <= abs_cx <= x + crop_w and y <= abs_cy <= y + crop_h):
            continue

        # Scale to resized image coordinates
        scale_x = w / crop_w
        scale_y = h / crop_h

        new_x1 = x1_crop * scale_x
        new_y1 = y1_crop * scale_y
        new_x2 = x2_crop * scale_x
        new_y2 = y2_crop * scale_y

        # Convert back to YOLO format (normalized)
        new_cx = (new_x1 + new_x2) / 2 / w
        new_cy = (new_y1 + new_y2) / 2 / h
        new_bw = (new_x2 - new_x1) / w
        new_bh = (new_y2 - new_y1) / h

        # Clip to [0, 1]
        new_cx = np.clip(new_cx, 0, 1)
        new_cy = np.clip(new_cy, 0, 1)
        new_bw = np.clip(new_bw, 0, 1)
        new_bh = np.clip(new_bh, 0, 1)

        adjusted_labels.append([cls_id, new_cx, new_cy, new_bw, new_bh])

    return cropped, np.array(adjusted_labels) if adjusted_labels else np.array([])


def sample_crop_params(
    h: int,
    w: int,
    scale_range: Tuple[float, float] = (0.3, 0.7)
) -> Tuple[int, int, int, int]:
    """Return (y, x, crop_h, crop_w) for a random zoom-in crop."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    crop_h, crop_w = int(h * scale), int(w * scale)
    y = np.random.randint(0, h - crop_h + 1)
    x = np.random.randint(0, w - crop_w + 1)
    return y, x, crop_h, crop_w


def apply_crop_zoom(
    img: np.ndarray,
    labels: np.ndarray,
    y: int,
    x: int,
    crop_h: int,
    crop_w: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply pre-computed crop params to img+labels (same logic as random_crop_zoom).

    Use sample_crop_params to get (y, x, crop_h, crop_w), then call this on
    multiple images sharing the same source crop so their labels stay in sync.
    """
    h, w = img.shape[:2]

    cropped = img[y:y + crop_h, x:x + crop_w]
    cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    if len(labels) == 0:
        return cropped, labels

    adjusted_labels = []
    for label in labels:
        cls_id, cx, cy, bw, bh = label

        abs_cx = cx * w
        abs_cy = cy * h
        abs_w = bw * w
        abs_h = bh * h

        x1 = abs_cx - abs_w / 2
        y1 = abs_cy - abs_h / 2
        x2 = abs_cx + abs_w / 2
        y2 = abs_cy + abs_h / 2

        x1_crop = max(x1, x) - x
        y1_crop = max(y1, y) - y
        x2_crop = min(x2, x + crop_w) - x
        y2_crop = min(y2, y + crop_h) - y

        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            continue

        new_w = x2_crop - x1_crop
        new_h = y2_crop - y1_crop
        if new_w * new_h < abs_w * abs_h * 0.7:
            continue

        if not (x <= abs_cx <= x + crop_w and y <= abs_cy <= y + crop_h):
            continue

        scale_x = w / crop_w
        scale_y = h / crop_h

        new_cx = np.clip((x1_crop + x2_crop) / 2 * scale_x / w, 0, 1)
        new_cy = np.clip((y1_crop + y2_crop) / 2 * scale_y / h, 0, 1)
        new_bw = np.clip((x2_crop - x1_crop) * scale_x / w, 0, 1)
        new_bh = np.clip((y2_crop - y1_crop) * scale_y / h, 0, 1)

        adjusted_labels.append([cls_id, new_cx, new_cy, new_bw, new_bh])

    return cropped, np.array(adjusted_labels) if adjusted_labels else np.array([])


class GrayscaleAugmentor:
    """Augmentor for grayscale training with binarization and zoom."""

    def __init__(
        self,
        binarize_prob: float = 0.25,
        crop_prob: float = 0.25,
        crop_scale_range: Tuple[float, float] = (0.3, 0.7)
    ):
        self.binarize_prob = binarize_prob
        self.crop_prob = crop_prob
        self.crop_scale_range = crop_scale_range

    def __call__(
        self,
        img: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations.

        Args:
            img: BGR image (color)
            labels: YOLO format labels (N, 5) or None

        Returns:
            (augmented_img, adjusted_labels)
        """
        # Keep original color image for potential binarization
        color_img = img.copy()

        # Random crop (zoom-in) - apply before color transforms
        if labels is not None and np.random.random() < self.crop_prob:
            color_img, labels = random_crop_zoom(color_img, labels, self.crop_scale_range)

        # Decide: binarize OR grayscale
        if np.random.random() < self.binarize_prob:
            # Binarization needs color image to detect pads/traces
            img = binarize_synth(color_img)
        else:
            # Simple grayscale conversion
            img = to_grayscale(color_img)

        return img, labels
