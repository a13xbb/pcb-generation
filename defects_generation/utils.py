from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "layout_generator"))

from utils import skeletonize, find_endpoints, skeleton_path_length_between_endpoints

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


MIN_TRACE_LENGTH = 50


def get_trace_length(trace_mask: np.ndarray) -> float:
    skel = skeletonize(trace_mask)
    endpoints = find_endpoints(trace_mask, skel)
    if len(endpoints) != 2:
        ys, xs = np.where(trace_mask > 0)
        if len(xs) == 0:
            return 0.0
        return float(max(xs.max() - xs.min(), ys.max() - ys.min()))
    length = skeleton_path_length_between_endpoints(skel, endpoints[0], endpoints[1])
    return length if length is not None else 0.0


def filter_long_traces(traces: list, min_length: float = MIN_TRACE_LENGTH) -> list:
    return [t for t in traces if get_trace_length(t.mask_world) >= min_length]


def find_trace_edge_points(
    trace_mask: np.ndarray,
    sample_count: int = 20,
) -> List[Tuple[int, int, float]]:
    mask = (trace_mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel)
    edge = dilated - mask

    edge_ys, edge_xs = np.where(edge > 0)
    if len(edge_xs) == 0:
        return []

    if len(edge_xs) > sample_count:
        indices = np.linspace(0, len(edge_xs) - 1, sample_count, dtype=int)
        edge_xs = edge_xs[indices]
        edge_ys = edge_ys[indices]

    results = []
    for x, y in zip(edge_xs, edge_ys):
        normal_angle = compute_outward_normal(mask, x, y)
        results.append((int(x), int(y), normal_angle))

    return results


def find_trace_edge_points_away_from_pads(
    trace_mask: np.ndarray,
    canvas: "CanvasState",
    sample_count: int = 20,
    pad_buffer: int = 10,
) -> List[Tuple[int, int, float]]:
    pad_mask = np.zeros_like(trace_mask)
    for pad in canvas.pad_instances:
        pad_mask = np.maximum(pad_mask, pad.mask_world)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_buffer * 2 + 1, pad_buffer * 2 + 1))
    pad_zone = cv2.dilate(pad_mask, kernel)

    all_points = find_trace_edge_points(trace_mask, sample_count * 3)
    return [(x, y, angle) for x, y, angle in all_points if pad_zone[y, x] == 0]


def compute_outward_normal(mask: np.ndarray, x: int, y: int, radius: int = 5) -> float:
    h, w = mask.shape
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    patch = mask[y1:y2, x1:x2]

    if patch.sum() == 0:
        return 0.0

    ys, xs = np.where(patch > 0)
    cx = xs.mean() + x1
    cy = ys.mean() + y1

    dx = x - cx
    dy = y - cy

    return float(np.arctan2(dy, dx))


def get_trace_width_at_point(
    trace_mask: np.ndarray,
    point: Tuple[int, int],
    angle: float,
) -> float:
    x, y = point
    h, w = trace_mask.shape

    perp_angle = angle + np.pi / 2
    dx = np.cos(perp_angle)
    dy = np.sin(perp_angle)

    width = 0
    for direction in [1, -1]:
        for dist in range(1, 50):
            nx = int(x + direction * dist * dx)
            ny = int(y + direction * dist * dy)
            if not (0 <= nx < w and 0 <= ny < h):
                break
            if trace_mask[ny, nx] == 0:
                break
            width += 1

    return float(width)


def find_nearby_trace_pairs(
    traces: list,
    max_distance: int = 60,
) -> List[Tuple[int, int, Tuple[int, int], Tuple[int, int]]]:
    pairs = []
    n = len(traces)

    for i in range(n):
        for j in range(i + 1, n):
            mask_i = traces[i].mask_world
            mask_j = traces[j].mask_world

            pts_i = np.argwhere(mask_i > 0)
            pts_j = np.argwhere(mask_j > 0)

            if len(pts_i) == 0 or len(pts_j) == 0:
                continue

            sample_i = pts_i[::max(1, len(pts_i) // 50)]
            sample_j = pts_j[::max(1, len(pts_j) // 50)]

            min_dist = float('inf')
            best_pi, best_pj = None, None

            for pi in sample_i:
                dists = np.sqrt(np.sum((sample_j - pi) ** 2, axis=1))
                min_idx = np.argmin(dists)
                if dists[min_idx] < min_dist:
                    min_dist = dists[min_idx]
                    best_pi = pi
                    best_pj = sample_j[min_idx]

            if min_dist <= max_distance and best_pi is not None:
                pt_i = (int(best_pi[1]), int(best_pi[0]))
                pt_j = (int(best_pj[1]), int(best_pj[0]))
                pairs.append((i, j, pt_i, pt_j))

    return pairs


def find_background_region(
    occupied_mask: np.ndarray,
    min_area: int = 500,
    erosion_size: int = 15,
) -> Optional[Tuple[int, int, int, int]]:
    h, w = occupied_mask.shape
    bg_mask = (occupied_mask == 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    eroded = cv2.erode(bg_mask, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)

    valid_regions = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            rw = stats[label, cv2.CC_STAT_WIDTH]
            rh = stats[label, cv2.CC_STAT_HEIGHT]
            valid_regions.append((x, y, x + rw, y + rh, area))

    if not valid_regions:
        return None

    valid_regions.sort(key=lambda r: r[4], reverse=True)
    x1, y1, x2, y2, _ = valid_regions[0]
    return (x1, y1, x2, y2)


def expand_bbox(
    x1: int, y1: int, x2: int, y2: int,
    padding: int,
    img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(img_w, x2 + padding),
        min(img_h, y2 + padding),
    )
