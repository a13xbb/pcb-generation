import os
import cv2
import shutil
import numpy as np
from typing import List, Tuple

from classes import Pad, TraceSegment


def filter_pads(
    folder,
    trash_folder="trash",
    margin=10,
    mask_prefix="mask",
    object_prefix="object",
):
    os.makedirs(os.path.join(folder, trash_folder), exist_ok=True)

    files = os.listdir(folder)

    mask_files = sorted([
        f for f in files
        if f.startswith(mask_prefix)
    ])

    for mask_name in mask_files:
        suffix = mask_name[len(mask_prefix):]
        object_name = object_prefix + suffix

        mask_path = os.path.join(folder, mask_name)
        object_path = os.path.join(folder, object_name)

        # если парного object нет — пропускаем
        if not os.path.exists(object_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        ys, xs = np.where(mask > 0)

        # 1. пустая маска
        if len(xs) == 0:
            invalid = True
        else:
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            # 2. касание границы
            touches_border = (
                xmin == 0 or ymin == 0 or
                xmax == w - 1 or ymax == h - 1
            )

            # 3. попадание в margin-зону
            touches_margin = (
                xmin < margin or ymin < margin or
                xmax > w - margin - 1 or
                ymax > h - margin - 1
            )

            invalid = touches_border or touches_margin

        if invalid:
            print(f"[REMOVE] {mask_name}")

            shutil.move(
                mask_path,
                os.path.join(folder, trash_folder, mask_name)
            )
            shutil.move(
                object_path,
                os.path.join(folder, trash_folder, object_name)
            )


def split_trace_instances(
    folder,
    trash_folder="trash",
    mask_prefix="mask",
    object_prefix="object",
    min_area=20,
):
    os.makedirs(os.path.join(folder, trash_folder), exist_ok=True)

    files = os.listdir(folder)
    mask_files = sorted([
        f for f in files
        if f.startswith(mask_prefix)
    ])

    for mask_name in mask_files:
        suffix = mask_name[len(mask_prefix):]
        object_name = object_prefix + suffix

        mask_path = os.path.join(folder, mask_name)
        object_path = os.path.join(folder, object_name)

        if not os.path.exists(object_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        obj = cv2.imread(object_path, cv2.IMREAD_COLOR)

        if mask is None or obj is None:
            continue

        # бинаризация на всякий случай
        bin_mask = (mask > 0).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(
            bin_mask,
            connectivity=8
        )

        # фон = label 0
        if num_labels <= 2:
            comp_mask = (labels == 1).astype(np.uint8) * 255
            area = comp_mask.sum() // 255

            if area < min_area:
                shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
                shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
            continue

        base_name, ext = os.path.splitext(mask_name)

        comp_idx = 0
        for label in range(1, num_labels):
            comp_mask = (labels == label).astype(np.uint8) * 255
            area = comp_mask.sum() // 255

            if area < min_area:
                continue

            # bounding box (чтобы не сохранять пустоты)
            ys, xs = np.where(comp_mask > 0)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            comp_mask_cropped = comp_mask[ymin:ymax+1, xmin:xmax+1]
            comp_obj_cropped = obj[ymin:ymax+1, xmin:xmax+1]

            # применяем маску
            comp_obj_cropped = cv2.bitwise_and(
                comp_obj_cropped,
                comp_obj_cropped,
                mask=comp_mask_cropped
            )

            new_suffix = suffix.replace(
                ext, f"_{comp_idx}{ext}"
            )

            cv2.imwrite(
                os.path.join(folder, f"{mask_prefix}{new_suffix}"),
                comp_mask_cropped
            )
            cv2.imwrite(
                os.path.join(folder, f"{object_prefix}{new_suffix}"),
                comp_obj_cropped
            )

            comp_idx += 1

        # переносим исходные файлы в trash
        shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
        shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
        

def compute_bbox_and_centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    cx = xs.mean()
    cy = ys.mean()
    return (xmin, ymin, xmax, ymax), (cx, cy)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255

    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(mask)

    # fallback: грубая, но рабочая версия
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = mask.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        subset = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, subset)
        temp = eroded.copy()

        if cv2.countNonZero(temp) == 0:
            break

    return skel


def find_border_endpoints(
    mask: np.ndarray,
    skeleton: np.ndarray | None = None,
    min_component_size: int = 3,
) -> List[Tuple[int, int]]:
    """
    Возвращает список краевых endpoint'ов трассы.
    Каждый endpoint соответствует одной связной компоненте маски,
    касающейся границы изображения.
    """

    h, w = mask.shape
    border_mask = np.zeros_like(mask, dtype=np.uint8)

    # верх / низ
    border_mask[0, :] = mask[0, :]
    border_mask[h - 1, :] = mask[h - 1, :]

    # лево / право
    border_mask[:, 0] |= mask[:, 0]
    border_mask[:, w - 1] |= mask[:, w - 1]

    border_mask = (border_mask > 0).astype(np.uint8)

    if border_mask.sum() == 0:
        return []

    # связные компоненты на границе
    num_labels, labels = cv2.connectedComponents(border_mask, connectivity=8)

    endpoints = []

    if skeleton is not None:
        skel_pts = np.column_stack(np.where(skeleton > 0))
    else:
        skel_pts = np.empty((0, 2))

    for lbl in range(1, num_labels):
        ys, xs = np.where(labels == lbl)

        if len(xs) < min_component_size:
            continue

        comp_pts = np.column_stack([xs, ys])

        # если есть skeleton — берём ближайшую точку
        if len(skel_pts) > 0:
            dists = (
                (skel_pts[:, None, 1] - comp_pts[None, :, 0]) ** 2 +
                (skel_pts[:, None, 0] - comp_pts[None, :, 1]) ** 2
            )
            idx = np.unravel_index(np.argmin(dists), dists.shape)[1]
            endpoint = tuple(comp_pts[idx])
        else:
            # иначе — центр компоненты
            endpoint = (
                int(xs.mean()),
                int(ys.mean())
            )

        endpoints.append(endpoint)

    return endpoints


def prune_skeleton_endpoints(endpoints, max_keep=2):
    """
    Оставляет не более max_keep endpoints,
    выбирая наиболее удалённые друг от друга
    """
    if len(endpoints) <= max_keep:
        return endpoints

    pts = np.array(endpoints)

    # попарные расстояния
    dists = np.sum((pts[:, None] - pts[None, :]) ** 2, axis=-1)

    # ищем самую далёкую пару
    i, j = np.unravel_index(np.argmax(dists), dists.shape)

    return [tuple(pts[i]), tuple(pts[j])]


def find_endpoints(mask: np.ndarray, skeleton: np.ndarray):
    endpoints = []

    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape

    # 1. endpoints по скелету (degree == 1)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 0:
                continue
            neighbors = skel[y-1:y+2, x-1:x+2]
            if neighbors.sum() - 1 == 1:
                endpoints.append((x, y))

    # 2. endpoints по границе
    border_eps = find_border_endpoints(mask, skeleton)

    for bx, by in border_eps:
        if all((bx - x) ** 2 + (by - y) ** 2 > 9 for x, y in endpoints):
            endpoints.append((bx, by))
            
    endpoints = prune_skeleton_endpoints(endpoints)

    return endpoints


def visualize_endpoints(mask: np.ndarray, skeleton: np.ndarray):
    """
    Визуализация mask + skeleton + endpoints
    """

    endpoints = find_endpoints(mask, skeleton)

    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # mask — серый
    vis[mask > 0] = (120, 120, 120)

    # skeleton — белый
    vis[skeleton > 0] = (255, 255, 255)

    # рисуем endpoints
    for (x, y) in endpoints:
        # красный кружок
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

    return vis


def build_trace_segment(trace_id: str, mask: np.ndarray) -> TraceSegment:
    skeleton = skeletonize(mask)
    endpoints = find_endpoints(skeleton)

    return TraceSegment(
        id=trace_id,
        mask=mask,
        skeleton=skeleton,
        endpoints=endpoints
    )
