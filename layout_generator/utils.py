import os
import cv2
import shutil
import numpy as np
from typing import List, Tuple, Optional
import heapq

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

        # --- бинаризация ---
        bin_mask = (mask > 0).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(
            bin_mask,
            connectivity=8
        )

        # --- Если всего одна компонента ---
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

            # --- FULL CANVAS MASK ---
            full_mask = comp_mask.astype(np.uint8)

            # --- FULL CANVAS OBJECT ---
            full_obj = np.zeros_like(obj)
            full_obj[full_mask > 0] = obj[full_mask > 0]

            new_suffix = suffix.replace(
                ext, f"_{comp_idx}{ext}"
            )

            cv2.imwrite(
                os.path.join(folder, f"{mask_prefix}{new_suffix}"),
                full_mask
            )

            cv2.imwrite(
                os.path.join(folder, f"{object_prefix}{new_suffix}"),
                full_obj
            )

            comp_idx += 1

        # --- перенос исходников ---
        shutil.move(mask_path, os.path.join(folder, trash_folder, mask_name))
        shutil.move(object_path, os.path.join(folder, trash_folder, object_name))
        

def compute_bbox_and_centroid(mask: np.ndarray):
    # Приводим маску к 2D: убираем измерения размера 1 и обрабатываем 3D-случаи
    if mask.ndim == 3:
        # Вариант 1: если последнее измерение = 1 (например, (H, W, 1))
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        # Вариант 2: если несколько каналов — объединяем через логическое ИЛИ (для бинарной маски)
        else:
            mask = mask.any(axis=2)  # или mask = (mask.sum(axis=2) > 0)
    elif mask.ndim == 1:
        raise ValueError(f"Expected 2D/3D mask, got 1D array with shape {mask.shape}")
    elif mask.ndim != 2:
        raise ValueError(f"Unsupported mask dimensionality: {mask.ndim}D (shape: {mask.shape})")
    
    # Получаем координаты ненулевых пикселей
    ys, xs = np.where(mask > 0)
    
    # Проверка на пустую маску
    if len(xs) == 0:
        return None, None  # или raise ValueError("Empty mask: no foreground pixels found")
    
    # Вычисляем bbox и центроид
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    cx = xs.mean()
    cy = ys.mean()
    # print((xmin, ymin, xmax, ymax))
    # print(cx, cy)
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
    skeleton=None,
    min_component_size: int = 3,
) -> List[Tuple[int, int]]:

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

    num_labels, labels = cv2.connectedComponents(border_mask, connectivity=8)

    endpoints = []

    if skeleton is not None:
        skel = (skeleton > 0).astype(np.uint8)
        skel_pts = np.column_stack(np.where(skel > 0))  # (y, x)
    else:
        skel = None
        skel_pts = np.empty((0, 2))

    for lbl in range(1, num_labels):

        ys, xs = np.where(labels == lbl)

        if len(xs) < min_component_size:
            continue

        comp_pts_xy = np.column_stack([xs, ys])   # (x, y)
        comp_pts_yx = np.column_stack([ys, xs])   # (y, x)

        endpoint = None

        # -------------------------------------------------
        # CASE A: skeleton лежит прямо на границе
        # -------------------------------------------------
        if skel is not None:

            skel_on_comp_mask = skel[ys, xs] > 0

            if np.any(skel_on_comp_mask):
                skel_border_pts = comp_pts_xy[skel_on_comp_mask]

                # БЕРЁМ СРЕДНЮЮ ТОЧКУ СКЕЛЕТА НА ГРАНИЦЕ
                endpoint = (
                    int(np.mean(skel_border_pts[:, 0])),
                    int(np.mean(skel_border_pts[:, 1]))
                )

        # -------------------------------------------------
        # CASE B: скелет есть, но не на границе → проекция
        # -------------------------------------------------
        if endpoint is None and len(skel_pts) > 0:

            comp_pts = comp_pts_xy

            dists = (
                (skel_pts[:, None, 1] - comp_pts[None, :, 0]) ** 2 +
                (skel_pts[:, None, 0] - comp_pts[None, :, 1]) ** 2
            )

            idx = np.unravel_index(np.argmin(dists), dists.shape)[1]
            endpoint = tuple(comp_pts[idx])

        # -------------------------------------------------
        # CASE C: нет скелета → центр компоненты
        # -------------------------------------------------
        if endpoint is None:
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


def skeleton_path_length_between_endpoints(
    skeleton: np.ndarray,
    ep1: Tuple[int, int],   # (x, y)
    ep2: Tuple[int, int],   # (x, y)
) -> Optional[float]:
    """
    Считает длину пути по skeleton между двумя endpoint'ами.
    Возвращает длину в пикселях (диагональ = sqrt(2)).
    Если путь не найден — возвращает None.
    """

    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape

    x1, y1 = ep1
    x2, y2 = ep2

    if not (0 <= x1 < w and 0 <= y1 < h): 
        return None
    if not (0 <= x2 < w and 0 <= y2 < h): 
        return None

    if skel[y1, x1] == 0 or skel[y2, x2] == 0:
        return None

    # 8-connected moves
    neighbors = [
        (-1, -1, np.sqrt(2)), (0, -1, 1.0), (1, -1, np.sqrt(2)),
        (-1,  0, 1.0),                         (1,  0, 1.0),
        (-1,  1, np.sqrt(2)), (0,  1, 1.0), (1,  1, np.sqrt(2)),
    ]

    dist = np.full((h, w), np.inf, dtype=np.float64)
    dist[y1, x1] = 0.0

    pq = [(0.0, x1, y1)]

    while pq:
        d, x, y = heapq.heappop(pq)

        if (x, y) == (x2, y2):
            return float(d)

        if d > dist[y, x]:
            continue

        for dx, dy, cost in neighbors:
            nx = x + dx
            ny = y + dy

            if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                nd = d + cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))

    return None