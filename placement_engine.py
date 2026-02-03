import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

from classes import *

def transform_mask(mask, angle, tx, ty, canvas_shape):
    h, w = mask.shape

    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[:, 2] += [tx - center[0], ty - center[1]]

    warped = cv2.warpAffine(
        mask,
        M,
        (canvas_shape[1], canvas_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    return warped, M


def transform_mask(mask, angle, tx, ty, canvas_shape, centroid=None):
    """
    Трансформирует маску так, чтобы её ЦЕНТРОИД оказался в точке (tx, ty) на холсте.
    
    Args:
        mask: бинарная маска (H, W)
        angle: угол поворота в градусах
        tx, ty: целевые координаты ЦЕНТРОИДА на холсте
        canvas_shape: (H_canvas, W_canvas)
        centroid: (cx, cy) — центроид в координатах исходной маски (опционально, если не передан — вычисляется)
    """
    h, w = mask.shape[:2]
    
    # Если центроид не передан — вычисляем
    if centroid is None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros(canvas_shape, dtype=mask.dtype), None
        cx, cy = xs.mean(), ys.mean()
    else:
        cx, cy = centroid  # ← ИСПОЛЬЗУЕМ СОХРАНЁННЫЙ ЦЕНТРОИД ИЗ PadAsset!
    
    # 1. Поворачиваем ВОКРУГ ЦЕНТРОИДА
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # 2. Смещаем так, чтобы центроид оказался в (tx, ty)
    M[0, 2] += tx - cx
    M[1, 2] += ty - cy
    
    # 3. Применяем трансформацию
    warped = cv2.warpAffine(
        mask,
        M,
        (canvas_shape[1], canvas_shape[0]),  # (width, height)
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )
    
    return warped, M


def transform_points(points, M):
    pts = np.array(points, dtype=np.float32)
    pts = np.hstack([pts, np.ones((len(pts), 1))])
    out = pts @ M.T
    return [tuple(map(int, p)) for p in out]


def check_collision(canvas_mask, new_mask):
    return np.any((canvas_mask > 0) & (new_mask > 0))


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()  # x1,y1,x2,y2


def _bbox_distance(b1, b2):
    """
    Минимальная дистанция между bbox.
    Если пересекаются → 0
    """
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    dx = max(x21 - x12, x11 - x22, 0)
    dy = max(y21 - y12, y11 - y22, 0)

    return np.sqrt(dx * dx + dy * dy)


def place_pad_random(
    canvas: CanvasState,
    pad_asset: PadAsset,
    padding: int = 10,
    min_asset_distance: int = 5,
    max_tries: int = 50):
    
    for _ in range(max_tries):
        angle = random.choice([0, 90, 180, 270])

        tx = np.random.randint(padding, canvas.w - padding)
        ty = np.random.randint(padding, canvas.h - padding)

        mask_world, M = transform_mask(
            pad_asset.mask,
            angle,
            tx,
            ty,
            (canvas.h, canvas.w),
            centroid=pad_asset.centroid  # ← КЛЮЧЕВАЯ СТРОКА
        )
        
        if mask_world.ndim == 3:
        # Вариант 1: если последнее измерение = 1 (например, (H, W, 1))
            if mask_world.shape[2] == 1:
                mask_world = mask_world[:, :, 0]
            # Вариант 2: если несколько каналов — объединяем через логическое ИЛИ (для бинарной маски)
            else:
                mask_world = mask_world.any(axis=2)  # или mask = (mask.sum(axis=2) > 0)
        elif mask_world.ndim == 1:
            raise ValueError(f"Expected 2D/3D mask, got 1D array with shape {mask_world.shape}")
        elif mask_world.ndim != 2:
            raise ValueError(f"Unsupported mask dimensionality: {mask_world.ndim}D (shape: {mask_world.shape})")

        if check_collision(canvas.occupied_mask, mask_world):
            continue

        bbox = _bbox_from_mask(mask_world)
        if bbox is None:
            continue

        if min_asset_distance > 0:
            too_close = False

            for inst in canvas.pad_instances:
                other_bbox = _bbox_from_mask(inst.mask_world)
                if other_bbox is None:
                    continue

                dist = _bbox_distance(bbox, other_bbox)
                if dist < min_asset_distance:
                    too_close = True
                    break

            if too_close:
                continue

        # --- 7. Размещение ---
        canvas.occupied_mask |= mask_world

        inst = PadInstance(
            pad_asset,
            Transform(angle, tx, ty),
            mask_world
        )

        canvas.pad_instances.append(inst)

        return inst

    return None


def visualize_canvas(canvas: CanvasState, title="Canvas"):
    img = canvas.occupied_mask.astype(np.uint8) * 255

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for inst in canvas.pad_instances:
        bbox = _bbox_from_mask(inst.mask_world)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        # bbox — зеленый
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # centroid — красный
        ys, xs = np.where(inst.mask_world > 0)
        if len(xs) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
            cv2.circle(vis, (cx, cy), 2, (255, 0, 0), -1)

    cv2.imwrite("pads_placement.png", vis)
    
    
def test_pad_placement(
    pad_assets_list: List[PadAsset],
    canvas_w=512,
    canvas_h=512,
    n_pads=20,
    padding=20,
    min_distance=10,
):
    canvas = CanvasState(canvas_w, canvas_h)

    placed = 0

    for i in range(n_pads):
        pad_asset = random.choice(pad_assets_list)
        inst = place_pad_random(
            canvas,
            pad_asset,
            padding=padding,
            min_asset_distance=min_distance,
            max_tries=100
        )

        if inst is not None:
            placed += 1

    print(f"Placed {placed}/{n_pads} pads")

    visualize_canvas(canvas, title=f"Placed {placed} pads")

    return canvas
