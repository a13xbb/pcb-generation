import numpy as np
import random
import cv2

from classes import *
#--------------------------------------------------------------------------------
# HELPER FUNCTIONS
#--------------------------------------------------------------------------------

_ERODE_KERNEL_3X3 = np.ones((3, 3), np.uint8)

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
    overlap = cv2.bitwise_and(canvas_mask, new_mask)
    return cv2.countNonZero(overlap) > 0


def _dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(max(0, radius))
    if radius <= 0:
        return (mask > 0).astype(np.uint8)

    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)


def _update_pad_keepout(canvas_state: CanvasState, pad_mask_world: np.ndarray):
    radius = int(getattr(canvas_state, "pad_keepout_radius", 0))
    if radius <= 0:
        return

    expanded = _dilate_binary_mask(pad_mask_world, radius)
    canvas_state.pad_keepout_mask |= expanded
    

def _find_pad_instance_by_id(canvas_state: CanvasState, pad_id: int | None):
    if pad_id is None:
        return None
    by_id = getattr(canvas_state, "pad_by_id", None)
    if by_id is not None:
        pad = by_id.get(pad_id)
        if pad is not None:
            return pad
    for pad in canvas_state.pad_instances:
        if pad.id == pad_id:
            return pad
    return None


def _can_place_without_pad_keepout_conflict(
    canvas_state: CanvasState,
    candidate_mask: np.ndarray,
    allowed_overlap_mask: np.ndarray | None = None
) -> bool:
    radius = int(getattr(canvas_state, "pad_keepout_radius", 0))
    if radius <= 0:
        return True

    keepout_mask = getattr(canvas_state, "pad_keepout_mask", None)
    if keepout_mask is None:
        return True

    conflict = cv2.bitwise_and(candidate_mask, keepout_mask)
    if allowed_overlap_mask is not None:
        conflict = cv2.bitwise_and(conflict, cv2.bitwise_not(allowed_overlap_mask))

    return cv2.countNonZero(conflict) == 0


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()  # x1,y1,x2,y2


def _ensure_pad_attach_cache(pad: PadInstance):
    if getattr(pad, "_attach_cache_ready", False):
        return

    mask_u8 = getattr(pad, "_mask_world_u8", None)
    if mask_u8 is None:
        mask_u8 = (pad.mask_world > 0).astype(np.uint8)
        pad._mask_world_u8 = mask_u8

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        pad._attach_center = None
        pad._attach_boundary = None
        pad._attach_cache_ready = True
        return

    if getattr(pad, "bbox_world", None) is None:
        pad.bbox_world = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    cx = float(xs.mean())
    cy = float(ys.mean())

    eroded = cv2.erode(mask_u8, _ERODE_KERNEL_3X3, iterations=1)
    boundary = cv2.bitwise_and(mask_u8, cv2.bitwise_not(eroded))
    by, bx = np.where(boundary > 0)

    pad._attach_center = (cx, cy)
    pad._attach_boundary = (bx, by)
    pad._attach_cache_ready = True


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

#--------------------------------------------------------------------------------
#PADS PLACEMENT
#--------------------------------------------------------------------------------

def place_pad_random(
    canvas: CanvasState,
    pad_asset: PadAsset,
    padding: int = 10,
    min_asset_distance: int = 5,
    max_tries: int = 50):
    
    for _ in range(max_tries):
        # angle = random.choice([0, 90, 180, 270])
        angle = 0

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

        mask_world_u8 = (mask_world > 0).astype(np.uint8)
        if check_collision(canvas.occupied_mask, mask_world_u8):
            continue

        if not _can_place_without_pad_keepout_conflict(canvas, mask_world_u8):
            continue

        bbox = _bbox_from_mask(mask_world_u8)
        if bbox is None:
            continue

        if min_asset_distance > 0:
            too_close = False

            for inst in canvas.pad_instances:
                other_bbox = inst.bbox_world
                if other_bbox is None:
                    other_bbox = _bbox_from_mask((inst.mask_world > 0).astype(np.uint8))
                    inst.bbox_world = other_bbox
                if other_bbox is None:
                    continue

                dist = _bbox_distance(bbox, other_bbox)
                if dist < min_asset_distance:
                    too_close = True
                    break

            if too_close:
                continue

        # --- 7. Размещение ---
        canvas.occupied_mask |= mask_world_u8
        _update_pad_keepout(canvas, mask_world_u8)

        inst = PadInstance(
            pad_asset,
            Transform(angle, tx, ty),
            mask_world_u8,
            bbox_world=bbox
        )
        inst._mask_world_u8 = mask_world_u8

        canvas.pad_instances.append(inst)

        return inst

    return None


def place_pad_attached_to_open_end(
    canvas_state: CanvasState,
    pad_asset: PadAsset,
    open_end: OpenEnd,
    open_ends: list[OpenEnd],
    traces_by_id: dict[int, TraceInstance],
    next_pad_id: int,
    angles=(0, 90, 180, 270),
    max_attempts: int = 20,
    require_touch: bool = True
):
    """
    Пытается разместить pad так, чтобы он приклеился к open_end (концу существующего трейса).
    Возвращает (pad_instance, new_next_pad_id) или (None, next_pad_id).
    """

    # 0) Найти trace instance
    if open_end.trace_id not in traces_by_id:
        return None, next_pad_id
    tr = traces_by_id[open_end.trace_id]
    tr_mask_world_u8 = (tr.mask_world > 0).astype(np.uint8)

    # Если этот конец уже закрыт — ничего не делаем
    if tr.pad_end[open_end.end_idx] is not None:
        # убрать open_end как устаревший
        try:
            open_ends.remove(open_end)
        except ValueError:
            pass
        return None, next_pad_id

    # Точка приклейки (world)
    attach_x, attach_y = open_end.pos_xy
    attach_x = float(attach_x)
    attach_y = float(attach_y)

    canvas_shape = (canvas_state.h, canvas_state.w)

    # если у пада уже 3 подключения — смысла ставить его нет (обычно у нового пада 0)
    # но оставим защиту
    # NOTE: мы создаём новый pad, у него будет 1 подключение после приклейки.

    angles_list = list(angles)

    for _ in range(max_attempts):
        angle = random.choice(angles_list)

        # affine: centroid pad -> attach point
        # здесь можно использовать твой transform_mask напрямую
        pad_mask_world, M = transform_mask(
            pad_asset.mask,
            angle,
            attach_x,
            attach_y,
            canvas_shape,
            centroid=pad_asset.centroid
        )
        if M is None or pad_mask_world is None:
            continue

        pad_mask_world_u8 = (pad_mask_world > 0).astype(np.uint8)  # 0/1

        if cv2.countNonZero(pad_mask_world_u8) == 0:
            continue

        # 1) Требуем контакт с трейсом (иначе можно “приклеить” рядом)
        if require_touch:
            if cv2.countNonZero(cv2.bitwise_and(pad_mask_world_u8, tr_mask_world_u8)) == 0:
                continue
        
        source_pad_mask = tr_mask_world_u8
        source_pad_id = tr.pad_end[1 - open_end.end_idx]
        source_pad = _find_pad_instance_by_id(canvas_state, source_pad_id)
        if source_pad is not None:
            source_pad_mask = cv2.bitwise_or(source_pad_mask, (source_pad.mask_world > 0).astype(np.uint8))
            
        if not _can_place_without_pad_keepout_conflict(
            canvas_state,
            pad_mask_world_u8,
            allowed_overlap_mask=source_pad_mask
        ):
            continue

        # 2) Коллизии: запрещаем пересечение со всем, кроме этого trace
        # Разрешаем overlap с trace (tr.mask_world), но не с остальным occupied
        overlap = cv2.bitwise_and(pad_mask_world_u8, canvas_state.occupied_mask)
        illegal_overlap = cv2.bitwise_and(overlap, cv2.bitwise_not(tr_mask_world_u8))
        if cv2.countNonZero(illegal_overlap) > 0:
            continue

        # 3) Явно запретим пересечение с другими трассами (кроме tr) — надёжно
        bad = False
        for other in canvas_state.trace_instances:
            if other.id == tr.id:
                continue
            if cv2.countNonZero(cv2.bitwise_and(pad_mask_world_u8, other.mask_world)) > 0:
                bad = True
                break
        if bad:
            continue

        # (опционально) 4) Ограничение степени нового пада после приклейки:
        # здесь всегда будет 1, но если ты потом захочешь “приклеивать к существующему паду” — пригодится.

        # SUCCESS -> создаём pad instance
        pad_bbox = _bbox_from_mask(pad_mask_world_u8)
        if pad_bbox is None:
            continue

        pad_inst = PadInstance(
            asset=pad_asset,
            transform=Transform(angle, attach_x, attach_y),
            mask_world=pad_mask_world_u8,
            id=next_pad_id,
            bbox_world=pad_bbox
        )
        pad_inst._mask_world_u8 = pad_mask_world_u8
        pad_inst.attached_traces = set([tr.id])

        # обновляем trace
        tr.pad_end[open_end.end_idx] = pad_inst.id

        # коммит в canvas
        canvas_state.pad_instances.append(pad_inst)
        canvas_state.register_pad(pad_inst)
        canvas_state.occupied_mask |= pad_mask_world_u8
        _update_pad_keepout(canvas_state, pad_mask_world_u8)

        # закрываем open end
        try:
            open_ends.remove(open_end)
        except ValueError:
            pass

        return pad_inst, next_pad_id + 1

    return None, next_pad_id


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

    visualize_canvas_real(canvas)

    return canvas

#--------------------------------------------------------------------------------
#TRACES PLACEMENT
#--------------------------------------------------------------------------------

def count_pad_connections(pad_instance, trace_instances, radius=5):

    cx, cy = map(int, pad_instance.asset.centroid)

    cnt = 0

    for tr in trace_instances:
        for ex, ey in tr.endpoints_world:
            if abs(ex - cx) <= radius and abs(ey - cy) <= radius:
                cnt += 1
                break

    return cnt

def _pick_pad_attach_point(pad: PadInstance, boundary_mix: float = 0.65):
    _ensure_pad_attach_cache(pad)
    if pad._attach_center is None:
        return None

    cx, cy = pad._attach_center

    boundary_mix = float(np.clip(boundary_mix, 0.0, 1.0))
    if boundary_mix <= 0.0:
        return cx, cy

    bx, by = pad._attach_boundary
    if len(bx) == 0:
        return cx, cy

    idx = random.randrange(len(bx))
    bxv, byv = float(bx[idx]), float(by[idx])

    tx = cx + boundary_mix * (bxv - cx)
    ty = cy + boundary_mix * (byv - cy)

    return tx, ty

def build_affine_align_point(mask_shape, centroid_xy, angle_deg, point_xy, target_xy):
    """
    Возвращает матрицу M (2x3), которая:
    - вращает вокруг centroid_xy
    - затем сдвигает так, чтобы point_xy перешёл в target_xy
    """
    cx, cy = centroid_xy
    px, py = point_xy
    tx, ty = target_xy

    # чистый поворот вокруг центроида (как в transform_mask)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # куда попал point после поворота (без доп. сдвига)
    p = np.array([px, py, 1.0], dtype=np.float32)
    prx, pry = (p @ M.T)

    # добавляем сдвиг так, чтобы pr -> target
    M[0, 2] += tx - prx
    M[1, 2] += ty - pry

    return M

def warp_mask_with_M(mask, M, canvas_shape):
    warped = cv2.warpAffine(
        mask,
        M,
        (canvas_shape[1], canvas_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )
    return warped

def pick_random_valid_pad(canvas_state: CanvasState, max_degree: int = 3):
    """
    Возвращает случайный pad_instance, у которого степень < max_degree.
    """
    candidates = [
        p for p in canvas_state.pad_instances
        if count_pad_connections(p, canvas_state.trace_instances) < max_degree
    ]
    if not candidates:
        return None
    return random.choice(candidates)

def place_trace_attached_to_specific_pad(
    canvas_state: CanvasState,
    trace_asset: TraceAsset,
    pad: PadInstance,
    open_ends: list[OpenEnd],
    next_trace_id: int,
    angles=(0, 90, 180, 270),
    max_attempts: int = 20,
    require_touch: bool = True,
    pad_boundary_mix: float = 0.65
):
    """
    Пытается приклеить trace_asset к конкретному pad_instance.
    Возвращает (trace_instance, attached_endpoint_index) или (None, None).
    """
    
    if pad.id is None:
        raise ValueError("pad.id must be set before attaching traces")
    
    canvas_shape = (canvas_state.h, canvas_state.w)
    _ensure_pad_attach_cache(pad)

    # Базовая точка приклейки на паде: между центроидом и границей.
    pad_mask_world_u8 = pad._mask_world_u8
    if pad_mask_world_u8 is None or cv2.countNonZero(pad_mask_world_u8) == 0:
        return None, None

    # Чтобы попытки были разнообразнее:
    endpoints = list(trace_asset.endpoints)
    angles_list = list(angles)

    for _ in range(max_attempts):
        ep_idx = random.randrange(len(endpoints))
        ep_local = endpoints[ep_idx]
        angle = random.choice(angles_list)

        attach_pt = _pick_pad_attach_point(pad, boundary_mix=pad_boundary_mix)
        if attach_pt is None:
            continue

        # affine: выбранный endpoint -> точка приклейки на pad
        M = build_affine_align_point(
            mask_shape=trace_asset.mask.shape,
            centroid_xy=trace_asset.centroid,
            angle_deg=angle,
            point_xy=ep_local,
            target_xy=attach_pt
        )

        mask_world = warp_mask_with_M(trace_asset.mask, M, canvas_shape)

        # быстрый отсев: пустая маска (вылетела за канвас)
        if mask_world is None:
            continue
        mask_world_u8 = (mask_world > 0).astype(np.uint8)
        if cv2.countNonZero(mask_world_u8) == 0:
            continue

        # endpoints world
        endpoints_world = transform_points(trace_asset.endpoints, M)

        # 1) Требуем реальный контакт с pad (защита от “почти приклеили”)
        if require_touch:
            if cv2.countNonZero(cv2.bitwise_and(mask_world_u8, pad_mask_world_u8)) == 0:
                continue

        # 2) Разрешаем overlap только с target pad
        overlap = cv2.bitwise_and(mask_world_u8, canvas_state.occupied_mask)
        illegal_overlap = cv2.bitwise_and(overlap, cv2.bitwise_not(pad_mask_world_u8))
        if cv2.countNonZero(illegal_overlap) > 0:
            continue

        # 3) Трейсы не должны пересекаться друг с другом (явная проверка)
        intersects_trace = False
        for tr in canvas_state.trace_instances:
            if cv2.countNonZero(cv2.bitwise_and(mask_world_u8, tr.mask_world)) > 0:
                intersects_trace = True
                break
        if intersects_trace:
            continue

        # ok -> создаём instance
        c = np.array([trace_asset.centroid[0], trace_asset.centroid[1], 1.0], dtype=np.float32)
        tx_world, ty_world = (c @ M.T)

        inst = TraceInstance(
            asset=trace_asset,
            transform=Transform(angle, float(tx_world), float(ty_world)),
            mask_world=mask_world_u8,
            endpoints_world=endpoints_world
        )
        
        # assign id
        inst.id = next_trace_id

        # connection: attached endpoint -> this pad
        inst.pad_end[ep_idx] = pad.id
        pad.attached_traces.add(inst.id)
        
        # create open end for the other endpoint
        free_idx = 1 - ep_idx
        free_pos = inst.endpoints_world[free_idx]
        open_ends.append(OpenEnd(trace_id=inst.id, end_idx=free_idx, pos_xy=free_pos))

        canvas_state.trace_instances.append(inst)
        canvas_state.occupied_mask |= mask_world_u8

        return inst, ep_idx

    return None, None

def place_trace_attached_to_pad(
    canvas_state: CanvasState,
    trace_asset: TraceAsset,
    max_degree: int = 3,
    angles=(0, 90, 180, 270),
    max_attempts_pad_pick: int = 20,
    max_attempts_attach: int = 20,
):
    """
    Старая семантика: сам выбирает валидный pad и пытается к нему приклеить trace.
    """
    for _ in range(max_attempts_pad_pick):
        pad = pick_random_valid_pad(canvas_state, max_degree=max_degree)
        if pad is None:
            return None, None
        inst, ep_idx = place_trace_attached_to_specific_pad(
            canvas_state,
            trace_asset,
            pad,
            angles=angles,
            max_attempts=max_attempts_attach
        )
        if inst is not None:
            return inst, ep_idx
    return None, None

# def place_trace_attached_to_pad(
#     canvas_state: CanvasState,
#     trace_asset: TraceAsset,
#     angles=(0, 90, 180, 270),
#     max_attempts=50
# ):

#     if len(canvas_state.pad_instances) == 0:
#         return None

#     # ---- выбираем допустимые пады ----
#     candidate_pads = [
#         p for p in canvas_state.pad_instances
#         if count_pad_connections(p, canvas_state.trace_instances) < 3
#     ]

#     if not candidate_pads:
#         return None

#     canvas_shape = (canvas_state.h, canvas_state.w)

#     # ---- пробуем placement ----
#     for _ in range(max_attempts):

#         pad = random.choice(candidate_pads)

#         # ---- pad world centroid ----
#         ys, xs = np.where(pad.mask_world > 0)
#         pad_world_cx = float(xs.mean())
#         pad_world_cy = float(ys.mean())

#         ep_local = random.choice(trace_asset.endpoints)
#         angle = random.choice(angles)

#         # Строим M так, чтобы ep_local приклеился к центру пада
#         M = build_affine_align_point(
#             mask_shape=trace_asset.mask.shape,
#             centroid_xy=trace_asset.centroid,
#             angle_deg=angle,
#             point_xy=ep_local,
#             target_xy=(pad_world_cx, pad_world_cy)
#         )

#         mask_world = warp_mask_with_M(trace_asset.mask, M, canvas_shape)
#         endpoints_world = transform_points(trace_asset.endpoints, M)

#         # ---- проверка пересечений ----
#         # разрешаем overlap только с target pad
#         pad_mask = pad.mask_world

#         overlap = mask_world & canvas_state.occupied_mask

#         illegal_overlap = overlap & (~pad_mask)

#         if np.any(illegal_overlap):
#             continue

#         # ---- проверка пересечения с trace отдельно (опционально, но чище)
#         intersects_trace = False
#         for tr in canvas_state.trace_instances:
#             if np.any(mask_world & tr.mask_world):
#                 intersects_trace = True
#                 break

#         if intersects_trace:
#             continue

#         # ---- создаём instance ----
#         c = np.array([trace_asset.centroid[0], trace_asset.centroid[1], 1.0], dtype=np.float32)
#         tx_world, ty_world = (c @ M.T)

#         instance = TraceInstance(
#             asset=trace_asset,
#             transform=Transform(angle, tx_world, ty_world),
#             mask_world=mask_world,
#             endpoints_world=endpoints_world
#         )

#         canvas_state.trace_instances.append(instance)
#         canvas_state.occupied_mask |= mask_world

#         return instance

#     return None

#--------------------------------------------------------------------------------
#VISUALIZATION OF CANVAS
#--------------------------------------------------------------------------------

def transform_image(image, angle, tx, ty, canvas_shape, centroid=None):

    h, w = image.shape[:2]

    if centroid is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 0)
        if len(xs) == 0:
            return np.zeros((*canvas_shape, 3), dtype=image.dtype), None
        cx, cy = xs.mean(), ys.mean()
    else:
        cx, cy = centroid

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    M[0, 2] += tx - cx
    M[1, 2] += ty - cy

    warped = cv2.warpAffine(
        image,
        M,
        (canvas_shape[1], canvas_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=(0, 0, 0)
    )

    return warped, M


def visualize_canvas_real(canvas: CanvasState, out_path="canvas_render.png"):

    canvas_img = np.zeros((canvas.h, canvas.w, 3), np.uint8)

    canvas_shape = (canvas.h, canvas.w)
    
     # ---- TRACES ----
    for inst in canvas.trace_instances:

        asset = inst.asset
        T = inst.transform

        img_world, _ = transform_image(
            asset.image,
            T.angle,
            T.tx,
            T.ty,
            canvas_shape,
            centroid=asset.centroid   # или centroid trace если добавишь
        )

        mask_world = inst.mask_world > 0

        canvas_img[mask_world] = img_world[mask_world]

    # ---- PADs ----
    for inst in canvas.pad_instances:

        asset = inst.asset
        T = inst.transform

        img_world, _ = transform_image(
            asset.image,
            T.angle,
            T.tx,
            T.ty,
            canvas_shape,
            centroid=asset.centroid
        )

        mask_world = inst.mask_world > 0

        canvas_img[mask_world] = img_world[mask_world]

    cv2.imwrite(out_path, canvas_img)

    
