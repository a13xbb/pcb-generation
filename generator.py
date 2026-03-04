import cv2
import random
import numpy as np

from utils import *
from classes import *
from placement_engine import *

def is_on_border(pt, h, w, margin=1):
    x, y = pt
    return (x <= margin or y <= margin or x >= w - 1 - margin or y >= h - 1 - margin)

def canvas_coverage(canvas: CanvasState) -> float:
    return float(np.count_nonzero(canvas.occupied_mask)) / float(canvas.h * canvas.w)

def pick_random_valid_pad(canvas: CanvasState, max_degree=3):
    cands = [p for p in canvas.pad_instances if len(p.attached_traces) < max_degree]
    return random.choice(cands) if cands else None


def generate_layout_by_coverage(
    canvas: CanvasState,
    pad_assets: list,
    trace_assets: list,
    target_coverage: float = 0.08,        # 8% занятости — стартовое значение, подбирается
    padding: int = 30,                    # для initial pad random
    max_pad_degree: int = 3,
    prob_close_end_with_pad: float = 0.75,   # как часто закрывать open_end падом
    prob_spawn_isolated_pad: float = 0.10,   # иногда добавить пад в пустоту
    max_steps: int = 500,                 # safety stop
    final_close_attempts_per_end: int = 100,
):
    # state
    open_ends: list[OpenEnd] = []
    traces_by_id: dict[int, TraceInstance] = {}

    next_pad_id = 0
    next_trace_id = 0

    # ---------- seed: one pad ----------
    p0 = place_pad_random(canvas, random.choice(pad_assets), padding=padding)
    if p0 is None:
        return False  # failed to seed

    p0.id = next_pad_id
    next_pad_id += 1
    p0.attached_traces = set()
    canvas.register_pad(p0)

    # ---------- seed: one trace from p0 ----------
    t0, attached_ep = place_trace_attached_to_specific_pad(
        canvas_state=canvas,
        trace_asset=random.choice(trace_assets),
        pad=p0,
        open_ends=open_ends,
        next_trace_id=next_trace_id,
        require_touch=True
    )
    if t0 is not None:
        traces_by_id[t0.id] = t0
        next_trace_id += 1

    # ---------- growth loop ----------
    steps = 0
    while steps < max_steps and canvas_coverage(canvas) < target_coverage:
        steps += 1

        # 1) иногда добавляем изолированный пад
        if random.random() < prob_spawn_isolated_pad:
            p = place_pad_random(canvas, random.choice(pad_assets), padding=padding)
            if p is not None:
                p.id = next_pad_id
                next_pad_id += 1
                p.attached_traces = set()
                canvas.register_pad(p)
            continue

        # 2) если есть открытые концы — чаще закрываем их падом
        if open_ends and random.random() < prob_close_end_with_pad:
            oe = random.choice(open_ends)
            # если конец на границе — можно оставить, но для густоты часто приятнее закрыть (на твой вкус)
            if is_on_border(oe.pos_xy, canvas.h, canvas.w, margin=1):
                # оставляем как "выход"
                # можно убрать из open_ends, чтобы не мешал дальше:
                open_ends.remove(oe)
                continue

            p_asset = random.choice(pad_assets)
            p_new, next_pad_id2 = place_pad_attached_to_open_end(
                canvas_state=canvas,
                pad_asset=p_asset,
                open_end=oe,
                open_ends=open_ends,
                traces_by_id=traces_by_id,
                next_pad_id=next_pad_id,
                angles=(0, 90, 180, 270),
                max_attempts=20,
                require_touch=True
            )
            if p_new is not None:
                next_pad_id = next_pad_id2
            continue

        # 3) иначе — добавляем трейс от случайного валидного пада (ветвление)
        pad = pick_random_valid_pad(canvas, max_degree=max_pad_degree)
        if pad is None:
            # нет падов, которым можно добавлять трейсы
            break

        tr_asset = random.choice(trace_assets)
        t, attached_ep = place_trace_attached_to_specific_pad(
            canvas_state=canvas,
            trace_asset=tr_asset,
            pad=pad,
            open_ends=open_ends,
            next_trace_id=next_trace_id,
            require_touch=True
        )
        if t is not None:
            traces_by_id[t.id] = t
            next_trace_id += 1

    # ---------- finalization: close remaining open ends (optional best-effort) ----------
    # Пытаемся закрыть все non-border open ends падом.
    # Те, что на border — считаем валидным выходом.
    for oe in list(open_ends):

        # если border не разрешён
        if is_on_border(oe.pos_xy, canvas.h, canvas.w, margin=1):
            return False

        success = False
        for _ in range(final_close_attempts_per_end):
            p_asset = random.choice(pad_assets)
            p_new, next_pad_id2 = place_pad_attached_to_open_end(
                canvas_state=canvas,
                pad_asset=p_asset,
                open_end=oe,
                open_ends=open_ends,
                traces_by_id=traces_by_id,
                next_pad_id=next_pad_id,
                angles=(0, 90, 180, 270),
                max_attempts=20,
                require_touch=True
            )
            if p_new is not None:
                next_pad_id = next_pad_id2
                success = True
                break

        if not success:
            return False

    # финальная проверка
    if len(open_ends) != 0:
        return False

    return True

def _rebuild_occupied_mask(canvas: CanvasState):
    canvas.occupied_mask[:] = 0
    if hasattr(canvas, "pad_keepout_mask"):
        canvas.pad_keepout_mask[:] = 0
    
    for tr in canvas.trace_instances:
        canvas.occupied_mask |= (tr.mask_world > 0).astype(np.uint8)
        
    pad_keepout_radius = int(getattr(canvas, "pad_keepout_radius", 0))
    keepout_kernel = None
    if pad_keepout_radius > 0:
        k = 2 * pad_keepout_radius + 1
        keepout_kernel = np.ones((k, k), dtype=np.uint8)
    
    for pad in canvas.pad_instances:
        pad_binary = (pad.mask_world > 0).astype(np.uint8)
        canvas.occupied_mask |= pad_binary
        if keepout_kernel is not None:
            canvas.pad_keepout_mask |= cv2.dilate(pad_binary, keepout_kernel, iterations=1)
    if hasattr(canvas, "rebuild_pad_index"):
        canvas.rebuild_pad_index()


def _snapshot_canvas(canvas: CanvasState):
    return {
        "occupied_mask": canvas.occupied_mask.copy(),
        "pad_keepout_mask": canvas.pad_keepout_mask.copy(),
        "pad_instances": list(canvas.pad_instances),
        "trace_instances": list(canvas.trace_instances),
        "pad_by_id": dict(getattr(canvas, "pad_by_id", {})),
    }


def _restore_canvas(canvas: CanvasState, snapshot):
    canvas.occupied_mask = snapshot["occupied_mask"]
    canvas.pad_keepout_mask = snapshot["pad_keepout_mask"]
    canvas.pad_instances = snapshot["pad_instances"]
    canvas.trace_instances = snapshot["trace_instances"]
    canvas.pad_by_id = snapshot.get("pad_by_id", {})


def _try_place_single_path(
    canvas: CanvasState,
    pad_assets: list,
    trace_assets: list,
    path_length: int,
    next_pad_id: int,
    next_trace_id: int,
    padding: int,
    trace_attach_attempts: int,
    close_attempts_per_end: int,
    debug_stats: dict | None = None,
):
    if path_length <= 0:
        return True, next_pad_id, next_trace_id

    traces_by_id: dict[int, TraceInstance] = {}

    start_pad = place_pad_random(canvas, random.choice(pad_assets), padding=padding)
    if start_pad is None:
        if debug_stats is not None:
            debug_stats["fail_start_pad"] = debug_stats.get("fail_start_pad", 0) + 1
        return False, next_pad_id, next_trace_id

    start_pad.id = next_pad_id
    start_pad.attached_traces = set()
    next_pad_id += 1
    canvas.register_pad(start_pad)

    local_pads: dict[int, PadInstance] = {start_pad.id: start_pad}
    end_pad_ids: set[int] = {start_pad.id}

    # Строим путь сегмент за сегментом. При неудаче откатываем только текущий сегмент,
    # а не весь уже построенный префикс пути.
    for _ in range(path_length):
        segment_placed = False

        for _ in range(trace_attach_attempts):
            if not end_pad_ids:
                break

            base_pad_id = random.choice(list(end_pad_ids))
            base_pad = local_pads.get(base_pad_id)
            if base_pad is None:
                continue

            segment_snapshot = _snapshot_canvas(canvas)
            local_pads_snapshot = dict(local_pads)
            end_pad_ids_snapshot = set(end_pad_ids)
            traces_snapshot = dict(traces_by_id)
            next_pad_id_snapshot = next_pad_id
            next_trace_id_snapshot = next_trace_id

            open_ends: list[OpenEnd] = []

            tr_asset = random.choice(trace_assets)
            t, _ = place_trace_attached_to_specific_pad(
                canvas_state=canvas,
                trace_asset=tr_asset,
                pad=base_pad,
                open_ends=open_ends,
                next_trace_id=next_trace_id,
                require_touch=True,
                pad_boundary_mix=0.65,
            )
            if t is None:
                _restore_canvas(canvas, segment_snapshot)
                local_pads = local_pads_snapshot
                end_pad_ids = end_pad_ids_snapshot
                traces_by_id = traces_snapshot
                next_pad_id = next_pad_id_snapshot
                next_trace_id = next_trace_id_snapshot
                continue

            traces_by_id[t.id] = t
            next_trace_id += 1

            open_end = next((oe for oe in open_ends if oe.trace_id == t.id), None)
            if open_end is None:
                if debug_stats is not None:
                    debug_stats["fail_open_end_missing"] = debug_stats.get("fail_open_end_missing", 0) + 1
                _restore_canvas(canvas, segment_snapshot)
                local_pads = local_pads_snapshot
                end_pad_ids = end_pad_ids_snapshot
                traces_by_id = traces_snapshot
                next_pad_id = next_pad_id_snapshot
                next_trace_id = next_trace_id_snapshot
                continue

            new_pad = None
            for _ in range(close_attempts_per_end):
                p_asset = random.choice(pad_assets)
                p_inst, next_pad_candidate = place_pad_attached_to_open_end(
                    canvas_state=canvas,
                    pad_asset=p_asset,
                    open_end=open_end,
                    open_ends=open_ends,
                    traces_by_id=traces_by_id,
                    next_pad_id=next_pad_id,
                    angles=(0, 90, 180, 270),
                    max_attempts=20,
                    require_touch=True,
                )
                if p_inst is not None:
                    new_pad = p_inst
                    next_pad_id = next_pad_candidate
                    break

            if new_pad is None:
                _restore_canvas(canvas, segment_snapshot)
                local_pads = local_pads_snapshot
                end_pad_ids = end_pad_ids_snapshot
                traces_by_id = traces_snapshot
                next_pad_id = next_pad_id_snapshot
                next_trace_id = next_trace_id_snapshot
                continue

            local_pads[new_pad.id] = new_pad

            if base_pad_id in end_pad_ids:
                end_pad_ids.remove(base_pad_id)
            end_pad_ids.add(new_pad.id)

            segment_placed = True
            break

        if not segment_placed:
            if debug_stats is not None:
                debug_stats["fail_trace_attach"] = debug_stats.get("fail_trace_attach", 0) + 1
            return False, next_pad_id, next_trace_id

    return True, next_pad_id, next_trace_id


def generate_layout_by_path_plan(
    canvas: CanvasState,
    pad_assets: list,
    trace_assets: list,
    path_length_counts: dict[int, int],
    isolated_pads: int = 0,
    padding: int = 30,
    max_path_attempts: int = 80,
    trace_attach_attempts: int = 30,
    close_attempts_per_end: int = 80,
    strict_plan: bool = False,
    debug_log: bool = False,
):
    next_pad_id = 0
    next_trace_id = 0
    total_placed_paths = 0
    global_debug = {
        "path_attempts": 0,
        "path_success": 0,
        "path_failed": 0,
        "fail_start_pad": 0,
        "fail_trace_attach": 0,
        "fail_open_end_missing": 0,
        "fail_close_end_with_pad": 0,
        "isolated_pads_requested": max(0, int(isolated_pads)),
        "isolated_pads_placed": 0,
    }

    for path_len in sorted(path_length_counts.keys(), reverse=True):
        cnt = int(path_length_counts[path_len])
        if path_len <= 0 or cnt <= 0:
            continue

        placed_for_this_len = 0
        for _ in range(cnt):
            placed = False
            for _ in range(max_path_attempts):
                global_debug["path_attempts"] += 1
                snapshot = _snapshot_canvas(canvas)

                ok, new_next_pad_id, new_next_trace_id = _try_place_single_path(
                    canvas=canvas,
                    pad_assets=pad_assets,
                    trace_assets=trace_assets,
                    path_length=path_len,
                    next_pad_id=next_pad_id,
                    next_trace_id=next_trace_id,
                    padding=padding,
                    trace_attach_attempts=trace_attach_attempts,
                    close_attempts_per_end=close_attempts_per_end,
                    debug_stats=global_debug,
                )

                if ok:
                    next_pad_id = new_next_pad_id
                    next_trace_id = new_next_trace_id
                    placed = True
                    placed_for_this_len += 1
                    total_placed_paths += 1
                    global_debug["path_success"] += 1
                    break

                _restore_canvas(canvas, snapshot)

            if not placed:
                global_debug["path_failed"] += 1

            if not placed and strict_plan:
                _rebuild_occupied_mask(canvas)
                if debug_log:
                    print("[path-plan][debug] strict failure")
                    print("[path-plan][debug]", global_debug)
                return False

        if strict_plan and placed_for_this_len < cnt:
            _rebuild_occupied_mask(canvas)
            return False

    for _ in range(max(0, int(isolated_pads))):
        p = place_pad_random(canvas, random.choice(pad_assets), padding=padding)
        if p is None:
            continue
        p.id = next_pad_id
        p.attached_traces = set()
        next_pad_id += 1
        canvas.register_pad(p)
        global_debug["isolated_pads_placed"] += 1

    _rebuild_occupied_mask(canvas)
    if debug_log:
        print("[path-plan][debug]", global_debug)
    return total_placed_paths > 0 or len(canvas.pad_instances) > 0
