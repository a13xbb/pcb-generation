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
