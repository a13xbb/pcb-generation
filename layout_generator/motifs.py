from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import random


class MotifType(Enum):
    PAD_ROW        = "pad_row"
    DUAL_ROW       = "dual_row"
    PAD_PAIR       = "pad_pair"
    TRACE_BUS      = "trace_bus"
    TRACE_CHAIN    = "trace_chain"     # single grow-forward multi-segment path
    PARALLEL_CHAINS = "parallel_chains"  # 2-3 side-by-side H-biased chains


@dataclass
class AbstractPad:
    pad_id: int
    pos: Tuple[float, float]
    angle: int = 0


@dataclass
class AbstractTrace:
    trace_id: int
    pad_id_a: int
    pad_id_b: int
    required_length: float
    required_angle: int  # 0=horizontal, 90=vertical


@dataclass
class AbstractMotif:
    motif_type: MotifType
    pads: List[AbstractPad]
    traces: List[AbstractTrace]
    bounding_box: Tuple[float, float, float, float]  # x1,y1,x2,y2


@dataclass
class MotifCell:
    cell_x: int
    cell_y: int
    origin: Tuple[float, float]
    size: Tuple[float, float]


def make_pad_row(origin_y: float, n_pads: int, pad_spacing: float, col_x: float) -> AbstractMotif:
    """Vertical column of n_pads isolated pads (IC pin row / connector)."""
    pads = [AbstractPad(pad_id=i, pos=(col_x, origin_y + i * pad_spacing)) for i in range(n_pads)]
    x1 = col_x - 20
    y1 = origin_y - 20
    x2 = col_x + 20
    y2 = origin_y + (n_pads - 1) * pad_spacing + 20
    return AbstractMotif(MotifType.PAD_ROW, pads, [], (x1, y1, x2, y2))


def make_dual_row(left_x: float, top_y: float, n_pads: int, pad_spacing: float, col_gap: float) -> AbstractMotif:
    """Two facing vertical pad columns with horizontal connecting traces (DIP IC pattern)."""
    pads = []
    traces = []
    for i in range(n_pads):
        y = top_y + i * pad_spacing
        pads.append(AbstractPad(pad_id=i,           pos=(left_x,            y)))
        pads.append(AbstractPad(pad_id=n_pads + i,  pos=(left_x + col_gap,  y)))
        traces.append(AbstractTrace(
            trace_id=i,
            pad_id_a=i, pad_id_b=n_pads + i,
            required_length=col_gap,
            required_angle=0,  # horizontal
        ))
    x1 = left_x - 20
    y1 = top_y - 20
    x2 = left_x + col_gap + 20
    y2 = top_y + (n_pads - 1) * pad_spacing + 20
    return AbstractMotif(MotifType.DUAL_ROW, pads, traces, (x1, y1, x2, y2))


def make_pad_pair(pad_a_pos: Tuple[float, float], direction_angle: int, trace_length: float) -> AbstractMotif:
    """Two pads connected by a single trace (SMD resistor/capacitor pattern)."""
    ax, ay = pad_a_pos
    if direction_angle == 0:
        bx, by = ax + trace_length, ay
    elif direction_angle == 90:
        bx, by = ax, ay + trace_length
    elif direction_angle == 180:
        bx, by = ax - trace_length, ay
    else:  # 270
        bx, by = ax, ay - trace_length

    pads = [AbstractPad(pad_id=0, pos=(ax, ay)), AbstractPad(pad_id=1, pos=(bx, by))]
    traces = [AbstractTrace(
        trace_id=0, pad_id_a=0, pad_id_b=1,
        required_length=trace_length,
        required_angle=direction_angle % 180,  # normalise: 0=H, 90=V
    )]
    x1 = min(ax, bx) - 20
    y1 = min(ay, by) - 20
    x2 = max(ax, bx) + 20
    y2 = max(ay, by) + 20
    return AbstractMotif(MotifType.PAD_PAIR, pads, traces, (x1, y1, x2, y2))


def make_trace_bus(origin: Tuple[float, float], k_traces: int, trace_length: float,
                   bus_spacing: float, orientation: int = 0) -> AbstractMotif:
    """K parallel traces with a pad at each end (routing bus pattern)."""
    ox, oy = origin
    pads = []
    traces = []
    for i in range(k_traces):
        if orientation == 0:  # horizontal bus
            ax, ay = ox, oy + i * bus_spacing
            bx, by = ox + trace_length, oy + i * bus_spacing
        else:  # vertical bus
            ax, ay = ox + i * bus_spacing, oy
            bx, by = ox + i * bus_spacing, oy + trace_length
        pad_id_a = 2 * i
        pad_id_b = 2 * i + 1
        pads.append(AbstractPad(pad_id=pad_id_a, pos=(ax, ay)))
        pads.append(AbstractPad(pad_id=pad_id_b, pos=(bx, by)))
        traces.append(AbstractTrace(
            trace_id=i, pad_id_a=pad_id_a, pad_id_b=pad_id_b,
            required_length=trace_length, required_angle=orientation,
        ))
    if orientation == 0:
        bb = (ox - 20, oy - 20, ox + trace_length + 20, oy + (k_traces - 1) * bus_spacing + 20)
    else:
        bb = (ox - 20, oy - 20, ox + (k_traces - 1) * bus_spacing + 20, oy + trace_length + 20)
    return AbstractMotif(MotifType.TRACE_BUS, pads, traces, bb)


def plan_grid(canvas_h: int, canvas_w: int, n_cols: int = 3, n_rows: int = 2,
              edge_margin: int = 30) -> List[MotifCell]:
    """Divide canvas into n_cols × n_rows cells."""
    usable_w = canvas_w - 2 * edge_margin
    usable_h = canvas_h - 2 * edge_margin
    cell_w = usable_w / n_cols
    cell_h = usable_h / n_rows
    cells = []
    for row in range(n_rows):
        for col in range(n_cols):
            ox = edge_margin + col * cell_w
            oy = edge_margin + row * cell_h
            cells.append(MotifCell(cell_x=col, cell_y=row, origin=(ox, oy), size=(cell_w, cell_h)))
    return cells


def _pick_length(lengths: List[float], preferred_range: Tuple[float, float]) -> Optional[float]:
    """Pick a random length from `lengths` that falls within `preferred_range`."""
    lo, hi = preferred_range
    candidates = [l for l in lengths if lo <= l <= hi]
    return random.choice(candidates) if candidates else None


_PAD_SPACING = 55   # center-to-center distance between pads in a column


def generate_abstract_motif_for_cell(cell: MotifCell,
                                     available_h_lengths: List[float],
                                     available_v_lengths: List[float]) -> Optional[AbstractMotif]:
    """
    Generate one AbstractMotif for the given grid cell, choosing a motif type
    based on which trace lengths are available.
    """
    ox, oy = cell.origin
    cw, ch = cell.size

    # Geometric centre of the cell
    cx = ox + cw / 2.0
    cy = oy + ch / 2.0

    # Build weighted list of feasible motif types (all require traces — no PAD_ROW).
    if not available_h_lengths:
        return None
    motif_types: List[MotifType] = [MotifType.DUAL_ROW, MotifType.PAD_PAIR, MotifType.TRACE_BUS]
    weights: List[float] = [0.40, 0.35, 0.25]

    r = random.random()
    cumulative = 0.0
    chosen = motif_types[0]
    for mt, w in zip(motif_types, weights):
        cumulative += w
        if r <= cumulative:
            chosen = mt
            break

    # ---- DUAL_ROW ----
    if chosen == MotifType.DUAL_ROW:
        col_gap = _pick_length(available_h_lengths, (45, 80))
        if col_gap is None:
            col_gap = _pick_length(available_h_lengths, (30, 130))
        if col_gap is None:
            return None
        n_pads = random.choice([3, 4])
        total_h = (n_pads - 1) * _PAD_SPACING
        left_x = cx - col_gap / 2.0
        return make_dual_row(left_x, cy - total_h / 2, n_pads, _PAD_SPACING, col_gap)

    # ---- PAD_PAIR ----
    if chosen == MotifType.PAD_PAIR:
        use_h = available_h_lengths and (not available_v_lengths or random.random() < 0.7)
        if use_h:
            tlen = _pick_length(available_h_lengths, (40, 130))
            if tlen is None:
                tlen = available_h_lengths[len(available_h_lengths) // 2]
            return make_pad_pair((cx - tlen / 2.0, cy), 0, tlen)
        elif available_v_lengths:
            tlen = _pick_length(available_v_lengths, (40, 130))
            if tlen is None:
                tlen = available_v_lengths[len(available_v_lengths) // 2]
            return make_pad_pair((cx, cy - tlen / 2.0), 90, tlen)
        else:
            return None

    # ---- TRACE_BUS ----
    if chosen == MotifType.TRACE_BUS:
        k = random.choice([3, 4])
        tlen = _pick_length(available_h_lengths, (100, 150))
        if tlen is None:
            tlen = _pick_length(available_h_lengths, (60, 200))
        if tlen is None:
            return None
        bus_spacing = 35
        total_h = (k - 1) * bus_spacing
        return make_trace_bus(
            origin=(cx - tlen / 2.0, cy - total_h / 2.0),
            k_traces=k, trace_length=tlen, bus_spacing=bus_spacing, orientation=0,
        )

    return None
