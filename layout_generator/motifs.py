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
    PAD_GRID       = "pad_grid"        # N×M regular pad array (IC, via field)
    DENSE_BUS      = "dense_bus"       # 8-16 tightly spaced parallel traces


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


def make_pad_grid(origin: Tuple[float, float], n_rows: int, n_cols: int,
                  row_spacing: float, col_spacing: float) -> AbstractMotif:
    """N×M regular pad grid (IC pin array, via field, connector)."""
    ox, oy = origin
    pads = []
    for r in range(n_rows):
        for c in range(n_cols):
            pad_id = r * n_cols + c
            pads.append(AbstractPad(pad_id=pad_id, pos=(ox + c * col_spacing, oy + r * row_spacing)))
    x1 = ox - 20
    y1 = oy - 20
    x2 = ox + (n_cols - 1) * col_spacing + 20
    y2 = oy + (n_rows - 1) * row_spacing + 20
    return AbstractMotif(MotifType.PAD_GRID, pads, [], (x1, y1, x2, y2))


def make_dense_bus(origin: Tuple[float, float], n_traces: int, trace_length: float,
                   spacing: float = 20, orientation: int = 0) -> AbstractMotif:
    """Dense parallel trace bus (8-16 traces at tight spacing)."""
    ox, oy = origin
    pads = []
    traces = []
    for i in range(n_traces):
        if orientation == 0:  # horizontal bus
            ax, ay = ox, oy + i * spacing
            bx, by = ox + trace_length, oy + i * spacing
        else:  # vertical bus
            ax, ay = ox + i * spacing, oy
            bx, by = ox + i * spacing, oy + trace_length
        pad_id_a = 2 * i
        pad_id_b = 2 * i + 1
        pads.append(AbstractPad(pad_id=pad_id_a, pos=(ax, ay)))
        pads.append(AbstractPad(pad_id=pad_id_b, pos=(bx, by)))
        traces.append(AbstractTrace(
            trace_id=i, pad_id_a=pad_id_a, pad_id_b=pad_id_b,
            required_length=trace_length, required_angle=orientation,
        ))
    if orientation == 0:
        bb = (ox - 20, oy - 20, ox + trace_length + 20, oy + (n_traces - 1) * spacing + 20)
    else:
        bb = (ox - 20, oy - 20, ox + (n_traces - 1) * spacing + 20, oy + trace_length + 20)
    return AbstractMotif(MotifType.DENSE_BUS, pads, traces, bb)


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

    # Build weighted list of feasible motif types
    motif_types: List[MotifType] = [
        MotifType.DUAL_ROW, MotifType.PAD_PAIR, MotifType.TRACE_BUS,
        MotifType.PAD_GRID, MotifType.DENSE_BUS
    ]
    weights: List[float] = [0.20, 0.15, 0.15, 0.25, 0.25]

    # PAD_GRID doesn't need traces; others do
    if not available_h_lengths:
        motif_types = [MotifType.PAD_GRID]
        weights = [1.0]

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
        k = random.choice([4, 5, 6])
        orientation = random.choice([0, 0, 90]) if available_v_lengths else 0
        lengths = available_h_lengths if orientation == 0 else available_v_lengths
        tlen = _pick_length(lengths, (100, 150))
        if tlen is None:
            tlen = _pick_length(lengths, (60, 200))
        if tlen is None:
            return None
        bus_spacing = random.choice([25, 30, 35])
        total_span = (k - 1) * bus_spacing
        if orientation == 0:
            return make_trace_bus(
                origin=(cx - tlen / 2.0, cy - total_span / 2.0),
                k_traces=k, trace_length=tlen, bus_spacing=bus_spacing, orientation=0,
            )
        else:
            return make_trace_bus(
                origin=(cx - total_span / 2.0, cy - tlen / 2.0),
                k_traces=k, trace_length=tlen, bus_spacing=bus_spacing, orientation=90,
            )

    # ---- PAD_GRID ----
    if chosen == MotifType.PAD_GRID:
        n_rows = random.choice([2, 3, 4])
        n_cols = random.choice([3, 4, 5, 6])
        row_spacing = random.choice([40, 45, 50, 55])
        col_spacing = random.choice([40, 45, 50, 55])
        total_w = (n_cols - 1) * col_spacing
        total_h = (n_rows - 1) * row_spacing
        return make_pad_grid(
            origin=(cx - total_w / 2.0, cy - total_h / 2.0),
            n_rows=n_rows, n_cols=n_cols,
            row_spacing=row_spacing, col_spacing=col_spacing,
        )

    # ---- DENSE_BUS ----
    if chosen == MotifType.DENSE_BUS:
        n_traces = random.choice([8, 10, 12, 14])
        tlen = _pick_length(available_h_lengths, (80, 150))
        if tlen is None:
            tlen = _pick_length(available_h_lengths, (50, 200))
        if tlen is None:
            return None
        spacing = random.choice([18, 20, 22, 25])
        orientation = random.choice([0, 90]) if available_v_lengths else 0
        total_span = (n_traces - 1) * spacing
        if orientation == 0:
            return make_dense_bus(
                origin=(cx - tlen / 2.0, cy - total_span / 2.0),
                n_traces=n_traces, trace_length=tlen, spacing=spacing, orientation=0,
            )
        else:
            return make_dense_bus(
                origin=(cx - total_span / 2.0, cy - tlen / 2.0),
                n_traces=n_traces, trace_length=tlen, spacing=spacing, orientation=90,
            )

    return None
