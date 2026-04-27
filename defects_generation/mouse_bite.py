from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .colors import sample_trace_color, darken_color, add_noise_to_color
from .types import DefectAnnotation, DefectResult, CLASS_IDS
from .utils import filter_long_traces, find_trace_edge_points, get_trace_width_at_point

if TYPE_CHECKING:
    from layout_generator.classes import CanvasState


def generate_mouse_bite(
    image: np.ndarray,
    canvas: "CanvasState",
    rng: np.random.Generator,
    bite_depth_ratio: float = 0.3,
    min_bite_width: int = 5,
    max_bite_width: int = 15,
) -> DefectResult:
    long_traces = filter_long_traces(canvas.trace_instances)
    if not long_traces:
        return DefectResult(success=False)

    trace = rng.choice(long_traces)

    edge_points = find_trace_edge_points(trace.mask_world)
    if not edge_points:
        return DefectResult(success=False)

    ex, ey, normal_angle = edge_points[rng.integers(len(edge_points))]

    trace_width = get_trace_width_at_point(trace.mask_world, (ex, ey), normal_angle)
    if trace_width < 5:
        trace_width = 10

    bite_depth = int(trace_width * bite_depth_ratio)
    bite_depth = max(3, min(bite_depth, int(trace_width * 0.4)))

    h, w = image.shape[:2]
    mask_h, mask_w = canvas.occupied_mask.shape
    scale_x = w / mask_w
    scale_y = h / mask_h

    ex_img = int(ex * scale_x)
    ey_img = int(ey * scale_y)
    bite_depth_img = int(bite_depth * scale_y)

    trace_color = sample_trace_color(image, trace.mask_world)
    dark_color = darken_color(trace_color, factor=0.5)
    dark_color = add_noise_to_color(dark_color, sigma=8.0, rng=rng)

    bite_width = rng.integers(min_bite_width, max_bite_width + 1)

    inward_angle = normal_angle + np.pi
    center_x = int(ex_img + bite_depth_img * 0.5 * np.cos(inward_angle))
    center_y = int(ey_img + bite_depth_img * 0.5 * np.sin(inward_angle))

    axes = (bite_width // 2, bite_depth_img // 2)
    angle_deg = int(np.degrees(normal_angle))

    cv2.ellipse(image, (center_x, center_y), axes, angle_deg, 0, 360, dark_color, -1)

    x1 = center_x - max(axes)
    y1 = center_y - max(axes)
    x2 = center_x + max(axes)
    y2 = center_y + max(axes)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    annotation = DefectAnnotation.from_pixel_bbox(
        CLASS_IDS["mouse_bite"], x1, y1, x2, y2, w, h
    )

    return DefectResult(success=True, annotation=annotation)
