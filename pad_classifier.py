import cv2
import numpy as np

def classify_pad_shape(mask: np.ndarray):
    if mask is None:
        return "unknown", {}

    bin_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown", {}

    cnt = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter <= 1e-6 or area <= 1e-6:
        return "unknown", {}

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = (w / h) if h > 0 else 0.0

    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    extent = area / float(w * h) if (w * h) > 0 else 0.0

    # --- (опционально) сгладить шум контура через approxPoly ---
    eps = 0.02 * perimeter
    approx = cv2.approxPolyDP(cnt, eps, True)
    n_vertices = len(approx)

    # =========================
    # КЛАССИФИКАЦИЯ
    # =========================

    # 1) SQUARE/RECT-like: почти квадрат по bbox + высокая "заполненность"
    #    (скруглённые углы всё равно дадут высокий extent)
    if 0.85 <= aspect_ratio <= 1.15 and extent >= 0.84:
        shape = "square"

    # 2) OVAL: сильно вытянуто
    elif aspect_ratio < 0.75 or aspect_ratio > 1.33:
        shape = "oval"

    else:
        # 3) CIRCLE: почти квадрат по bbox, но extent близок к кругу (~0.785)
        #    и достаточно круглый по circularity.
        #    Верхнюю границу extent ставим, чтобы отрезать "скруглённые квадраты".
        if (0.85 <= aspect_ratio <= 1.15
            and circularity >= 0.82
            and extent <= 0.83
            and n_vertices >= 6):   # у круга/овала вершин обычно больше
            shape = "circle"
        else:
            shape = "oval"

    metrics = {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "extent": extent,
        "n_vertices": n_vertices,
        "bbox_w": w,
        "bbox_h": h,
    }
    return shape, metrics