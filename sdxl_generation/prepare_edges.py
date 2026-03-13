import cv2
import os
from pathlib import Path

layout_dir = Path("layouts")
edge_dir = Path("edges")
edge_dir.mkdir(exist_ok=True)

for p in layout_dir.glob("*.png"):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    # небольшой blur, чтобы линии стали плавнее
    img = cv2.GaussianBlur(img, (3,3), 0)

    edges = cv2.Canny(img, 50, 150)

    # чуть утолщаем линии — ControlNet лучше реагирует
    edges = cv2.dilate(edges, None, iterations=1)

    cv2.imwrite(str(edge_dir / p.name), edges)