import os
import cv2
import shutil
import numpy as np

from classes import Pad, TraceSegment
from utils import * 

def main():
    # filter_pads(
    #     folder="PADS",
    #     trash_folder="trash",
    #     margin=10
    # )

    # split_trace_instances(
    #     folder="COPPER_TRACES",
    #     trash_folder="trash",
    #     min_area=200
    # )

    mask_path = "COPPER_TRACES/mask_0057.png"   # <-- путь к маске дорожки
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    skeleton = skeletonize(mask)

    # Visual skeleton check
    # overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # overlay[skeleton > 0] = (0, 0, 255)  # skeleton красным

    # cv2.imwrite("debug_mask.png", mask)
    # cv2.imwrite("debug_skeleton.png", skeleton)
    # cv2.imwrite("debug_overlay.png", overlay)
    
    #Visual endpoints check
    vis = visualize_endpoints(mask, skeleton)
    cv2.imwrite("debug_endpoints.png", vis)
            
if __name__ == "__main__":
    main()
