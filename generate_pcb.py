import os
import cv2
import shutil
import numpy as np

from classes import *
from placement_engine import *
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

    # mask_path = "COPPER_TRACES/mask_0057.png"   # <-- путь к маске дорожки
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # skeleton = skeletonize(mask)

    # Visual skeleton check
    # overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # overlay[skeleton > 0] = (0, 0, 255)  # skeleton красным

    # cv2.imwrite("debug_mask.png", mask)
    # cv2.imwrite("debug_skeleton.png", skeleton)
    # cv2.imwrite("debug_overlay.png", overlay)
    
    #Visual endpoints check
    # vis = visualize_endpoints(mask, skeleton)
    # cv2.imwrite("debug_endpoints.png", vis)
    
    pads_folder = "PADS"
    pads_masks = [cv2.imread(os.path.join(pads_folder, name), cv2.IMREAD_GRAYSCALE) for name in os.listdir(pads_folder)
                  if name.startswith("mask") and name.endswith(".png")]
    pad_assets = []
    for pad_mask in pads_masks:
        bbox, centroid = compute_bbox_and_centroid(pad_mask)
        asset = PadAsset(pad_mask, centroid, bbox)
        pad_assets.append(asset)
        
    test_pad_placement(pad_assets, padding=20)
            
if __name__ == "__main__":
    main()
