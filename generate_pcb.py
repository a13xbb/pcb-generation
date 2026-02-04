import os
import cv2
import shutil
import numpy as np

from classes import *
from placement_engine import *
from utils import * 

def main():
    # np,random.seed(20)
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
    
    
    orig_img = cv2.imread("images/example.jpg")
    
    pads_folder = "PADS"
    pads_masks = [cv2.imread(os.path.join(pads_folder, name), cv2.IMREAD_GRAYSCALE) for name in os.listdir(pads_folder)
                  if name.startswith("mask") and name.endswith(".png")]
    pad_assets = []
    for pad_mask in pads_masks:
        bbox, centroid = compute_bbox_and_centroid(pad_mask)
        asset = PadAsset(pad_mask, orig_img, centroid, bbox)
        pad_assets.append(asset)
        
    traces_folder = "COPPER_TRACES"
    trace_masks = [cv2.imread(os.path.join(traces_folder, name), cv2.IMREAD_GRAYSCALE) for name in os.listdir(traces_folder)
                  if name.startswith("mask") and name.endswith(".png")]
    trace_assets = []
    for trace_mask in trace_masks:
        skel = skeletonize(trace_mask)
        endpoints = find_endpoints(trace_mask, skel)
        # cv2.imwrite(f"debug_endpoints{i}.png", visualize_endpoints(trace_mask, skel))
        # print(endpoints)
        if len(endpoints) == 1:
            continue
        assert len(endpoints) == 2, "Invalid amount of endpoints!"
        skel_len = skeleton_path_length_between_endpoints(skel, endpoints[0], endpoints[1])
        if skel_len is None:
            skel_len = 0
        bbox, centroid = compute_bbox_and_centroid(trace_mask)
        trace_asset = TraceAsset(trace_mask, orig_img, skel, endpoints, centroid, skel_len)
        trace_assets.append(trace_asset)
        
        
    # test_pad_placement(pad_assets, padding=20)
    canvas = CanvasState(600, 600)
    
    place_pad_random(canvas, np.random.choice(pad_assets), padding=30)
    
    place_trace_attached_to_pad(canvas, np.random.choice(trace_assets))
    
    visualize_canvas_real(canvas, "canvas.png")
            
if __name__ == "__main__":
    main()
