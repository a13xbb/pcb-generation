import os
import cv2
import shutil
import numpy as np
import time

from classes import *
from placement_engine import *
from utils import * 
from generator import generate_layout_by_coverage, generate_layout_by_path_plan, canvas_coverage
from pad_classifier import classify_pad_shape

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
    
    # orig_img = cv2.imread("images/example.jpg")
    # assert orig_img is not None, "Failed to read images/example.jpg"

    # # ---------- Load PAD assets ----------
    # pads_folder = "PADS"
    # pad_assets: list[PadAsset] = []

    # pad_mask_files = [
    #     name for name in os.listdir(pads_folder)
    #     if name.startswith("mask") and name.endswith(".png")
    # ]

    # for name in pad_mask_files:
    #     pad_mask = cv2.imread(os.path.join(pads_folder, name), cv2.IMREAD_GRAYSCALE)
    #     if pad_mask is None:
    #         continue

    #     pad_mask = (pad_mask > 0).astype(np.uint8)  # 0/1

    #     bbox, centroid = compute_bbox_and_centroid(pad_mask)
    #     pad_assets.append(PadAsset(pad_mask, orig_img, centroid, bbox))

    # assert len(pad_assets) > 0, "No pad assets loaded"

    # # ---------- Load TRACE assets ----------
    # traces_folder = "COPPER_TRACES"
    # trace_assets: list[TraceAsset] = []

    # trace_mask_files = sorted([
    #     name for name in os.listdir(traces_folder)
    #     if name.startswith("mask") and name.endswith(".png")
    # ])

    # for name in trace_mask_files:
    #     trace_mask = cv2.imread(os.path.join(traces_folder, name), cv2.IMREAD_GRAYSCALE)
    #     if trace_mask is None:
    #         continue

    #     trace_mask = (trace_mask > 0).astype(np.uint8)  # 0/1

    #     skel = skeletonize(trace_mask)                  # ожидаем 0/1 или 0/255 — не критично, но лучше 0/1
    #     skel = (skel > 0).astype(np.uint8)

    #     endpoints = find_endpoints(trace_mask, skel)
    #     if len(endpoints) != 2:
    #         continue

    #     length = skeleton_path_length_between_endpoints(skel, endpoints[0], endpoints[1])
    #     if length is None:
    #         length = 0.0

    #     bbox, centroid = compute_bbox_and_centroid(trace_mask)

    #     trace_assets.append(TraceAsset(trace_mask, orig_img, skel, endpoints, centroid, length))

    # assert len(trace_assets) > 0, "No trace assets loaded"

    # # ---------- Create canvas ----------
    # canvas = CanvasState(600, 600)  # occupied_mask должен быть 0/1 uint8

    # # ---------- Place ONE PAD ----------
    # pad_asset = random.choice(pad_assets)
    # pad_inst = place_pad_random(canvas, pad_asset, padding=30)

    # assert pad_inst is not None, "Failed to place a pad"

    # pad_inst.id = 0
    # pad_inst.attached_traces = set()

    # # ---------- Attach ONE TRACE to this PAD ----------
    # open_ends: list[OpenEnd] = []
    # next_trace_id = 0

    # trace_asset = random.choice(trace_assets)
    
    # cur_pad_inst = pick_random_valid_pad(canvas, max_degree=3)

    # trace_inst, attached_ep_idx = place_trace_attached_to_specific_pad(
    #     canvas_state=canvas,
    #     trace_asset=trace_asset,
    #     pad=cur_pad_inst,
    #     open_ends=open_ends,
    #     next_trace_id=next_trace_id,
    #     max_attempts=30,
    #     require_touch=True
    # )

    # if trace_inst is None:
    #     print("Failed to attach a trace to the pad (try increasing max_attempts or relaxing constraints)")
    # else:
    #     print("Attached trace:", trace_inst.id, "attached_end:", attached_ep_idx)
    #     print("Open ends:", open_ends)

    # # ---------- Render ----------
    # visualize_canvas_real(canvas, "canvas.png")
    
    
    
    
    
    
    
    
    orig_img = cv2.imread("images/example.jpg")
    assert orig_img is not None, "Failed to read images/example.jpg"

    # ---------- Load PAD assets ----------
    pads_folder = "PADS"
    pad_assets: list[PadAsset] = []

    pad_mask_files = [
        name for name in os.listdir(pads_folder)
        if name.startswith("mask") and name.endswith(".png")
    ]

    for name in pad_mask_files:
        pad_mask = cv2.imread(os.path.join(pads_folder, name), cv2.IMREAD_GRAYSCALE)
        if pad_mask is None:
            continue

        pad_mask = (pad_mask > 0).astype(np.uint8)  # 0/1

        bbox, centroid = compute_bbox_and_centroid(pad_mask)
        pad_assets.append(PadAsset(pad_mask, orig_img, centroid, bbox))

    assert len(pad_assets) > 0, "No pad assets loaded"

    # ---------- Load TRACE assets ----------
    traces_folder = "COPPER_TRACES"
    trace_assets: list[TraceAsset] = []

    trace_mask_files = sorted([
        name for name in os.listdir(traces_folder)
        if name.startswith("mask") and name.endswith(".png")
    ])

    for name in trace_mask_files:
        trace_mask = cv2.imread(os.path.join(traces_folder, name), cv2.IMREAD_GRAYSCALE)
        if trace_mask is None:
            continue

        trace_mask = (trace_mask > 0).astype(np.uint8)  # 0/1

        skel = skeletonize(trace_mask)                  # ожидаем 0/1 или 0/255 — не критично, но лучше 0/1
        skel = (skel > 0).astype(np.uint8)

        endpoints = find_endpoints(trace_mask, skel)
        if len(endpoints) != 2:
            continue

        length = skeleton_path_length_between_endpoints(skel, endpoints[0], endpoints[1])
        if length is None:
            length = 0.0

        bbox, centroid = compute_bbox_and_centroid(trace_mask)

        trace_assets.append(TraceAsset(trace_mask, orig_img, skel, endpoints, centroid, length))

    assert len(trace_assets) > 0, "No trace assets loaded"
    
    canvas = CanvasState(1000, 1000, pad_keepout_radius=15)
    
    # ok = generate_layout_by_coverage(
    #     canvas=canvas,
    #     pad_assets=pad_assets,
    #     trace_assets=trace_assets,
    #     target_coverage=0.14,   # например 10%
    #     padding=30
    # )
    
    path_plan = {
        3: 2,
        2: 4,
        1: 6
    }

    start = time.time()
    ok = generate_layout_by_path_plan(
        canvas=canvas,
        pad_assets=pad_assets,
        trace_assets=trace_assets,
        path_length_counts=path_plan,
        isolated_pads=10,
        padding=10,
        max_path_attempts=50,
        trace_attach_attempts=20,
        close_attempts_per_end=50,
    )
    end = time.time()

    print(f"Time elapsed: {end - start}")
    visualize_canvas_real(canvas, "canvas.png")
    print("ok:", ok, "coverage:", canvas_coverage(canvas))
            
if __name__ == "__main__":
    main()
