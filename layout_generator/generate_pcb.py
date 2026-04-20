import os
import cv2
import shutil
import numpy as np
import time
from tqdm import tqdm

from classes import *
from placement_engine import *
from utils import * 
from generator import (generate_layout_by_coverage, generate_layout_by_path_plan,
                        generate_layout_motif_based, canvas_coverage)
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
    
    
    
    
    
    
    
    
    _dir = os.path.dirname(os.path.abspath(__file__))
    assets_root = os.path.join(_dir, '..', 'assets')
    images_root = os.path.join(_dir, '..', 'images')

    def _load_object_img(folder: str, mask_name: str, fallback_shape):
        """Load the matching object_XXXX.png crop, or synthesise a grey placeholder."""
        obj_name = mask_name.replace("mask", "object", 1)
        obj_path = os.path.join(folder, obj_name)
        img = cv2.imread(obj_path)
        if img is not None:
            return img
        h, w = fallback_shape
        return np.full((h, w, 3), 160, dtype=np.uint8)

    # ---------- Load PAD assets ----------
    pads_folder = os.path.join(assets_root, 'PADS')
    pad_assets: list[PadAsset] = []

    pad_mask_files = [
        name for name in os.listdir(pads_folder)
        if name.startswith("mask") and name.endswith(".png")
    ]

    for name in pad_mask_files:
        pad_mask = cv2.imread(os.path.join(pads_folder, name), cv2.IMREAD_GRAYSCALE)
        if pad_mask is None:
            continue

        pad_mask = (pad_mask > 0).astype(np.uint8)
        obj_img = _load_object_img(pads_folder, name, pad_mask.shape[:2])

        bbox, centroid = compute_bbox_and_centroid(pad_mask)
        pad_assets.append(PadAsset(pad_mask, obj_img, centroid, bbox))

    assert len(pad_assets) > 0, "No pad assets loaded"

    # ---------- Load TRACE assets ----------
    traces_folder = os.path.join(assets_root, 'COPPER_TRACES')
    trace_assets: list[TraceAsset] = []

    trace_mask_files = sorted([
        name for name in os.listdir(traces_folder)
        if name.startswith("mask") and name.endswith(".png")
    ])

    for name in trace_mask_files:
        trace_mask = cv2.imread(os.path.join(traces_folder, name), cv2.IMREAD_GRAYSCALE)
        if trace_mask is None:
            continue

        trace_mask = (trace_mask > 0).astype(np.uint8)

        skel = skeletonize(trace_mask)
        skel = (skel > 0).astype(np.uint8)

        endpoints = find_endpoints(trace_mask, skel)
        if len(endpoints) != 2:
            continue

        length = skeleton_path_length_between_endpoints(skel, endpoints[0], endpoints[1])
        if length is None:
            length = 0.0

        bbox, centroid = compute_bbox_and_centroid(trace_mask)
        obj_img = _load_object_img(traces_folder, name, trace_mask.shape[:2])
        trace_assets.append(TraceAsset(trace_mask, obj_img, skel, endpoints, centroid, length))

    assert len(trace_assets) > 0, "No trace assets loaded"
    print(f"Loaded {len(pad_assets)} pad assets, {len(trace_assets)} trace assets")

    out_dir = os.path.join(images_root, 'layouts')
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(20)):
        canvas = CanvasState(600, 600, pad_keepout_radius=8)

        start = time.time()
        ok = generate_layout_motif_based(
            canvas=canvas,
            pad_assets=pad_assets,
            trace_assets=trace_assets,
            n_cols=4,
            n_rows=3,
            edge_padding=10,
            max_motif_attempts=5,
            isolated_pads=8,
            debug_log=True,
        )
        end = time.time()

        print(f"Time elapsed: {end - start:.1f}s  ok={ok}  coverage={canvas_coverage(canvas):.3f}")
        visualize_canvas_real(canvas, os.path.join(out_dir, f'layout_{i}.png'))
            
if __name__ == "__main__":
    main()
