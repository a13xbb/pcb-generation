# Synthetic PCB Image Generation for Defect Detection

A pipeline for generating large labeled synthetic datasets of PCB (Printed Circuit Board) images with realistic defects, intended as training data for automated visual defect detection systems.

This work was developed as part of a bachelor's thesis on applying text-to-image generation to synthetic training data creation. The pipeline combines SAM3-based asset extraction, procedural layout composition, SDXL ControlNet/LoRA diffusion, and programmatic defect injection to produce photorealistic PCB images with pixel-accurate YOLO annotations.

---

## Pipeline Overview

```
Real PCB photo
    └─ extract_assets.py  (SAM3 segmentation)
           └─ assets/PADS/ + assets/COPPER_TRACES/   ← one-time setup

assets/ ──► pipeline_stage1_layouts.py  ──► layouts/ + structure_maps/
                                                    │
                                     pipeline_stage2_diffusion.py  ──► images/
                                                    │
                                      pipeline_stage3_defects.py  ──► defects/ + YOLO labels
```

**Stage 1** procedurally composes synthetic PCB layouts from extracted component masks.  
**Stage 2** renders those layouts into photorealistic PCB images using SDXL + fine-tuned ControlNet + LoRA.  
**Stage 3** injects synthetic defects and outputs YOLO-format bounding box annotations.

---

## Requirements & Setup

Python 3.11 is required.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `torch==2.9.1`, `diffusers==0.37.0`, `ultralytics==8.4.8`, `opencv-contrib-python==4.13.0.90`, `transformers==5.3.0`, `accelerate==1.13.0`, `peft`.

---

## Asset Extraction (one-time)

Before running the pipeline, extract pad and trace masks from a real PCB photo using SAM3. The results are saved to `assets/` and reused across all pipeline runs.

```bash
python extract_assets.py --image images/example.jpg
```

| Flag | Default | Description |
|---|---|---|
| `--image` | `images/example.jpg` | Input PCB photo |
| `--pads_dir` | `assets/PADS` | Output directory for pad masks |
| `--traces_dir` | `assets/COPPER_TRACES` | Output directory for trace masks |
| `--model` | `models/sam3.pt` | SAM3 checkpoint path |
| `--conf` | `0.25` | SAM confidence threshold |
| `--pad_margin` | `10` | Border margin (px) for pad filtering |
| `--min_trace_area` | `200` | Minimum trace area (px) after splitting |

The script segments contact pads (prompt: `"small silver element"`), builds a binary copper mask, segments copper traces from it, then filters and deduplicates both asset sets. The repo ships with 53 pad masks and 44 trace masks already extracted.

---

## Running the Pipeline

All three stages read from and write to a shared `--output_dir`. Running them in sequence on the same directory produces the full dataset.

### Stage 1 — Layout Generation

```bash
python pipeline_stage1_layouts.py \
    --n_layouts 100 \
    --output_dir images/run1
```

| Flag | Default | Description |
|---|---|---|
| `--n_layouts` | `10` | Number of layouts to generate |
| `--output_dir` | `images/pipeline` | Output directory |
| `--height` | `600` | Canvas height (px) |
| `--width` | `600` | Canvas width (px) |
| `--start_idx` | `0` | Starting index for output filenames |

Outputs: `<output_dir>/layouts/layout_N.png` and `<output_dir>/structure_maps/layout_N.png`.

### Stage 2 — Diffusion (SDXL + ControlNet + LoRA)

```bash
python pipeline_stage2_diffusion.py \
    --input_dir images/run1 \
    --output_dir images/run1 \
    --controlnet_path trained/controlnet_aug_600x600 \
    --lora_path trained/lora_aug_600x600
```

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | `images/pipeline` | Directory with structure maps from Stage 1 |
| `--output_dir` | `images/pipeline` | Output directory |
| `--controlnet_path` | `trained/controlnet_aug_600x600` | ControlNet checkpoint |
| `--lora_path` | `trained/lora_aug_600x600` | LoRA checkpoint |
| `--steps` | `50` | Diffusion inference steps |
| `--guidance_scale` | `6.0` | CFG scale |
| `--controlnet_scale` | `0.85` | ControlNet conditioning scale |
| `--lora_scale` | `0.8` | LoRA weight scale |
| `--refine_strength` | `0.3` | img2img refinement strength |
| `--seed` | `42` | Random seed |
| `--verbose` | `False` | Save intermediate images |
| `--skip_borders` | `False` | Skip border painting step |

Outputs: `<output_dir>/images/layout_N.png`.

### Stage 3 — Defects Generation

```bash
python pipeline_stage3_defects.py \
    --input_dir images/run1 \
    --output_dir images/run1 \
    --repeats_per_image 3
```

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | `images/pipeline` | Directory with images from Stage 2 |
| `--output_dir` | `images/pipeline` | Output directory |
| `--repeats_per_image` | `1` | Defective variants to generate per image |
| `--min_defects` | `4` | Minimum defects per image |
| `--max_defects` | `6` | Maximum defects per image |
| `--seed` | `42` | Random seed |

Outputs: `<output_dir>/defects/layout_N_k.png` and `<output_dir>/labels/layout_N_k.txt` (YOLO format).

### Defect Classes

| Class | Description |
|---|---|
| `missing_hole` | Pad with drill hole removed |
| `mouse_bite` | Irregular notch on board edge |
| `open_circuit` | Break in a copper trace |
| `spur` | Unwanted copper protrusion from a trace |
| `spurious_copper` | Isolated copper deposit on the board |

---

## Training

### SDXL ControlNet

Fine-tunes `diffusers/controlnet-canny-sdxl-1.0` on paired (structure map, PCB image) data.

```bash
python structure_generation/train_controlnet.py \
    --images_dir pcb-defect-dataset/train/images \
    --structures_dir pcb-defect-dataset/train/structure_maps \
    --output_dir trained/my_controlnet \
    --resolution 600 \
    --epochs 5 \
    --lr 1e-5
```

### SDXL LoRA

Fine-tunes SDXL UNet on a folder of PCB images (rank-16 LoRA on attention projections).

```bash
python structure_generation/train_lora.py \
    --dataset_dir <folder_of_pcb_images> \
    --output_dir trained/my_lora \
    --resolution 600 \
    --epochs 5 \
    --lr 1e-5 \
    --rank 16
```

### YOLOv8 Defect Detector

Three training configurations are available. All use `yolov8m` by default and apply on-the-fly grayscale + binarization + random crop augmentations.

**Combined (real + synthetic):**
```bash
python yolo_training/train_combined_v1.py \
    --data datasets/combined/data.yaml \
    --epochs 100
```

**Real data only:**
```bash
python yolo_training/train_real_grayscale_v3.py \
    --data datasets/pcb-defect-dataset/data_synth_aligned.yaml \
    --epochs 100
```

**Synthetic data only:**
```bash
python yolo_training/train_grayscale_synth_v2.py \
    --data datasets/generated_dataset/v2/defects/data.yaml \
    --epochs 150
```

Common flags across all training scripts:

| Flag | Default | Description |
|---|---|---|
| `--model` | `models/yolov8m.pt` | Base YOLO checkpoint |
| `--epochs` | `100` | Training epochs |
| `--imgsz` | `640` | Input resolution |
| `--binarize_p` | `0.3` | Probability of binarization augmentation |
| `--crop_p` | `0.3` | Probability of random zoom-in crop |

**Evaluation & error visualization:**

`yolo_training/visualize_errors.py` loads a trained model and test set into [FiftyOne](https://voxel51.com/fiftyone/), computes TP/FP/FN per class, and opens an interactive browser UI for filtering errors by class, confidence, or evaluation outcome.

```bash
python yolo_training/visualize_errors.py \
    --model yolo_training/results/synth_train/weights/best.pt \
    --name synth_analysis

# custom thresholds
python yolo_training/visualize_errors.py \
    --model yolo_training/results/real_train/weights/best.pt \
    --name real_analysis \
    --conf 0.25 --iou 0.5
```

Example FiftyOne UI filters:
```
F("eval") == "fn"                                    # missed detections
F("ground_truth.detections.label") == "mouse_bite"  # by GT class
F("predictions.detections.confidence") > 0.5        # by confidence
```

---

## Project Structure

```
├── pipeline_stage1_layouts.py    # Stage 1: layout generation
├── pipeline_stage2_diffusion.py  # Stage 2: ControlNet+LoRA inference
├── pipeline_stage3_defects.py    # Stage 3: defect injection + annotation
├── extract_assets.py             # One-time SAM3 asset extraction
│
├── layout_generator/             # Procedural layout engine
│   ├── classes.py                # PadAsset, TraceAsset, CanvasState
│   ├── generator.py              # Layout generation algorithms
│   ├── placement_engine.py       # Collision-aware placement
│   ├── motifs.py                 # Grid motif planner
│   └── utils.py                  # Skeletonization, Dijkstra path length
│
├── structure_generation/         # Diffusion training & preprocessing
│   ├── image_to_structure.py     # Layout → Canny edge map
│   ├── train_controlnet.py       # ControlNet fine-tuning
│   ├── train_lora.py             # LoRA fine-tuning
│   └── controlnet_dataset.py     # Dataset loader for ControlNet training
│
├── defects_generation/           # Defect synthesis
│   ├── orchestrator.py           # Defect placement & annotation logic
│   ├── missing_hole.py
│   ├── mouse_bite.py
│   ├── open_circuit.py
│   ├── spur.py
│   └── spurious_copper.py
│
└── yolo_training/                # Defect detector training & evaluation
    ├── augmentations.py          # Shared grayscale/binarization transforms
    ├── train_combined_v1.py
    ├── train_real_grayscale_v2.py
    ├── train_synth_grayscale_v2.py
    ├── train_real_v1.py
    ├── train_synth_v1.py
    └── visualize_errors.py       # FiftyOne error analysis (TP/FP/FN)
```
