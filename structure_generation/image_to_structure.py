import os
os.environ['HF_HOME'] = '/mnt/ssdm2/users/alexblokh/cache'

from __future__ import annotations

import argparse
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MORPH_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


@dataclass(frozen=True)
class StructureConfig:
    pre_blur_kernel: int = 5
    canny_low: int = 50
    canny_high: int = 150
    dilate_kernel: int = 3
    dilate_shape: str = "ellipse"
    dilate_iterations: int = 1
    post_blur_kernel: int = 9
    post_blur_sigma: float = 2.0
    noise_sigma: float = 10.0
    augment_max_dilate_delta: int = 1
    augment_blur_scale_min: float = 0.85
    augment_blur_scale_max: float = 1.25
    augment_noise_scale_min: float = 0.5
    augment_noise_scale_max: float = 1.5


def _validate_kernel_size(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd, got {value}")
    return value


def _validate_config(config: StructureConfig) -> None:
    _validate_kernel_size(config.pre_blur_kernel, "pre_blur_kernel")
    _validate_kernel_size(config.post_blur_kernel, "post_blur_kernel")
    if config.dilate_kernel < 1:
        raise ValueError(f"dilate_kernel must be >= 1, got {config.dilate_kernel}")
    if config.dilate_shape not in MORPH_SHAPES:
        raise ValueError(
            f"dilate_shape must be one of {sorted(MORPH_SHAPES)}, got {config.dilate_shape!r}"
        )
    if config.dilate_iterations < 0:
        raise ValueError(f"dilate_iterations must be >= 0, got {config.dilate_iterations}")
    if config.post_blur_sigma < 0 or config.noise_sigma < 0:
        raise ValueError("post_blur_sigma and noise_sigma must be >= 0")


def _to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            gray = arr[:, :, 0]
        else:
            gray = arr[:, :, :3].astype(np.float32).mean(axis=2)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {arr.shape}")

    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)

    gray = gray.astype(np.float32)
    gray = np.clip(gray, 0, 255)
    return np.ascontiguousarray(gray.astype(np.uint8))


def _deterministic_rng_from_array(array: np.ndarray) -> np.random.Generator:
    contiguous = np.ascontiguousarray(array)
    seed = zlib.adler32(contiguous.view(np.uint8)) & 0xFFFFFFFF
    shape_seed = zlib.adler32(str(contiguous.shape).encode("utf-8"), seed) & 0xFFFFFFFF
    return np.random.default_rng(shape_seed)


def _structure_kernel(config: StructureConfig) -> np.ndarray:
    return cv2.getStructuringElement(
        MORPH_SHAPES[config.dilate_shape],
        (config.dilate_kernel, config.dilate_kernel),
    )


def _duplicate_channels(image: np.ndarray) -> np.ndarray:
    return np.repeat(image[:, :, None], 3, axis=2)


def _normalize_to_float(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.clip(normalized, 0, 255)
    return normalized.astype(np.float32)


def raster_to_structure(
    edge_like_raster: np.ndarray,
    config: StructureConfig | None = None,
    rng: np.random.Generator | None = None,
    return_rgb: bool = False,
) -> np.ndarray:
    config = config or StructureConfig()
    _validate_config(config)

    edge_map = _to_grayscale_uint8(edge_like_raster)
    if not np.any(edge_map):
        empty = np.zeros_like(edge_map, dtype=np.uint8)
        return _duplicate_channels(empty) if return_rgb else empty

    working = edge_map
    if config.dilate_iterations > 0:
        working = cv2.dilate(working, _structure_kernel(config), iterations=config.dilate_iterations)

    working = cv2.GaussianBlur(
        working,
        (config.post_blur_kernel, config.post_blur_kernel),
        sigmaX=config.post_blur_sigma,
        sigmaY=config.post_blur_sigma,
    ).astype(np.float32)

    if float(working.max()) <= float(working.min()):
        structure = np.clip(working, 0, 255)
    else:
        structure = _normalize_to_float(working)

    noise_rng = rng if rng is not None else _deterministic_rng_from_array(edge_map)
    if config.noise_sigma > 0:
        structure = structure + noise_rng.normal(0.0, config.noise_sigma, size=structure.shape)

    structure = np.clip(structure, 0, 255).astype(np.uint8)
    return _duplicate_channels(structure) if return_rgb else structure


def image_to_structure(
    image: np.ndarray,
    config: StructureConfig | None = None,
    rng: np.random.Generator | None = None,
    return_rgb: bool = False,
) -> np.ndarray:
    config = config or StructureConfig()
    _validate_config(config)

    grayscale = _to_grayscale_uint8(image)
    preblurred = cv2.GaussianBlur(
        grayscale,
        (config.pre_blur_kernel, config.pre_blur_kernel),
        sigmaX=0,
        sigmaY=0,
    )
    edges = cv2.Canny(preblurred, config.canny_low, config.canny_high)
    return raster_to_structure(edges, config=config, rng=rng, return_rgb=return_rgb)


def augment_structure_map(
    structure: np.ndarray,
    config: StructureConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    config = config or StructureConfig()
    _validate_config(config)

    aug_rng = rng if rng is not None else np.random.default_rng()
    preserve_rgb = np.asarray(structure).ndim == 3 and np.asarray(structure).shape[2] >= 3
    working = _to_grayscale_uint8(structure).astype(np.float32)

    if not np.any(working):
        empty = np.zeros_like(working, dtype=np.uint8)
        return _duplicate_channels(empty) if preserve_rgb else empty

    delta = int(aug_rng.integers(-config.augment_max_dilate_delta, config.augment_max_dilate_delta + 1))
    kernel = _structure_kernel(config)
    if delta > 0:
        working = cv2.dilate(working.astype(np.uint8), kernel, iterations=delta).astype(np.float32)
    elif delta < 0:
        working = cv2.erode(working.astype(np.uint8), kernel, iterations=abs(delta)).astype(np.float32)

    blur_scale = float(
        aug_rng.uniform(config.augment_blur_scale_min, config.augment_blur_scale_max)
    )
    blur_sigma = max(0.1, config.post_blur_sigma * blur_scale)
    working = cv2.GaussianBlur(
        working,
        (config.post_blur_kernel, config.post_blur_kernel),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
    ).astype(np.float32)

    noise_sigma = float(
        config.noise_sigma
        * aug_rng.uniform(config.augment_noise_scale_min, config.augment_noise_scale_max)
    )
    working = np.clip(working + aug_rng.normal(0.0, noise_sigma, size=working.shape), 0, 255).astype(np.uint8)

    return _duplicate_channels(working) if preserve_rgb else working


def _iter_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _seed_for_path(base_seed: int, relative_path: Path) -> int:
    return zlib.adler32(relative_path.as_posix().encode("utf-8"), base_seed) & 0xFFFFFFFF


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config: StructureConfig | None = None,
    save_rgb: bool = False,
    seed: int = 0,
) -> int:
    source_root = Path(input_dir)
    target_root = Path(output_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {source_root}")

    config = config or StructureConfig()
    _validate_config(config)

    image_paths = list(_iter_image_files(source_root))
    if not image_paths:
        raise RuntimeError(f"No supported images found in: {source_root}")

    processed = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        relative_path = image_path.relative_to(source_root)
        out_path = (target_root / relative_path).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        file_seed = _seed_for_path(seed, relative_path)
        rng = np.random.default_rng(file_seed)
        structure = image_to_structure(image, config=config, rng=rng, return_rgb=save_rgb)
        cv2.imwrite(str(out_path), structure)
        processed += 1

    return processed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structural maps from PCB images.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory with PCB images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for structural maps.")
    parser.add_argument(
        "--save_rgb",
        action="store_true",
        help="Save 3-channel RGB copies instead of single-channel maps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed used for deterministic per-file noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    processed = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        save_rgb=args.save_rgb,
        seed=args.seed,
    )
    print(f"Processed {processed} image(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
