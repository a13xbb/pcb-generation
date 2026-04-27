from __future__ import annotations
import os
import env
os.environ['HF_HOME'] = env.HF_HOME

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PairRecord:
    stem: str
    image_path: Path
    structure_path: Path


class PairedPCBControlNetDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        structures_dir: str | Path,
        resolution: int,
        max_pairs: int,
        seed: int,
        augment: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.structures_dir = Path(structures_dir)
        self.augment = augment
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")
        if not self.structures_dir.exists():
            raise FileNotFoundError(f"Structure maps directory does not exist: {self.structures_dir}")

        image_index = self._index_by_stem(self.images_dir)
        structure_index = self._index_by_stem(self.structures_dir)
        paired_stems = sorted(image_index.keys() & structure_index.keys())
        if not paired_stems:
            raise RuntimeError("No paired structure/image files found by basename.")

        self.unmatched_images = sorted(image_index.keys() - structure_index.keys())
        self.unmatched_structures = sorted(structure_index.keys() - image_index.keys())
        records = [
            PairRecord(
                stem=stem,
                image_path=image_index[stem],
                structure_path=structure_index[stem],
            )
            for stem in paired_stems
        ]

        rng = random.Random(seed)
        if max_pairs > 0 and len(records) > max_pairs:
            records = rng.sample(records, max_pairs)
            records.sort(key=lambda item: item.stem)

        self.records = records

        self.resize_crop = v2.Compose([
            v2.Resize(
                resolution,
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.CenterCrop(resolution),
        ])

        self.spatial_augment = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.25),
        ])

        self.color_augment = v2.Compose([
            v2.RandomApply([
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
            ], p=0.3),
            v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
        ])

        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.normalize = v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.stats = {
            "images_dir": str(self.images_dir),
            "structures_dir": str(self.structures_dir),
            "paired_count": len(self.records),
            "unmatched_images_count": len(self.unmatched_images),
            "unmatched_structures_count": len(self.unmatched_structures),
            "unmatched_images_sample": self.unmatched_images[:20],
            "unmatched_structures_sample": self.unmatched_structures[:20],
            "augment": augment,
        }

    @staticmethod
    def _index_by_stem(root: Path) -> Dict[str, Path]:
        files = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        duplicates: Dict[str, List[Path]] = {}
        index: Dict[str, Path] = {}
        for path in files:
            if path.stem in index:
                duplicates.setdefault(path.stem, [index[path.stem]]).append(path)
            else:
                index[path.stem] = path

        if duplicates:
            sample = {stem: [str(item) for item in paths] for stem, paths in list(duplicates.items())[:10]}
            raise RuntimeError(f"Duplicate basenames found in {root}: {sample}")

        return index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        structure_gray = Image.open(record.structure_path).convert("L")
        structure = Image.merge("RGB", (structure_gray, structure_gray, structure_gray))

        image = self.resize_crop(image)
        structure = self.resize_crop(structure)

        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()

            torch.manual_seed(seed)
            image = self.spatial_augment(image)
            torch.manual_seed(seed)
            structure = self.spatial_augment(structure)

            image = self.color_augment(image)

        image_tensor = self.normalize(self.to_tensor(image))
        structure_tensor = self.to_tensor(structure)

        return {
            "pixel_values": image_tensor,
            "conditioning_pixel_values": structure_tensor,
            "stem": record.stem,
        }
