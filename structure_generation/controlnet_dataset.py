from __future__ import annotations
import os
os.environ['HF_HOME'] = '/mnt/ssdm2/users/alexblokh/cache'

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
    ) -> None:
        self.images_dir = Path(images_dir)
        self.structures_dir = Path(structures_dir)
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
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.conditioning_transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )
        self.stats = {
            "images_dir": str(self.images_dir),
            "structures_dir": str(self.structures_dir),
            "paired_count": len(self.records),
            "unmatched_images_count": len(self.unmatched_images),
            "unmatched_structures_count": len(self.unmatched_structures),
            "unmatched_images_sample": self.unmatched_images[:20],
            "unmatched_structures_sample": self.unmatched_structures[:20],
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
        return {
            "pixel_values": self.image_transform(image),
            "conditioning_pixel_values": self.conditioning_transform(structure),
            "stem": record.stem,
        }
