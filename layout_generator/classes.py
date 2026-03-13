import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

class PadAsset:
    def __init__(self, mask: np.ndarray, image: np.ndarray, centroid: Tuple[float, float], bbox: Tuple[int, int, int, int]):  
        self.mask = mask
        self.image = image
        self.centroid = centroid
        self.bbox = bbox
        self.h, self.w = mask.shape
    
class TraceAsset:
    def __init__(self, mask: np.ndarray, image: np.ndarray, skeleton: np.ndarray, endpoints: List[Tuple[int, int]], centroid: Tuple[float, float], length: float):
        self.mask = mask
        self.image = image
        self.skeleton = skeleton
        self.endpoints = endpoints
        self.centroid = centroid
        self.length = length
        self.h, self.w = mask.shape
        
class Transform:
    def __init__(self, angle_deg, tx, ty):
        self.angle = angle_deg
        self.tx = tx
        self.ty = ty
        
class PadInstance:
    def __init__(self, asset, transform, mask_world, id=None, bbox_world=None):
        self.asset = asset
        self.transform = transform
        self.mask_world = mask_world
        self.id = id
        self.bbox_world = bbox_world
        self.attached_traces = set()
        # Lazy caches for placement hot paths.
        self._mask_world_u8 = None
        self._attach_center = None
        self._attach_boundary = None
        self._attach_cache_ready = False
        
class TraceInstance:
    def __init__(self, asset, transform, mask_world, endpoints_world, id=None):
        self.asset = asset
        self.transform = transform
        self.mask_world = mask_world
        self.endpoints_world = endpoints_world
        self.id = id
        self.pad_end = [None, None]
        
@dataclass
class OpenEnd:
    trace_id: int
    end_idx: int              # 0 или 1
    pos_xy: Tuple[int, int]
    
class CanvasState:
    def __init__(self, h, w, pad_keepout_radius: int = 0):
        self.h = h
        self.w = w

        self.occupied_mask = np.zeros((h, w), np.uint8)
        self.pad_keepout_radius = max(0, int(pad_keepout_radius))
        self.pad_keepout_mask = np.zeros((h, w), np.uint8)

        self.pad_instances = []
        self.trace_instances = []
        self.pad_by_id = {}

    def register_pad(self, pad: PadInstance):
        if pad.id is None:
            return
        self.pad_by_id[pad.id] = pad

    def rebuild_pad_index(self):
        self.pad_by_id = {
            pad.id: pad
            for pad in self.pad_instances
            if pad.id is not None
        }
        
