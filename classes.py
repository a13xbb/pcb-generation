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
    def __init__(self, asset, transform, mask_world, id=None):
        self.asset = asset
        self.transform = transform
        self.mask_world = mask_world
        self.id = id
        self.attached_traces = set()
        
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
    def __init__(self, h, w):
        self.h = h
        self.w = w

        self.occupied_mask = np.zeros((h, w), np.uint8)

        self.pad_instances = []
        self.trace_instances = []
        
