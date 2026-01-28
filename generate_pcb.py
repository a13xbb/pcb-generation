import os
import cv2
import shutil
import numpy as np

from utils import filter_pads, split_trace_instances

def main():
    filter_pads(
        folder="PADS",
        trash_folder="trash",
        margin=10
    )

    split_trace_instances(
        folder="COPPER_TRACES",
        trash_folder="trash",
        min_area=200
    )

            
if __name__ == "__main__":
    main()
