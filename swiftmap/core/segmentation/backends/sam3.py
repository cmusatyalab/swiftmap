# Copyright (C) 2024 Carnegie Mellon University

"""SAM 3 text-promptable segmentation backend.

Wraps Meta's SAM 3 image model behind ``BaseSegmenter``. SAM 3 does promptable
*concept* segmentation: given a short phrase ("person") it detects and segments
every instance of that concept. For v1 we union the per-instance masks into one
semantic mask per frame; the per-instance masks remain available for a future
tracked-instance upgrade.

Weights load from a local checkpoint (``constants.SAM3_CHECKPOINT``) or, if that
is unset, are downloaded from the HuggingFace Hub by the SAM 3 builder. The heavy
``sam3`` package is imported lazily so it is only loaded when segmentation runs.
"""

import os
from typing import Optional

import numpy as np
import torch

from swiftmap.core import constants
from swiftmap.core.segmentation.base import BaseSegmenter
from swiftmap.core.segmentation.registry import register_segmenter


@register_segmenter(
    "sam3",
    label="SAM 3",
    description="Meta SAM 3 promptable concept segmentation (text -> instance masks).",
)
class SAM3Segmenter(BaseSegmenter):
    """Text-promptable segmentation via SAM 3."""

    def __init__(self, device=None,
                 checkpoint_path: Optional[str] = None,
                 bpe_path: Optional[str] = None,
                 confidence_threshold: Optional[float] = None):
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else constants.SAM3_CHECKPOINT
        self.bpe_path = bpe_path or constants.SAM3_BPE_PATH
        self.confidence_threshold = (confidence_threshold if confidence_threshold is not None
                                     else constants.SAM3_CONF_THRESHOLD)
        self.processor = None

    def initialize_model(self) -> bool:
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            ckpt = self.checkpoint_path if (self.checkpoint_path and os.path.isfile(self.checkpoint_path)) else None
            bpe = self.bpe_path if (self.bpe_path and os.path.isfile(self.bpe_path)) else None
            if ckpt:
                print(f"Initializing SAM 3 from local checkpoint {ckpt} ...")
            else:
                print("Initializing SAM 3 (no local checkpoint -> downloading facebook/sam3 from HF)...")

            self.model = build_sam3_image_model(
                bpe_path=bpe,
                device=self.device,
                eval_mode=True,
                checkpoint_path=ckpt,
                load_from_HF=(ckpt is None),
            )
            self.processor = Sam3Processor(
                self.model, device=self.device,
                confidence_threshold=self.confidence_threshold)
            self.is_initialized = True
            print("SAM 3 initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing SAM 3: {e}")
            self.is_initialized = False
            return False

    def _segment_image(self, image_rgb: np.ndarray, query: str) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(np.ascontiguousarray(image_rgb), mode="RGB")
        h, w = image_rgb.shape[:2]

        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device == "cuda" else _nullcontext())
        with autocast:
            state = self.processor.set_image(pil)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=query, state=state)

        masks = state.get("masks")  # (N, 1, H, W) bool tensor at input resolution
        if masks is None or len(masks) == 0:
            return np.zeros((h, w), dtype=bool)
        if masks.ndim == 4:            # drop the channel dim -> (N, H, W)
            masks = masks.squeeze(1)
        union = masks.any(dim=0)       # (H, W)
        arr = union.detach().cpu().numpy().astype(bool)
        if arr.shape != (h, w):        # safety: align exactly to the frame grid
            import cv2
            arr = cv2.resize(arr.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        return arr


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False
