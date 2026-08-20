# Copyright (C) 2024 Carnegie Mellon University

"""
Keyframe Selection

Pure keyframe-selection logic: decides, per incoming frame, whether it carries
enough new motion (optical-flow disparity) to be worth keeping for 3D mapping.

This class holds no network or storage state. The MappingSession owns the TCP
transport, feeds each received frame through ``is_keyframe``, and handles the
collection of selected keyframes.
"""

import time
import cv2
import numpy as np
from typing import Any, Dict, Optional

from swiftmap.core import constants
from swiftmap.core.transport.keyframe_selector.frame_tracker import FrameTracker


class KeyframeSelector:
    """Optical-flow keyframe selector: a pure decision over a single frame."""

    def __init__(self, min_disparity: float = constants.DEFAULT_MIN_DISPARITY,
                 visualize_flow: bool = False):
        """
        Args:
            min_disparity: minimum mean optical-flow disparity (pixels) for a frame
                           to be selected as a keyframe. 0 (or less) skips selection
                           entirely -- every frame is kept.
            visualize_flow: draw optical-flow vectors (debugging) in compute_disparity.
        """
        self.min_disparity = min_disparity
        self.visualize_flow = visualize_flow

        # Stateful optical-flow tracker (tracks features vs. the last keyframe).
        self.frame_tracker = FrameTracker()

        self.stats = {
            "total_frames_processed": 0,
            "keyframes_selected": 0,
            "processing_times": [],
        }

        # Optical-flow overlay (BGR) of the last selected keyframe.
        self._keyframe_vis = None

    def is_keyframe(self, image: np.ndarray) -> bool:
        """
        Decide whether ``image`` should be a keyframe.

        Args:
            image: received frame (H, W, C), BGR.

        Returns:
            True if the frame's motion vs. the last keyframe exceeds min_disparity.
        """
        start = time.time()
        if self.min_disparity <= 0:
            # No selection: keep every frame.
            if self.visualize_flow:
                self._keyframe_vis = image
            self._update_stats(time.time() - start, True)
            return True
        try:
            selected = self.frame_tracker.compute_disparity(
                image, self.min_disparity, self.visualize_flow
            )
        except Exception as e:
            print(f"Error in keyframe selection: {e}")
            selected = False
        self._update_stats(time.time() - start, selected)
        if selected:
            # Freeze the preview on the frame that was actually kept.
            self._keyframe_vis = self.frame_tracker.last_flow_vis
        return selected

    @property
    def latest_flow_vis(self) -> Optional[np.ndarray]:
        """Preview (RGB) of the last frame added to the buffer, or None.

        Without selection it is the raw last frame; with disparity scoring it is the
        optical-flow overlay of the last selected keyframe. Populated only while
        ``visualize_flow`` is enabled; holds between additions.
        """
        if self._keyframe_vis is None:
            return None
        return cv2.cvtColor(self._keyframe_vis, cv2.COLOR_BGR2RGB)

    def reset(self):
        """Reset the optical-flow tracker and per-run statistics (new session)."""
        self.frame_tracker.reset()
        self.stats["total_frames_processed"] = 0
        self.stats["keyframes_selected"] = 0
        self.stats["processing_times"] = []
        self._keyframe_vis = None

    def configure_disparity_threshold(self, min_disparity: float):
        """Update the minimum disparity threshold."""
        self.min_disparity = min_disparity

    def _update_stats(self, processing_time: float, is_keyframe: bool):
        self.stats["total_frames_processed"] += 1
        self.stats["processing_times"].append(processing_time)
        if is_keyframe:
            self.stats["keyframes_selected"] += 1
        # Keep the processing-time history bounded.
        if len(self.stats["processing_times"]) > 1000:
            self.stats["processing_times"] = self.stats["processing_times"][-500:]

    def get_stats(self) -> Dict[str, Any]:
        """Selection statistics (frame counts, timing, tracker state)."""
        pts = self.stats["processing_times"]
        stats: Dict[str, Any] = {
            "total_frames_processed": self.stats["total_frames_processed"],
            "keyframes_selected": self.stats["keyframes_selected"],
            "min_disparity": self.min_disparity,
            "frame_tracker_stats": self.frame_tracker.get_stats(),
        }
        if pts:
            stats["average_processing_time"] = float(np.mean(pts))
            stats["max_processing_time"] = float(np.max(pts))
            stats["min_processing_time"] = float(np.min(pts))
        if self.stats["total_frames_processed"] > 0:
            stats["keyframe_selection_rate"] = (
                self.stats["keyframes_selected"] / self.stats["total_frames_processed"] * 100
            )
        return stats
