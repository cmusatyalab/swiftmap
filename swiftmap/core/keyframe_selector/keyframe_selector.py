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
from typing import Any, Dict, List, Optional

from swiftmap.core import constants
from swiftmap.core.keyframe_selector.frame_tracker import FrameTracker


class KeyframeSelector:
    """Optical-flow keyframe selector: a pure decision over a single frame."""

    def __init__(self, min_disparity: float = constants.DEFAULT_MIN_DISPARITY,
                 visualize_flow: bool = False, keep_all: bool = False):
        """
        Args:
            min_disparity: minimum mean optical-flow disparity (pixels) for a frame
                           to be selected as a keyframe.
            visualize_flow: draw optical-flow vectors (debugging) in compute_disparity.
            keep_all: if True, skip optical-flow selection -- every frame is a
                      candidate keyframe (the session's cap then bounds them as a FIFO
                      window). Useful for a denser, unfiltered trajectory.
        """
        self.min_disparity = min_disparity
        self.visualize_flow = visualize_flow
        self.keep_all = keep_all

        # Stateful optical-flow tracker (tracks features vs. the last keyframe).
        self.frame_tracker = FrameTracker()

        self.stats = {
            "total_frames_processed": 0,
            "keyframes_selected": 0,
            "processing_times": [],
        }

        # Per-selected-keyframe score (the disparity that selected it), in selection
        # order. Carried by the session into each keyframe's metadata.
        self.keyframe_values = []

        # Optical-flow overlay (BGR) of the last *selected* keyframe, so the preview
        # shows what was actually kept and holds steady between selections.
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
        if self.keep_all:
            # No selection -> every frame is kept; the preview is just the raw last
            # frame added (no optical-flow overlay). Record 0.0 as the (unused) score.
            if self.visualize_flow:
                self._keyframe_vis = image
            self._update_stats(time.time() - start, True)
            self.keyframe_values.append(0.0)
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
            self.keyframe_values.append(float(self.frame_tracker.last_disparity))
            # Freeze the preview on the frame that was actually kept.
            self._keyframe_vis = self.frame_tracker.last_flow_vis
        return selected

    def refine_to_cap(self, paths: List[str], cap: int) -> List[int]:
        """Re-select among already-selected keyframes so at most ``cap`` remain.

        Re-runs the sequential optical-flow selection over the saved keyframe
        images, tightening the disparity threshold by a fixed factor each round
        (starting from ``min_disparity``) and filtering the previous round's
        survivors, until the batch fits the cap. If tightening does not converge
        (e.g. tracking keeps failing between now-distant frames, forcing
        keyframes), falls back to an even temporal subsample.

        Live selection state (``frame_tracker``, stats) is untouched: each round
        uses a throwaway tracker, so this is safe to call between captures.

        Args:
            paths: keyframe image paths, in capture order.
            cap: maximum number of keyframes to keep (<= 0 means no cap).

        Returns:
            Sorted indices into ``paths`` of the keyframes to keep.
        """
        n = len(paths)
        if cap <= 0 or n <= cap:
            return list(range(n))

        threshold = self.min_disparity if self.min_disparity > 0 else 1.0
        survivors = list(range(n))
        for _ in range(constants.KEYFRAME_REFINE_MAX_ROUNDS):
            threshold *= constants.KEYFRAME_REFINE_FACTOR
            tracker = FrameTracker()
            kept = []
            for idx in survivors:
                image = cv2.imread(paths[idx])
                if image is None:
                    continue  # unreadable file: drop it
                if tracker.compute_disparity(image, threshold, verbose=False):
                    kept.append(idx)
            survivors = kept or survivors
            print(f"Keyframe refinement: threshold {threshold:.1f} px -> "
                  f"{len(survivors)}/{n} keyframes")
            if len(survivors) <= cap:
                return survivors

        print(f"Keyframe refinement did not converge; evenly subsampling "
              f"{len(survivors)} -> {cap} keyframes")
        picks = np.linspace(0, len(survivors) - 1, cap).round().astype(int)
        return [survivors[i] for i in sorted(set(picks.tolist()))]

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
        self.keyframe_values = []
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
