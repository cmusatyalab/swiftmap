# Copyright (C) 2024 Carnegie Mellon University

"""
Lightweight Optical Flow Frame Tracker

Adapted from VGGT-SLAM/vggt_slam/frame_overlap.py FrameTracker class.
Uses CV2-based Lucas-Kanade optical flow for keyframe selection without
requiring heavy RAFT model dependencies.

This tracker identifies when sufficient motion has occurred between frames
to warrant selecting a new keyframe for 3D mapping.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class FrameTracker:
    """
    Lightweight optical flow-based frame tracker for keyframe selection.
    
    Uses Lucas-Kanade optical flow to track feature points between frames
    and determines keyframe selection based on motion disparity.
    """
    
    def __init__(self):
        """Initialize the frame tracker."""
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None
        self.frame_count = 0
        # Disparity of the last selection, used as its priority; forced ones get +inf.
        self.last_disparity = 0.0
        # Optical-flow overlay (BGR) when visualization is on; None until drawn.
        self.last_flow_vis = None

        # Feature detection parameters
        self.max_corners = 1000
        self.quality_level = 0.01
        self.min_distance = 8
        self.block_size = 7
        
        # Optical flow parameters
        self.win_size = (21, 21)
        self.max_level = 3
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        
        # Minimum number of good tracking points required
        self.min_tracking_points = 10
        
    def initialize_keyframe(self, image: np.ndarray) -> bool:
        """
        Initialize a new keyframe with feature detection.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.last_kf = image.copy()
            self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect good features to track
            self.kf_pts = cv2.goodFeaturesToTrack(
                self.kf_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size
            )
            
            if self.kf_pts is None or len(self.kf_pts) < self.min_tracking_points:
                print(f"Warning: Only {len(self.kf_pts) if self.kf_pts is not None else 0} feature points detected")
                return False
                
            print(f"Keyframe initialized with {len(self.kf_pts)} feature points")
            return True
            
        except Exception as e:
            print(f"Error initializing keyframe: {e}")
            return False
    
    def compute_disparity(self, image: np.ndarray, min_disparity: float, visualize: bool = False) -> bool:
        """
        Compute motion disparity and determine if image should be a keyframe.
        
        Args:
            image: Current input image as numpy array (H, W, C) in BGR format
            min_disparity: Minimum mean disparity threshold for keyframe selection
            visualize: Whether to display optical flow visualization
            
        Returns:
            bool: True if image should be selected as keyframe, False otherwise
        """
        try:
            self.frame_count += 1
            
            # First frame is always a keyframe
            if self.last_kf is None or self.kf_pts is None or len(self.kf_pts) < self.min_tracking_points:
                success = self.initialize_keyframe(image)
                if success:
                    self.last_disparity = float("inf")  # anchor frame: always keep
                    print(f"Frame {self.frame_count}: First keyframe selected")
                return success
            
            # Convert current frame to grayscale
            curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Track keyframe points into current frame using Lucas-Kanade optical flow
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self.kf_gray, curr_gray, self.kf_pts, None,
                winSize=self.win_size, 
                maxLevel=self.max_level,
                criteria=self.criteria
            )
            
            # Filter points based on tracking status
            status = status.flatten()
            good_kf = self.kf_pts[status == 1]
            good_next = next_pts[status == 1]
            
            # Check if we have enough good tracking points
            if len(good_kf) < self.min_tracking_points:
                print(f"Frame {self.frame_count}: Insufficient tracking points ({len(good_kf)}), reinitializing keyframe")
                success = self.initialize_keyframe(image)
                if success:
                    self.last_disparity = float("inf")  # forced keyframe: always keep
                    print(f"Frame {self.frame_count}: Keyframe selected (reinitialization)")
                return success
            
            # Compute displacement between keyframe and current frame
            displacement = np.linalg.norm(good_next - good_kf, axis=1)
            mean_disparity = np.mean(displacement)
            self.last_disparity = float(mean_disparity)
            
            # Optional visualization: cache the annotated frame.
            if visualize:
                self.last_flow_vis = self._visualize_optical_flow(
                    image, good_kf, good_next, mean_disparity, min_disparity)
            
            # Determine if disparity is large enough for new keyframe
            is_keyframe = mean_disparity > min_disparity
            
            if is_keyframe:
                print(f"Frame {self.frame_count}: Keyframe selected (disparity: {mean_disparity:.2f} > {min_disparity})")
                # Initialize new keyframe
                success = self.initialize_keyframe(image)
                return success
            else:
                print(f"Frame {self.frame_count}: Skipped (disparity: {mean_disparity:.2f} <= {min_disparity})")
                return False
                
        except Exception as e:
            print(f"Error computing disparity: {e}")
            return False
    
    def _visualize_optical_flow(self, image: np.ndarray, good_kf: np.ndarray, 
                               good_next: np.ndarray, mean_disparity: float, 
                               min_disparity: float) -> Optional[np.ndarray]:
        """
        Draw optical-flow vectors and status onto a copy of the frame.

        Args:
            image: Current input image (BGR)
            good_kf: Keyframe feature points that were successfully tracked
            good_next: Current frame feature points corresponding to good_kf
            mean_disparity: Computed mean disparity value
            min_disparity: Threshold for keyframe selection

        Returns:
            The annotated frame (BGR), or None on error. Returned (not shown) so the
            caller can surface it in the web UI — this server is headless, so there
            is no cv2.imshow window.
        """
        try:
            vis = image.copy()

            # Red once motion crosses the threshold, green while below it.
            is_kf = mean_disparity > min_disparity
            color = (0, 0, 255) if is_kf else (0, 255, 0)
            for p1, p2 in zip(good_kf, good_next):
                p1 = tuple(p1.ravel().astype(int))
                p2 = tuple(p2.ravel().astype(int))
                cv2.arrowedLine(vis, p1, p2, color=color, thickness=2, tipLength=0.3)

            # Add text information
            text = f"Mean Disparity: {mean_disparity:.2f} (Threshold: {min_disparity})"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            status_text = "KEYFRAME" if is_kf else "SKIP"
            status_color = (0, 0, 255) if is_kf else (0, 255, 0)
            cv2.putText(vis, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            return vis

        except Exception as e:
            print(f"Error in optical flow visualization: {e}")
            return None
    
    def get_stats(self) -> dict:
        """
        Get tracker statistics.
        
        Returns:
            dict: Statistics including frame count, feature points, etc.
        """
        return {
            "frame_count": self.frame_count,
            "has_keyframe": self.last_kf is not None,
            "feature_points": len(self.kf_pts) if self.kf_pts is not None else 0,
            "keyframe_initialized": self.kf_gray is not None
        }
    
    def reset(self) -> None:
        """Reset the frame tracker to initial state."""
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None
        self.frame_count = 0
        self.last_disparity = 0.0
        self.last_flow_vis = None
        print("Frame tracker reset")


def get_uniform_points(height: int, width: int, delta: int = 15) -> list:
    """
    Generate uniform grid points for visualization purposes.
    
    Args:
        height: Image height
        width: Image width  
        delta: Grid spacing
        
    Returns:
        list: List of [x, y] coordinate pairs
    """
    uniform_points = []
    for r in range(height):
        if r % delta == 0:
            for c in range(width):
                if c % delta == 0:
                    uniform_points.append([c, r])
    return uniform_points


# Example usage and testing
if __name__ == "__main__":
    # Simple test with webcam or test images
    tracker = FrameTracker()
    
    # Test with webcam if available
    try:
        cap = cv2.VideoCapture(0)
        min_disparity = 40.0
        
        print("Testing FrameTracker with webcam. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            is_keyframe = tracker.compute_disparity(frame, min_disparity, visualize=True)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        
        print("Final stats:", tracker.get_stats())
        
    except Exception as e:
        print(f"Webcam test failed: {e}")
        print("FrameTracker implementation complete - no webcam available for testing")