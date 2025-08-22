"""
Lane detection module for identifying road lane boundaries.
"""

import cv2
import numpy as np

class LaneDetector:
    """
    Class for detecting lane boundaries using edge detection and line fitting techniques.
    """
    
    def __init__(self, config=None):
        """
        Initialize the lane detector.
        
        Args:
            config: Optional configuration dictionary for lane detection parameters.
        """
        # Default configuration
        self.config = {
            'roi_vertices': None,  # Region of interest vertices
            'canny_low_threshold': 50,
            'canny_high_threshold': 150,
            'gaussian_kernel_size': 5,
            'hough_rho': 1,
            'hough_theta': np.pi/180,
            'hough_threshold': 20,
            'hough_min_line_length': 20,
            'hough_max_line_gap': 300
        }
        
        # Update config if provided
        if config:
            self.config.update(config)
    
    def detect(self, frame):
        """
        Detect lane boundaries in the given frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            Dictionary containing lane information:
            - left_lane: Parameters of the left lane boundary
            - right_lane: Parameters of the right lane boundary
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            gray, 
            (self.config['gaussian_kernel_size'], self.config['gaussian_kernel_size']),
            0
        )
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.config['canny_low_threshold'],
            self.config['canny_high_threshold']
        )
        
        # Apply region of interest mask
        if self.config['roi_vertices'] is None:
            # Default ROI if not specified: bottom half trapezoid
            height, width = frame.shape[:2]
            self.config['roi_vertices'] = np.array([
                [(0, height), (width / 2 - 50, height / 2 + 50),
                 (width / 2 + 50, height / 2 + 50), (width, height)]
            ], dtype=np.int32)
        
        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, self.config['roi_vertices'], 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(
            masked_edges,
            self.config['hough_rho'],
            self.config['hough_theta'],
            self.config['hough_threshold'],
            None,
            self.config['hough_min_line_length'],
            self.config['hough_max_line_gap']
        )
        
        # Process lines to identify left and right lanes
        left_lane, right_lane = self._process_lines(lines, frame.shape[:2])
        
        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'edges': edges,
            'masked_edges': masked_edges
        }
    
    def _process_lines(self, lines, frame_shape):
        """
        Process detected lines to identify left and right lane boundaries.
        
        Args:
            lines: Lines detected by Hough transform
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Tuple of (left_lane, right_lane) parameters
        """
        height, width = frame_shape
        
        # Lists to store left and right lane segments
        left_segments = []
        right_segments = []
        
        if lines is None:
            return None, None
            
        # Separate line segments by slope
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Filter out horizontal lines
                if x2 - x1 == 0:
                    continue
                
                # Calculate slope and intercept
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Categorize lines based on slope
                if abs(slope) < 0.1:
                    # Skip near-horizontal lines
                    continue
                elif slope < 0:
                    left_segments.append((slope, intercept))
                else:
                    right_segments.append((slope, intercept))
        
        # Average left and right segments
        left_lane = self._average_lane(left_segments, height)
        right_lane = self._average_lane(right_segments, height)
        
        return left_lane, right_lane
    
    def _average_lane(self, segments, height):
        """
        Average multiple line segments to get a single lane line.
        
        Args:
            segments: List of (slope, intercept) tuples
            height: Height of the frame
            
        Returns:
            Lane parameters as (x1, y1, x2, y2) or None if no segments
        """
        if not segments:
            return None
        
        # Average slope and intercept
        avg_slope = np.mean([slope for slope, _ in segments])
        avg_intercept = np.mean([intercept for _, intercept in segments])
        
        # Calculate lane line endpoints
        y1 = height
        y2 = int(height * 0.6)  # Extend to 60% up the image
        
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        
        return (x1, y1, x2, y2)