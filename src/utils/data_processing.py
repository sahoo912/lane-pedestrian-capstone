"""
Data processing utilities.
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def load_video(video_path):
    """
    Load a video file as a capture object.
    
    Args:
        video_path: Path to video file
        
    Returns:
        OpenCV VideoCapture object or None if loading failed
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        logger.info(f"Loaded video from {video_path}")
        return cap
    except Exception as e:
        logger.error(f"Error loading video: {e}")
        return None

def preprocess_frame(frame, resize=None):
    """
    Preprocess a frame for analysis.
    
    Args:
        frame: Input frame
        resize: Optional tuple (width, height) to resize frame
        
    Returns:
        Preprocessed frame
    """
    if frame is None:
        return None
        
    # Make a copy to avoid modifying the original
    processed = frame.copy()
    
    # Resize if specified
    if resize:
        processed = cv2.resize(processed, resize)
    
    return processed