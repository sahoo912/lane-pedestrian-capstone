"""
Pedestrian detection module using YOLOv8.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO

class PedestrianDetector:
    """
    Class for detecting pedestrians in images using YOLOv8.
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize the pedestrian detector.
        
        Args:
            model_path: Path to the YOLOv8 model weights
            confidence_threshold: Minimum confidence score for detections
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        
        # Use default model path if not specified
        if model_path is None:
            model_path = Path("models/yolov8n.pt")
            
            # Check if model exists, otherwise download it
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"YOLOv8 model not found. Using default model.")
        
        try:
            self.model = YOLO(model_path)
            self.logger.info(f"Loaded YOLOv8 model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect pedestrians in the given frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            List of bounding boxes in format [x1, y1, x2, y2]
        """
        # Run inference
        results = self.model(frame, classes=0)  # class 0 is person in COCO dataset
        
        # Extract pedestrian bounding boxes
        pedestrian_boxes = []
        
        for result in results:
            for box in result.boxes:
                # Check if detection is a person and confidence exceeds threshold
                if (box.cls == 0 and box.conf >= self.confidence_threshold):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    pedestrian_boxes.append([x1, y1, x2, y2])
        
        return pedestrian_boxes