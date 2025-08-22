"""
Posture analysis module for pedestrian intent prediction.
"""

import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle
from pathlib import Path

class PostureAnalyzer:
    """
    Class for analyzing pedestrian posture to predict crossing intent.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the posture analyzer.
        
        Args:
            model_path: Optional path to a trained classifier model
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Intent classifier (either load or use a simple rule-based one)
        self.classifier = None
        if model_path:
            try:
                with open(model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.logger.info(f"Loaded posture classifier from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load classifier: {e}")
    
    def analyze(self, frame, bbox=None):
        """
        Analyze pedestrian posture and predict intent.
        
        Args:
            frame: Input frame (BGR image)
            bbox: Optional bounding box of the pedestrian
            
        Returns:
            Dictionary containing:
            - landmarks: Detected pose landmarks
            - intent: Predicted intent (crossing, waiting, other)
            - confidence: Confidence in the prediction
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image to find pose landmarks
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {
                'landmarks': None,
                'intent': 'unknown',
                'confidence': 0.0,
                'feature_vector': None
            }
        
        # Extract landmarks as a list of (x, y, visibility) tuples
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.visibility))
        
        # Extract features from landmarks
        feature_vector = self._extract_features(landmarks)
        
        # Predict intent
        intent, confidence = self._predict_intent(feature_vector)
        
        return {
            'landmarks': landmarks,
            'intent': intent,
            'confidence': confidence,
            'feature_vector': feature_vector
        }
    
    def _extract_features(self, landmarks):
        """
        Extract meaningful features from pose landmarks.
        
        Args:
            landmarks: List of pose landmarks
            
        Returns:
            Feature vector for intent classification
        """
        # Extract key joints for analysis
        if not landmarks:
            return None
            
        # Calculate relevant features (angles, distances, etc.)
        features = []
        
        # Position and angle of legs (to detect walking vs. standing)
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calculate leg angles
        left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        features.extend([left_leg_angle, right_leg_angle])
        
        # Calculate distance between ankles (normalized by hip width)
        hip_width = np.sqrt((right_hip[0] - left_hip[0])**2 + 
                             (right_hip[1] - left_hip[1])**2)
        ankle_distance = np.sqrt((right_ankle[0] - left_ankle[0])**2 + 
                                  (right_ankle[1] - left_ankle[1])**2)
        normalized_ankle_distance = ankle_distance / hip_width if hip_width > 0 else 0
        features.append(normalized_ankle_distance)
        
        # Head position relative to hips (to detect looking behavior)
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        hip_center = ((left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2)
        head_offset_x = nose[0] - hip_center[0]
        features.append(head_offset_x)
        
        # More features can be added here
        
        return features
    
    def _calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points.
        
        Args:
            a, b, c: Points as (x, y, visibility) tuples
            
        Returns:
            Angle in degrees
        """
        # Extract coordinates
        a_coords = np.array([a[0], a[1]])
        b_coords = np.array([b[0], b[1]])
        c_coords = np.array([c[0], c[1]])
        
        # Calculate vectors
        ba = a_coords - b_coords
        bc = c_coords - b_coords
        
        # Calculate cosine of angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle numerical errors
        
        # Calculate angle in degrees
        angle = np.arccos(cosine_angle) * 180 / np.pi
        
        return angle
    
    def _predict_intent(self, feature_vector):
        """
        Predict pedestrian intent based on pose features.
        
        Args:
            feature_vector: Extracted features from pose landmarks
            
        Returns:
            Tuple of (intent, confidence)
        """
        if feature_vector is None:
            return "unknown", 0.0
            
        if self.classifier:
            # Use trained classifier if available
            intent_proba = self.classifier.predict_proba([feature_vector])[0]
            intent_idx = np.argmax(intent_proba)
            intent = self.classifier.classes_[intent_idx]
            confidence = intent_proba[intent_idx]
        else:
            # Simple rule-based logic if no classifier is available
            leg_angles = feature_vector[:2]
            ankle_distance = feature_vector[2]
            
            if ankle_distance > 0.5:
                # Wide stance or mid-stride
                intent = "crossing"
                confidence = min(1.0, ankle_distance * 0.8)
            elif abs(leg_angles[0] - leg_angles[1]) > 30:
                # Different leg angles often indicate movement
                intent = "crossing"
                confidence = min(1.0, abs(leg_angles[0] - leg_angles[1]) / 90)
            else:
                # Similar leg angles often indicate standing
                intent = "waiting"
                confidence = min(1.0, (180 - abs(leg_angles[0] - leg_angles[1])) / 180)
        
        return intent, float(confidence)