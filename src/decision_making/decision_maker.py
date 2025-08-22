"""
Decision making module for autonomous driving actions.
"""

import logging
import numpy as np

class DecisionMaker:
    """
    Class for making driving decisions based on lane positions and pedestrian intent.
    """
    
    def __init__(self, config=None):
        """
        Initialize the decision maker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            'stop_distance_threshold': 10.0,  # meters
            'slow_distance_threshold': 20.0,  # meters
            'crossing_confidence_threshold': 0.6,
            'waiting_confidence_threshold': 0.7,
            'lane_deviation_threshold': 0.2  # proportion of lane width
        }
        
        # Update config if provided
        if config:
            self.config.update(config)
    
    def decide(self, lanes, pedestrians):
        """
        Make driving decisions based on lane positions and pedestrian intent.
        
        Args:
            lanes: Lane detection results
            pedestrians: List of pedestrian detections with pose analysis
            
        Returns:
            Dictionary containing:
            - action: One of ['maintain', 'slow', 'stop']
            - steering: Steering angle in degrees
            - explanation: Text explanation of decision
        """
        # Default decision
        decision = {
            'action': 'maintain',
            'steering': 0.0,
            'explanation': 'No obstacles detected.'
        }
        
        # Calculate steering based on lane positions
        steering = self._calculate_steering(lanes)
        decision['steering'] = steering
        
        # Check for pedestrians with crossing intent
        stop_required = False
        slow_required = False
        critical_pedestrian = None
        
        for pedestrian in pedestrians:
            if pedestrian.get('intent') == 'crossing' and \
               pedestrian.get('confidence', 0) >= self.config['crossing_confidence_threshold']:
                # Assume distance is encoded in the bounding box size
                # In a real system, this would come from depth estimation or other sensors
                estimated_distance = self._estimate_distance(pedestrian)
                
                if estimated_distance < self.config['stop_distance_threshold']:
                    stop_required = True
                    critical_pedestrian = pedestrian
                    break
                elif estimated_distance < self.config['slow_distance_threshold']:
                    slow_required = True
                    if critical_pedestrian is None or \
                       self._estimate_distance(pedestrian) < self._estimate_distance(critical_pedestrian):
                        critical_pedestrian = pedestrian
        
        # Make final decision
        if stop_required:
            decision['action'] = 'stop'
            decision['explanation'] = 'Stopping for pedestrian with crossing intent.'
        elif slow_required:
            decision['action'] = 'slow'
            decision['explanation'] = 'Slowing down for pedestrian with potential crossing intent.'
        else:
            # Check if we need lane centering
            if abs(steering) > self.config['lane_deviation_threshold']:
                decision['explanation'] = f'Correcting lane position (steering: {steering:.2f} degrees).'
            else:
                decision['explanation'] = 'Maintaining current trajectory.'
        
        return decision
    
    def _calculate_steering(self, lanes):
        """
        Calculate steering angle based on lane positions.
        
        Args:
            lanes: Lane detection results
            
        Returns:
            Steering angle in degrees
        """
        # If no lanes detected, maintain current steering
        if not lanes or (lanes['left_lane'] is None and lanes['right_lane'] is None):
            return 0.0
        
        # Calculate center position between lanes
        if lanes['left_lane'] is not None and lanes['right_lane'] is not None:
            # Both lanes detected
            left_x = (lanes['left_lane'][0] + lanes['left_lane'][2]) / 2
            right_x = (lanes['right_lane'][0] + lanes['right_lane'][2]) / 2
            center_x = (left_x + right_x) / 2
            
            # Calculate image center
            # Assuming frame width is encoded in lane positions
            frame_width = max(right_x, left_x) * 2  # Rough estimate
            image_center = frame_width / 2
            
            # Calculate deviation
            deviation = (center_x - image_center) / frame_width
            
            # Convert to steering angle (simple linear model)
            steering_angle = -deviation * 25.0  # Max 25 degrees for full deviation
            
        elif lanes['left_lane'] is not None:
            # Only left lane detected
            left_x = (lanes['left_lane'][0] + lanes['left_lane'][2]) / 2
            # Assume lane width
            estimated_right_x = left_x + 450  # Typical lane width in pixels
            center_x = (left_x + estimated_right_x) / 2
            
            # Estimate image center
            frame_width = estimated_right_x * 1.5
            image_center = frame_width / 2
            
            # Calculate deviation
            deviation = (center_x - image_center) / frame_width
            steering_angle = -deviation * 15.0  # Less confidence, reduced angle
            
        else:
            # Only right lane detected
            right_x = (lanes['right_lane'][0] + lanes['right_lane'][2]) / 2
            # Assume lane width
            estimated_left_x = right_x - 450  # Typical lane width in pixels
            center_x = (estimated_left_x + right_x) / 2
            
            # Estimate image center
            frame_width = right_x * 1.5
            image_center = frame_width / 2
            
            # Calculate deviation
            deviation = (center_x - image_center) / frame_width
            steering_angle = -deviation * 15.0  # Less confidence, reduced angle
        
        return steering_angle
    
    def _estimate_distance(self, pedestrian):
        """
        Estimate distance to a pedestrian based on bounding box.
        
        In a real system, this would use depth sensors or stereo vision.
        Here we use a simple heuristic based on bounding box size.
        
        Args:
            pedestrian: Pedestrian detection data
            
        Returns:
            Estimated distance in arbitrary units
        """
        # This is a placeholder implementation
        # In a real system, accurate distance would come from sensors
        
        # Simple inverse relationship between box height and distance
        if 'bbox' in pedestrian:
            bbox = pedestrian['bbox']
            height = bbox[3] - bbox[1]
            # Crude estimate: assume 1.75m tall person fills the frame at 2m distance
            # Assuming frame height is around 720 pixels
            estimated_distance = (720 / height) * 2 if height > 0 else float('inf')
            return estimated_distance
        
        # If no bounding box, return a large value
        return float('inf')