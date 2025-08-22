"""
Visualization utilities for lane and pedestrian detection.
"""

import cv2
import numpy as np

def visualize_results(frame, results):
    """
    Visualize detection results on the frame.
    
    Args:
        frame: Original frame to annotate
        results: Detection results containing lanes and pedestrian data
        
    Returns:
        Annotated frame
    """
    output_frame = frame.copy()
    
    # Draw lane boundaries
    if 'lanes' in results:
        lanes = results['lanes']
        
        # Draw left lane
        if lanes.get('left_lane') is not None:
            x1, y1, x2, y2 = lanes['left_lane']
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # Draw right lane
        if lanes.get('right_lane') is not None:
            x1, y1, x2, y2 = lanes['right_lane']
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    
    # Draw pedestrians and their intent
    if 'pedestrians' in results:
        for i, box in enumerate(results['pedestrians']):
            x1, y1, x2, y2 = box
            
            # Get pedestrian pose if available
            pose_data = results.get('poses', [])[i] if i < len(results.get('poses', [])) else None
            
            # Determine color based on intent
            if pose_data and 'intent' in pose_data:
                intent = pose_data['intent']
                confidence = pose_data.get('confidence', 0.0)
                
                if intent == 'crossing':
                    color = (0, 0, 255)  # Red for crossing
                elif intent == 'waiting':
                    color = (0, 255, 0)  # Green for waiting
                else:
                    color = (255, 165, 0)  # Orange for unknown
                    
                # Draw bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw intent label
                label = f"{intent} ({confidence:.2f})"
                cv2.rectangle(output_frame, (x1, y1-25), (x1+len(label)*9, y1), color, -1)
                cv2.putText(output_frame, label, (x1, y1-7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw pose landmarks if available
                if pose_data and 'landmarks' in pose_data and pose_data['landmarks']:
                    landmarks = pose_data['landmarks']
                    for j, landmark in enumerate(landmarks):
                        if landmark[2] > 0.5:  # Only visible landmarks
                            x, y = int(landmark[0] * (x2 - x1) + x1), int(landmark[1] * (y2 - y1) + y1)
                            cv2.circle(output_frame, (x, y), 3, color, -1)
            else:
                # No pose data, draw simple bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
    
    # Draw decision info
    if 'decision' in results:
        decision = results['decision']
        action = decision.get('action', 'unknown')
        steering = decision.get('steering', 0.0)
        explanation = decision.get('explanation', '')
        
        # Set action color
        if action == 'stop':
            action_color = (0, 0, 255)  # Red
        elif action == 'slow':
            action_color = (0, 165, 255)  # Orange
        else:
            action_color = (0, 255, 0)  # Green
            
        # Draw action rectangle
        height, width = output_frame.shape[:2]
        cv2.rectangle(output_frame, (0, height-80), (width, height), (0, 0, 0), -1)
        cv2.rectangle(output_frame, (10, height-70), (150, height-10), action_color, -1)
        cv2.putText(output_frame, action.upper(), (20, height-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw steering indicator
        steering_x = width - 100
        steering_y = height - 40
        cv2.circle(output_frame, (steering_x, steering_y), 30, (255, 255, 255), 2)
        steer_rad = steering * np.pi / 180
        end_x = int(steering_x + 25 * np.sin(steer_rad))
        end_y = int(steering_y - 25 * np.cos(steer_rad))
        cv2.line(output_frame, (steering_x, steering_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Draw explanation
        cv2.putText(output_frame, explanation, (170, height-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return output_frame