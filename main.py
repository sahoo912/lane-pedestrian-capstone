#!/usr/bin/env python3
"""
Main entry point for the Lane Detection & Pedestrian Intent Analysis system.
"""

import argparse
import cv2
import time
import logging
from pathlib import Path

from src.lane_detection.detector import LaneDetector
from src.pedestrian_detection.detector import PedestrianDetector
from src.posture_analysis.analyzer import PostureAnalyzer
from src.decision_making.decision_maker import DecisionMaker
from src.utils.visualization import visualize_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Lane and Pedestrian Intent Detection System')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input video file or "cam" for webcam')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (optional)')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with visualization')
    return parser.parse_args()

def process_frame(frame, lane_detector, pedestrian_detector, posture_analyzer, decision_maker):
    """Process a single frame through the pipeline."""
    # Detect lanes
    lanes = lane_detector.detect(frame)
    
    # Detect pedestrians
    pedestrian_boxes = pedestrian_detector.detect(frame)
    
    # Analyze posture for each detected pedestrian
    pedestrian_poses = []
    for box in pedestrian_boxes:
        cropped_pedestrian = frame[box[1]:box[3], box[0]:box[2]]
        pose_data = posture_analyzer.analyze(cropped_pedestrian, box)
        pedestrian_poses.append(pose_data)
    
    # Make decisions based on lanes and pedestrian intent
    decision = decision_maker.decide(lanes, pedestrian_poses)
    
    # Return results
    return {
        'lanes': lanes,
        'pedestrians': pedestrian_boxes,
        'poses': pedestrian_poses,
        'decision': decision
    }

def main():
    """Main function."""
    args = parse_arguments()
    
    # Initialize components
    lane_detector = LaneDetector()
    pedestrian_detector = PedestrianDetector()
    posture_analyzer = PostureAnalyzer()
    decision_maker = DecisionMaker()
    
    # Open video capture
    if args.input.lower() == 'cam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        logger.error(f"Error: Could not open video source {args.input}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    video_writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )
    
    logger.info("Starting processing...")
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = process_frame(
                frame, lane_detector, pedestrian_detector, 
                posture_analyzer, decision_maker
            )
            
            # Visualize results
            visualized_frame = visualize_results(frame.copy(), results)
            
            # Display the frame
            if args.debug:
                cv2.imshow('Lane and Pedestrian Analysis', visualized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write frame to output video
            if video_writer is not None:
                video_writer.write(visualized_frame)
                
            frame_count += 1
            
            # Log progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                logger.info(f"Processed {frame_count} frames at {fps:.2f} FPS")
    
    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()