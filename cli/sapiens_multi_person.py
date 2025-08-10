#!/usr/bin/env python
"""
Complete Multi-Person Sapiens Pipeline
- YOLO detection
- DeepSORT tracking  
- Pose (308 keypoints) + Segmentation (28 body parts)
- Features only output (no overlay on original video)
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.insert(0, '/home/fbliss/workspace/deep_sort')

# Import local modules
from sapiens_inference import SapiensModelLoader, OptimizedInference, PoseProcessor
from sapiens_segmentation import SegmentationInference, BODY_PART_PALETTE
from sapiens_constants import GOLIATH_KEYPOINTS, GOLIATH_SKELETON_INFO

# Import YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("YOLO not available, install with: pip install ultralytics")

# Import DeepSORT
try:
    from deep_sort.tracker import Tracker as DeepSORTTracker
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    HAS_DEEPSORT = True
except ImportError:
    HAS_DEEPSORT = False
    print("DeepSORT not available")


@dataclass
class TrackedPerson:
    """Tracked person with bbox and ID"""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    keypoints: Optional[Dict] = None
    segmentation: Optional[np.ndarray] = None


class MultiPersonSapiens:
    """Complete multi-person pipeline"""
    
    def __init__(self, 
                 pose_model_path: str,
                 seg_model_path: Optional[str] = None,
                 detector: str = 'yolov8m',
                 use_tracking: bool = True):
        """
        Args:
            pose_model_path: Path to pose model
            seg_model_path: Optional path to segmentation model
            detector: YOLO model name (yolov8m, yolov8l, etc.)
            use_tracking: Whether to use DeepSORT tracking
        """
        
        print("Initializing Multi-Person Sapiens Pipeline...")
        
        # Load pose model
        print("Loading pose model...")
        loader = SapiensModelLoader()
        pose_model, format_type, metadata = loader.load_model(pose_model_path)
        self.pose_inference = OptimizedInference(pose_model, format_type, metadata)
        self.pose_processor = PoseProcessor()
        
        # Load segmentation model if provided
        self.seg_inference = None
        if seg_model_path and Path(seg_model_path).exists():
            print("Loading segmentation model...")
            self.seg_inference = SegmentationInference(seg_model_path)
        
        # Initialize YOLO detector
        if HAS_YOLO:
            print(f"Loading YOLO detector: {detector}")
            model_path = f"{detector}.pt" if not detector.endswith('.pt') else detector
            self.detector = YOLO(model_path)
        else:
            raise RuntimeError("YOLO not available")
        
        # Initialize tracker
        self.tracker = None
        if use_tracking and HAS_DEEPSORT:
            print("Initializing DeepSORT tracker...")
            max_cosine_distance = 0.3
            nn_budget = 100
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", max_cosine_distance, nn_budget
            )
            self.tracker = DeepSORTTracker(metric, max_age=30, n_init=3)
    
    def detect_people(self, frame: np.ndarray) -> List[TrackedPerson]:
        """Detect people in frame using YOLO"""
        results = self.detector(frame, classes=[0])  # class 0 is person
        
        people = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    
                    # Filter low confidence
                    if conf > 0.5:
                        people.append(TrackedPerson(
                            track_id=-1,  # Will be assigned by tracker
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=conf
                        ))
        
        return people
    
    def update_tracking(self, people: List[TrackedPerson], frame: np.ndarray) -> List[TrackedPerson]:
        """Update tracking IDs using DeepSORT"""
        if not self.tracker or not people:
            # No tracking, just assign sequential IDs
            for i, person in enumerate(people):
                person.track_id = i
            return people
        
        # Convert to DeepSORT format
        detections = []
        for person in people:
            bbox = person.bbox
            # Convert to [x, y, w, h] format
            bbox_xywh = [
                bbox[0], bbox[1],
                bbox[2] - bbox[0], bbox[3] - bbox[1]
            ]
            # Simple feature (could be improved with ReID network)
            feature = np.random.rand(128)
            detections.append(Detection(bbox_xywh, person.confidence, feature))
        
        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections)
        
        # Match tracks to detections
        tracked_people = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            tracked_people.append(TrackedPerson(
                track_id=track.track_id,
                bbox=bbox.tolist(),
                confidence=1.0
            ))
        
        return tracked_people
    
    def process_person(self, frame: np.ndarray, person: TrackedPerson):
        """Process pose and segmentation for one person"""
        x1, y1, x2, y2 = [int(x) for x in person.bbox]
        
        # Crop person with padding
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return
        
        # Process pose
        pose_input = self.pose_inference.preprocess(person_crop)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pose_output = self.pose_inference.infer(pose_input)
        
        if pose_output.dim() == 4:
            pose_output = pose_output[0]
        
        # Decode keypoints relative to crop
        crop_bbox = [0, 0, x2-x1, y2-y1]
        keypoints = self.pose_processor.decode_heatmaps(pose_output, crop_bbox)
        
        # Adjust keypoints to full frame coordinates
        if keypoints:
            for kpt_name, kpt_data in keypoints.items():
                kpt_data['x'] += x1
                kpt_data['y'] += y1
        
        person.keypoints = keypoints
        
        # Process segmentation if available
        if self.seg_inference:
            seg_result = self.seg_inference.process_frame(person_crop, overlay=False)
            person.segmentation = seg_result
    
    def visualize_frame(self, frame: np.ndarray, people: List[TrackedPerson], 
                       show_seg: bool = True) -> np.ndarray:
        """Create visualization with features only (no original video)"""
        h, w = frame.shape[:2]
        
        # Create black canvas
        canvas = np.zeros_like(frame)
        
        # Draw each person
        for person in people:
            x1, y1, x2, y2 = [int(x) for x in person.bbox]
            
            # Removed bounding box and ID drawing - only show features
            
            # Draw pose skeleton
            if person.keypoints:
                self._draw_skeleton(canvas, person.keypoints)
            
            # Draw segmentation if available
            if show_seg and person.segmentation is not None:
                # Blend segmentation into region
                seg = person.segmentation
                # Ensure segmentation matches the bbox size
                bbox_h = y2 - y1
                bbox_w = x2 - x1
                if seg.shape[:2] != (bbox_h, bbox_w):
                    seg = cv2.resize(seg, (bbox_w, bbox_h))
                
                # Blend only if sizes match
                if canvas[y1:y2, x1:x2].shape == seg.shape:
                    canvas[y1:y2, x1:x2] = cv2.addWeighted(
                        canvas[y1:y2, x1:x2], 0.3, seg, 0.7, 0
                    )
        
        return canvas
    
    def _draw_skeleton(self, img: np.ndarray, keypoints: Dict):
        """Draw pose skeleton on image"""
        # Draw connections
        for link_info in GOLIATH_SKELETON_INFO.values():
            pt1_name, pt2_name = link_info['link']
            color = link_info['color'][::-1]  # RGB to BGR
            
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                
                if pt1['confidence'] > 0.3 and pt2['confidence'] > 0.3:
                    cv2.line(img,
                            (int(pt1['x']), int(pt1['y'])),
                            (int(pt2['x']), int(pt2['y'])),
                            color, 2)
        
        # Draw keypoints
        for kpt_name, kpt_data in keypoints.items():
            if kpt_data['confidence'] > 0.3:
                cv2.circle(img, 
                          (int(kpt_data['x']), int(kpt_data['y'])),
                          3, (0, 255, 0), -1)
    
    def _get_track_color(self, track_id: int) -> tuple:
        """Get consistent color for track ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (128, 0, 255)
        ]
        return colors[track_id % len(colors)]
    
    def process_video(self, video_path: str, output_path: str, 
                     max_frames: Optional[int] = None,
                     show_seg: bool = True):
        """Process full video with tracking and visualization"""
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"\nProcessing video: {width}x{height} @ {fps} FPS")
        print(f"Frames to process: {total_frames}")
        
        # Create output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Store tracking data
        tracking_data = []
        
        pbar = tqdm(total=total_frames, desc="Processing frames")
        frame_idx = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people
            people = self.detect_people(frame)
            
            # Update tracking
            if self.tracker:
                people = self.update_tracking(people, frame)
            
            # Process each person
            for person in people:
                self.process_person(frame, person)
            
            # Create visualization
            vis_frame = self.visualize_frame(frame, people, show_seg)
            
            # Save frame data
            frame_data = {
                'frame': frame_idx,
                'people': []
            }
            for person in people:
                person_data = {
                    'track_id': person.track_id,
                    'bbox': person.bbox,
                    'confidence': person.confidence
                }
                if person.keypoints:
                    person_data['keypoints'] = person.keypoints
                frame_data['people'].append(person_data)
            
            tracking_data.append(frame_data)
            
            # Write frame
            out.write(vis_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        # Save tracking data
        json_path = output_path.replace('.mp4', '_tracking.json')
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"\nOutput saved to: {output_path}")
        print(f"Tracking data saved to: {json_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-person Sapiens pipeline')
    parser.add_argument('input', help='Input video')
    parser.add_argument('output', help='Output video')
    parser.add_argument('--pose-model', required=True, help='Pose model path')
    parser.add_argument('--seg-model', help='Segmentation model path (optional)')
    parser.add_argument('--detector', default='yolov8m', help='YOLO model')
    parser.add_argument('--no-tracking', action='store_true', help='Disable tracking')
    parser.add_argument('--max-frames', type=int, help='Max frames to process')
    parser.add_argument('--no-seg', action='store_true', help='Disable segmentation visualization')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MultiPersonSapiens(
        pose_model_path=args.pose_model,
        seg_model_path=args.seg_model,
        detector=args.detector,
        use_tracking=not args.no_tracking
    )
    
    # Process video
    pipeline.process_video(
        args.input, 
        args.output,
        max_frames=args.max_frames,
        show_seg=not args.no_seg and args.seg_model is not None
    )


if __name__ == '__main__':
    main()