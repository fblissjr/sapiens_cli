#!/usr/bin/env python
"""
Complete Sapiens Pipeline with Two-Stage Processing
Stage 1: Person Detection
Stage 2: Pose Estimation with Refinement
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass
import time
from collections import defaultdict

# Add local imports
import sys
sys.path.append(str(Path(__file__).parent))
from sapiens_refinement import SapiensKeypointDecoder
from sapiens_constants import GOLIATH_KEYPOINTS, GOLIATH_SKELETON_INFO

# Optional: Import tracking libraries
try:
    from bytetrack import BYTETracker
    HAS_BYTETRACK = True
except ImportError:
    HAS_BYTETRACK = False
    print("ByteTrack not available, using IoU tracker")


@dataclass
class BoundingBox:
    """Person bounding box"""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    track_id: Optional[int] = None
    
    def expand(self, factor: float = 1.25) -> 'BoundingBox':
        """Expand bbox by factor for context"""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        
        new_w = w * factor
        new_h = h * factor
        
        return BoundingBox(
            x1=cx - new_w/2,
            y1=cy - new_h/2,
            x2=cx + new_w/2,
            y2=cy + new_h/2,
            score=self.score,
            track_id=self.track_id
        )
    
    def to_square(self) -> 'BoundingBox':
        """Convert to square bbox maintaining center"""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        
        size = max(w, h)
        
        return BoundingBox(
            x1=cx - size/2,
            y1=cy - size/2,
            x2=cx + size/2,
            y2=cy + size/2,
            score=self.score,
            track_id=self.track_id
        )
    
    def clip_to_image(self, img_w: int, img_h: int) -> 'BoundingBox':
        """Clip bbox to image boundaries"""
        return BoundingBox(
            x1=max(0, self.x1),
            y1=max(0, self.y1),
            x2=min(img_w, self.x2),
            y2=min(img_h, self.y2),
            score=self.score,
            track_id=self.track_id
        )


class MultiPersonTracker:
    """Multi-person tracking across frames"""
    
    def __init__(
        self,
        method: str = 'iou',
        iou_threshold: float = 0.3,
        max_lost: int = 30,
        min_hits: int = 3
    ):
        """
        Args:
            method: Tracking method ('bytetrack', 'iou', 'sort')
            iou_threshold: IoU threshold for matching
            max_lost: Maximum frames to keep lost tracks
            min_hits: Minimum hits to start tracking
        """
        self.method = method
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        
        self.tracks = {}  # track_id -> Track object
        self.next_id = 1
        self.frame_count = 0
        
        if method == 'bytetrack' and HAS_BYTETRACK:
            self.tracker = BYTETracker(
                track_thresh=0.5,
                track_buffer=max_lost,
                match_thresh=iou_threshold
            )
        else:
            self.tracker = None  # Use built-in IoU tracker
    
    def update(self, detections: List[BoundingBox]) -> List[BoundingBox]:
        """
        Update tracks with new detections
        Returns: Detections with assigned track IDs
        """
        self.frame_count += 1
        
        if self.tracker is not None:
            # Use ByteTrack
            return self._update_bytetrack(detections)
        else:
            # Use built-in IoU tracker
            return self._update_iou_tracker(detections)
    
    def _update_iou_tracker(self, detections: List[BoundingBox]) -> List[BoundingBox]:
        """Simple IoU-based tracker"""
        # Get active tracks
        active_tracks = {tid: t for tid, t in self.tracks.items() 
                        if t['lost_frames'] < self.max_lost}
        
        if not active_tracks:
            # No tracks, create new ones
            tracked_dets = []
            for det in detections:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det,
                    'hits': 1,
                    'lost_frames': 0,
                    'keypoints_history': []
                }
                self.next_id += 1
                tracked_dets.append(det)
            return tracked_dets
        
        # Match detections to tracks using IoU
        track_ids = list(active_tracks.keys())
        track_boxes = [active_tracks[tid]['bbox'] for tid in track_ids]
        
        matched_pairs = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        if detections and track_boxes:
            # Calculate IoU matrix
            iou_matrix = self._calculate_iou_matrix(detections, track_boxes)
            
            # Hungarian matching or greedy matching
            for _ in range(min(len(detections), len(track_boxes))):
                if iou_matrix.size == 0:
                    break
                
                # Find best match
                max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_iou_idx]
                
                if max_iou > self.iou_threshold:
                    det_idx, track_idx = max_iou_idx
                    matched_pairs.append((det_idx, track_idx))
                    
                    # Remove from unmatched lists
                    if det_idx in unmatched_dets:
                        unmatched_dets.remove(det_idx)
                    if track_idx in unmatched_tracks:
                        unmatched_tracks.remove(track_idx)
                    
                    # Mask out this match
                    iou_matrix[det_idx, :] = -1
                    iou_matrix[:, track_idx] = -1
                else:
                    break
        
        # Update matched tracks
        tracked_dets = []
        for det_idx, track_idx in matched_pairs:
            track_id = track_ids[track_idx]
            det = detections[det_idx]
            det.track_id = track_id
            
            # Update track
            self.tracks[track_id]['bbox'] = det
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['lost_frames'] = 0
            
            tracked_dets.append(det)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            det.track_id = self.next_id
            self.tracks[self.next_id] = {
                'bbox': det,
                'hits': 1,
                'lost_frames': 0,
                'keypoints_history': []
            }
            self.next_id += 1
            tracked_dets.append(det)
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id]['lost_frames'] += 1
        
        # Clean up old tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() 
                      if t['lost_frames'] < self.max_lost}
        
        return tracked_dets
    
    def _calculate_iou_matrix(
        self, 
        dets1: List[BoundingBox], 
        dets2: List[BoundingBox]
    ) -> np.ndarray:
        """Calculate IoU matrix between two sets of boxes"""
        n1, n2 = len(dets1), len(dets2)
        iou_matrix = np.zeros((n1, n2))
        
        for i, box1 in enumerate(dets1):
            for j, box2 in enumerate(dets2):
                iou_matrix[i, j] = self._calculate_iou(box1, box2)
        
        return iou_matrix
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _update_bytetrack(self, detections: List[BoundingBox]) -> List[BoundingBox]:
        """Update using ByteTrack"""
        # Convert to ByteTrack format
        if detections:
            det_array = np.array([[d.x1, d.y1, d.x2, d.y2, d.score] 
                                 for d in detections])
        else:
            det_array = np.empty((0, 5))
        
        # Run ByteTrack
        online_targets = self.tracker.update(det_array, [1080, 1920], [1080, 1920])
        
        # Assign track IDs
        tracked_dets = []
        for t in online_targets:
            if t.is_activated:
                # Find corresponding detection
                for det in detections:
                    if (abs(det.x1 - t.tlbr[0]) < 1 and 
                        abs(det.y1 - t.tlbr[1]) < 1):
                        det.track_id = t.track_id
                        tracked_dets.append(det)
                        break
        
        return tracked_dets
    
    def get_track_history(self, track_id: int) -> List:
        """Get keypoint history for a track"""
        if track_id in self.tracks:
            return self.tracks[track_id].get('keypoints_history', [])
        return []
    
    def add_keypoints_to_track(self, track_id: int, keypoints: Dict):
        """Add keypoints to track history for smoothing"""
        if track_id in self.tracks:
            history = self.tracks[track_id].get('keypoints_history', [])
            history.append(keypoints)
            # Keep only last N frames
            self.tracks[track_id]['keypoints_history'] = history[-10:]


class PersonDetector:
    """Person detection stage"""
    
    def __init__(self, method: str = 'yolov8', confidence_threshold: float = 0.3):
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.detector = None
        
        if method == 'yolov8':
            self._init_yolov8()
        elif method == 'rtmdet':
            self._init_rtmdet()
        else:
            raise ValueError(f"Unknown detector: {method}")
    
    def _init_yolov8(self):
        """Initialize YOLOv8 detector"""
        try:
            from ultralytics import YOLO
            # Use YOLOv8 medium for balance of speed/accuracy
            self.detector = YOLO('yolov8m.pt')
            print("Initialized YOLOv8 detector")
        except ImportError:
            print("YOLOv8 not available, install with: pip install ultralytics")
            raise
    
    def _init_rtmdet(self):
        """Initialize RTMDet from mmdet"""
        try:
            from mmdet.apis import init_detector, inference_detector
            # RTMDet config (would need proper config file)
            config = 'configs/rtmdet/rtmdet_m_8xb32-300e_coco.py'
            checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
            self.detector = init_detector(config, checkpoint, device='cuda')
            print("Initialized RTMDet detector")
        except ImportError:
            print("MMDetection not available")
            raise
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect people in image"""
        if self.method == 'yolov8':
            return self._detect_yolov8(image)
        elif self.method == 'rtmdet':
            return self._detect_rtmdet(image)
    
    def _detect_yolov8(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect using YOLOv8"""
        results = self.detector(image, conf=self.confidence_threshold, classes=[0])  # class 0 = person
        
        boxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = box.conf[0].cpu().numpy()
                    boxes.append(BoundingBox(x1, y1, x2, y2, score))
        
        return boxes
    
    def _detect_rtmdet(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect using RTMDet"""
        from mmdet.apis import inference_detector
        
        result = inference_detector(self.detector, image)
        pred_instances = result.pred_instances
        
        # Filter for person class (usually index 0)
        person_mask = pred_instances.labels == 0
        person_boxes = pred_instances.bboxes[person_mask]
        person_scores = pred_instances.scores[person_mask]
        
        boxes = []
        for bbox, score in zip(person_boxes, person_scores):
            if score > self.confidence_threshold:
                x1, y1, x2, y2 = bbox.cpu().numpy()
                boxes.append(BoundingBox(x1, y1, x2, y2, score.item()))
        
        return boxes


class SapiensTwoStagePipeline:
    """
    Complete two-stage Sapiens pipeline with tracking:
    1. Detect people
    2. Track across frames
    3. Run pose estimation on crops with refinement
    """
    
    def __init__(
        self,
        pose_model_path: str,
        detector_method: str = 'yolov8',
        tracker_method: str = 'iou',
        det_threshold: float = 0.3,
        kpt_threshold: float = 0.3,
        refinement: str = 'dark_udp',
        blur_kernel_size: int = 11,
        expand_bbox: float = 1.25,
        max_people: int = 20,
        device: str = 'cuda',
        enable_smoothing: bool = True
    ):
        """
        Args:
            pose_model_path: Path to Sapiens pose model
            detector_method: Detection method ('yolov8', 'rtmdet')
            tracker_method: Tracking method ('bytetrack', 'iou')
            det_threshold: Detection confidence threshold
            kpt_threshold: Keypoint confidence threshold
            refinement: Refinement method ('dark_udp', 'dark', 'simple', 'none')
            blur_kernel_size: Gaussian blur kernel size for refinement
            expand_bbox: Factor to expand bounding boxes
            max_people: Maximum number of people to track
            device: Compute device
            enable_smoothing: Enable temporal smoothing
        """
        self.device = device
        self.expand_bbox = expand_bbox
        self.kpt_threshold = kpt_threshold
        self.max_people = max_people
        self.enable_smoothing = enable_smoothing
        
        # Initialize detector
        print("Initializing person detector...")
        self.detector = PersonDetector(detector_method, det_threshold)
        
        # Initialize tracker
        print(f"Initializing {tracker_method} tracker...")
        self.tracker = MultiPersonTracker(
            method=tracker_method,
            iou_threshold=0.3,
            max_lost=30,
            min_hits=3
        )
        
        # Load pose model
        print(f"Loading pose model from {pose_model_path}...")
        self.pose_model = self._load_pose_model(pose_model_path)
        
        # Initialize refinement decoder
        self.refinement = refinement
        self.blur_kernel_size = blur_kernel_size
        
        # Standard Sapiens input size
        self.input_size = (1024, 768)  # W x H
        
        # Performance stats
        self.frame_count = 0
        self.total_people_tracked = 0
    
    def _load_pose_model(self, model_path: str):
        """Load Sapiens pose model"""
        if model_path.endswith('.pt2'):
            # TorchScript model
            model = torch.jit.load(model_path, map_location=self.device)
        else:
            # Regular PyTorch
            model = torch.load(model_path, map_location=self.device)
        
        model.eval()
        return model
    
    def _crop_person(self, image: np.ndarray, bbox: BoundingBox) -> Tuple[np.ndarray, BoundingBox]:
        """
        Crop person from image with padding
        Returns: (cropped_image, adjusted_bbox)
        """
        h, w = image.shape[:2]
        
        # Expand bbox for context
        bbox_exp = bbox.expand(self.expand_bbox)
        
        # Make square
        bbox_sq = bbox_exp.to_square()
        
        # Clip to image
        bbox_clipped = bbox_sq.clip_to_image(w, h)
        
        # Extract crop
        x1, y1, x2, y2 = map(int, [bbox_clipped.x1, bbox_clipped.y1, 
                                   bbox_clipped.x2, bbox_clipped.y2])
        
        crop = image[y1:y2, x1:x2]
        
        # Pad if needed (when bbox extends beyond image)
        if bbox_sq.x1 < 0 or bbox_sq.y1 < 0 or bbox_sq.x2 > w or bbox_sq.y2 > h:
            # Calculate padding
            pad_left = int(max(0, -bbox_sq.x1))
            pad_top = int(max(0, -bbox_sq.y1))
            pad_right = int(max(0, bbox_sq.x2 - w))
            pad_bottom = int(max(0, bbox_sq.y2 - h))
            
            # Apply padding
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)  # Gray padding
            )
            
            # Adjust bbox coordinates
            bbox_clipped = BoundingBox(
                x1 - pad_left,
                y1 - pad_top,
                x2 + pad_right,
                y2 + pad_bottom,
                bbox.score,
                bbox.track_id
            )
        
        return crop, bbox_clipped
    
    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess crop for Sapiens model"""
        # Resize to model input size
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        img = resized.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float()
        img = img[[2, 1, 0], ...]  # BGR -> RGB
        
        # Normalize (Sapiens normalization)
        mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).view(-1, 1, 1)
        std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).view(-1, 1, 1)
        img = (img - mean) / std
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img.to(self.device)
    
    def _run_pose_inference(self, crop_tensor: torch.Tensor) -> np.ndarray:
        """Run pose model on preprocessed crop"""
        with torch.no_grad():
            if self.device == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = self.pose_model(crop_tensor)
            else:
                output = self.pose_model(crop_tensor)
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            heatmaps = output[0]
        else:
            heatmaps = output
        
        # Convert to numpy
        if heatmaps.dtype == torch.bfloat16:
            heatmaps = heatmaps.float()
        
        return heatmaps[0].cpu().numpy()  # Remove batch dim
    
    def _decode_and_refine(
        self, 
        heatmaps: np.ndarray, 
        crop_size: Tuple[int, int],
        original_bbox: BoundingBox
    ) -> Dict:
        """Decode heatmaps with refinement and transform to image coordinates"""
        
        # Initialize decoder
        H, W = heatmaps.shape[1:]
        decoder = SapiensKeypointDecoder(
            input_size=crop_size,
            heatmap_size=(W, H),
            refinement=self.refinement,
            blur_kernel_size=self.blur_kernel_size,
            confidence_threshold=self.kpt_threshold
        )
        
        # Decode with refinement
        keypoints, scores = decoder.decode(heatmaps)
        
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
            scores = scores[0]
        
        # Transform to original image coordinates
        scale_x = (original_bbox.x2 - original_bbox.x1) / crop_size[0]
        scale_y = (original_bbox.y2 - original_bbox.y1) / crop_size[1]
        
        keypoints[:, 0] = keypoints[:, 0] * scale_x + original_bbox.x1
        keypoints[:, 1] = keypoints[:, 1] * scale_y + original_bbox.y1
        
        # Convert to dictionary
        kpt_dict = {}
        for i, (kpt, score) in enumerate(zip(keypoints, scores)):
            if i < len(GOLIATH_KEYPOINTS) and score > self.kpt_threshold:
                kpt_dict[GOLIATH_KEYPOINTS[i]] = {
                    'x': float(kpt[0]),
                    'y': float(kpt[1]),
                    'confidence': float(score)
                }
        
        return kpt_dict
    
    def process_frame(self, image: np.ndarray) -> List[Dict]:
        """
        Process single frame through complete pipeline with tracking
        
        Returns:
            List of person dictionaries with keypoints and track IDs
        """
        h, w = image.shape[:2]
        self.frame_count += 1
        
        # Stage 1: Detect people
        detections = self.detector.detect(image)
        
        # Apply NMS to remove duplicate detections
        detections = self._nms_boxes(detections)
        
        # Limit to max_people
        if len(detections) > self.max_people:
            # Keep highest scoring detections
            detections = sorted(detections, key=lambda x: x.score, reverse=True)[:self.max_people]
        
        # Stage 2: Update tracking
        tracked_detections = self.tracker.update(detections)
        
        results = []
        
        # Stage 3: Process each tracked person
        for bbox in tracked_detections:
            # Crop person with padding
            crop, crop_bbox = self._crop_person(image, bbox)
            crop_h, crop_w = crop.shape[:2]
            
            # Preprocess crop
            crop_tensor = self._preprocess_crop(crop)
            
            # Run inference
            heatmaps = self._run_pose_inference(crop_tensor)
            
            # Decode and refine keypoints
            keypoints = self._decode_and_refine(
                heatmaps, 
                (crop_w, crop_h),
                bbox  # Original bbox in image coords
            )
            
            # Filter by confidence/visibility
            keypoints = self._filter_keypoints(keypoints)
            
            # Apply temporal smoothing if enabled
            if self.enable_smoothing and bbox.track_id is not None:
                keypoints = self._smooth_keypoints(keypoints, bbox.track_id)
            
            # Store in tracker history
            if bbox.track_id is not None:
                self.tracker.add_keypoints_to_track(bbox.track_id, keypoints)
            
            results.append({
                'bbox': {
                    'x1': bbox.x1,
                    'y1': bbox.y1,
                    'x2': bbox.x2,
                    'y2': bbox.y2,
                    'score': bbox.score
                },
                'track_id': bbox.track_id,
                'keypoints': keypoints,
                'num_keypoints': len(keypoints)
            })
        
        self.total_people_tracked = len(set(r['track_id'] for r in results if r['track_id'] is not None))
        
        return results
    
    def _smooth_keypoints(self, keypoints: Dict, track_id: int, window: int = 5) -> Dict:
        """Apply temporal smoothing to keypoints using track history"""
        history = self.tracker.get_track_history(track_id)
        
        if len(history) < 2:
            return keypoints
        
        # Get recent history (up to window size)
        recent_history = history[-window+1:] if len(history) >= window-1 else history
        recent_history.append(keypoints)
        
        # Smooth each keypoint
        smoothed = {}
        for kpt_name in keypoints:
            # Collect historical positions
            x_vals = []
            y_vals = []
            conf_vals = []
            
            for frame_kpts in recent_history:
                if kpt_name in frame_kpts:
                    x_vals.append(frame_kpts[kpt_name]['x'])
                    y_vals.append(frame_kpts[kpt_name]['y'])
                    conf_vals.append(frame_kpts[kpt_name]['confidence'])
            
            if x_vals:
                # Apply weighted average (more recent = higher weight)
                weights = np.linspace(0.5, 1.0, len(x_vals))
                weights = weights / weights.sum()
                
                smoothed[kpt_name] = {
                    'x': float(np.average(x_vals, weights=weights)),
                    'y': float(np.average(y_vals, weights=weights)),
                    'confidence': float(np.mean(conf_vals))
                }
        
        return smoothed
    
    def _nms_boxes(self, boxes: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
        """Apply NMS to bounding boxes"""
        if not boxes:
            return boxes
        
        # Convert to numpy array
        boxes_array = np.array([[b.x1, b.y1, b.x2, b.y2, b.score] for b in boxes])
        
        # Sort by score
        indices = np.argsort(boxes_array[:, 4])[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the first one
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            rest_indices = indices[1:]
            ious = self._calculate_iou(boxes_array[i], boxes_array[rest_indices])
            
            # Keep only boxes with IoU less than threshold
            indices = rest_indices[ious < iou_threshold]
        
        return [boxes[i] for i in keep]
    
    def _calculate_iou(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and multiple boxes"""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = area1 + areas - intersection
        
        return intersection / (union + 1e-6)
    
    def _filter_keypoints(self, keypoints: Dict, min_visible: int = 5) -> Dict:
        """Filter keypoints by various criteria"""
        # Could implement:
        # - Minimum visible keypoints
        # - Bone length consistency
        # - Temporal smoothing
        # - Outlier removal
        
        if len(keypoints) < min_visible:
            return {}
        
        return keypoints
    
    def visualize(self, image: np.ndarray, results: List[Dict], 
                  show_track_ids: bool = True, show_bbox: bool = True) -> np.ndarray:
        """Visualize results on image with track IDs"""
        img = image.copy()
        
        # Use different colors for different tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
            (128, 0, 255), (255, 0, 128), (0, 128, 255), (0, 255, 128),
            (128, 128, 255), (128, 255, 128), (255, 128, 128)
        ]
        
        for person in results:
            track_id = person.get('track_id')
            color = colors[track_id % len(colors)] if track_id else (0, 255, 0)
            
            # Draw bbox
            if show_bbox:
                bbox = person['bbox']
                cv2.rectangle(
                    img,
                    (int(bbox['x1']), int(bbox['y1'])),
                    (int(bbox['x2']), int(bbox['y2'])),
                    color, 2
                )
                
                # Draw track ID
                if show_track_ids and track_id is not None:
                    label = f"ID: {track_id} ({bbox['score']:.2f})"
                    label_y = max(int(bbox['y1']) - 10, 20)
                    cv2.putText(
                        img, label,
                        (int(bbox['x1']), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )
            
            # Draw keypoints and skeleton
            keypoints = person['keypoints']
            
            # Draw skeleton connections
            for link_info in GOLIATH_SKELETON_INFO.values():
                pt1_name, pt2_name = link_info['link']
                color = link_info['color'][::-1]  # RGB to BGR
                
                if pt1_name in keypoints and pt2_name in keypoints:
                    pt1 = keypoints[pt1_name]
                    pt2 = keypoints[pt2_name]
                    
                    cv2.line(
                        img,
                        (int(pt1['x']), int(pt1['y'])),
                        (int(pt2['x']), int(pt2['y'])),
                        color, 2
                    )
            
            # Draw keypoints
            for kpt_name, kpt_data in keypoints.items():
                cv2.circle(
                    img,
                    (int(kpt_data['x']), int(kpt_data['y'])),
                    3, (0, 0, 255), -1
                )
        
        return img


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Sapiens Two-Stage Pipeline")
    parser.add_argument('input', help='Input video or image')
    parser.add_argument('output', help='Output path')
    parser.add_argument('--model', required=True, help='Pose model path')
    parser.add_argument('--detector', default='yolov8', 
                       choices=['yolov8', 'rtmdet'],
                       help='Detector method')
    parser.add_argument('--refinement', default='dark_udp',
                       choices=['dark_udp', 'dark', 'simple', 'none'],
                       help='Refinement method')
    parser.add_argument('--max-frames', type=int, help='Max frames to process')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SapiensTwoStagePipeline(
        pose_model_path=args.model,
        detector_method=args.detector,
        refinement=args.refinement
    )
    
    # Process video
    cap = cv2.VideoCapture(args.input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.max_frames and frame_count >= args.max_frames:
            break
        
        # Process frame
        results = pipeline.process_frame(frame)
        
        # Visualize
        vis_frame = pipeline.visualize(frame, results)
        out.write(vis_frame)
        
        frame_count += 1
        print(f"Processed frame {frame_count}")
    
    cap.release()
    out.release()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()