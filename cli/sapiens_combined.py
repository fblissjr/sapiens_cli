#!/usr/bin/env python
"""
Combined Sapiens Pose + Segmentation Inference
Outputs both skeleton and colored body parts together
"""

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import time
from tqdm import tqdm
import argparse
from typing import Optional
import json

# Import from existing modules
from sapiens_inference import SapiensModelLoader, OptimizedInference, PoseProcessor
from sapiens_segmentation import SegmentationInference, BODY_PART_PALETTE

def process_video_combined(video_path: str, output_path: str, 
                          pose_model_path: str, seg_model_path: str,
                          max_frames: Optional[int] = None, 
                          mode: str = 'side_by_side'):
    """
    Process video with both pose and segmentation
    
    Args:
        video_path: Input video
        output_path: Output video
        pose_model_path: Path to pose model
        seg_model_path: Path to segmentation model
        max_frames: Max frames to process
        mode: 'side_by_side', 'overlay', or 'blend'
    """
    
    print("\nLoading models...")
    
    # Load pose model
    loader = SapiensModelLoader()
    pose_model, format_type, metadata = loader.load_model(pose_model_path)
    pose_inference = OptimizedInference(pose_model, format_type, metadata)
    pose_processor = PoseProcessor()
    
    # Load segmentation model
    seg_inference = SegmentationInference(seg_model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"\nProcessing video: {width}x{height} @ {fps} FPS")
    print(f"Frames to process: {total_frames}")
    print(f"Output mode: {mode}")
    
    # Setup output based on mode
    if mode == 'side_by_side':
        out_width = width * 2
        out_height = height
    else:
        out_width = width
        out_height = height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_count = 0
    json_output = []
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        bbox = [0, 0, w, h]
        
        # POSE PROCESSING
        pose_input = pose_inference.preprocess(frame)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pose_output = pose_inference.infer(pose_input)
        
        if pose_output.dim() == 4:
            pose_output = pose_output[0]
        keypoints = pose_processor.decode_heatmaps(pose_output, bbox)
        
        # Draw pose on black background
        pose_viz = pose_processor.draw_pose(frame, keypoints, overlay=False)
        
        # SEGMENTATION PROCESSING
        seg_viz = seg_inference.process_frame(frame, overlay=False)
        
        # Combine outputs based on mode
        if mode == 'side_by_side':
            # Put pose on left, segmentation on right
            combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
            combined[:, :width] = pose_viz
            combined[:, width:] = seg_viz
            
            # Add labels
            cv2.putText(combined, "POSE (308 keypoints)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "SEGMENTATION (28 parts)", (width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif mode == 'overlay':
            # Overlay both on original frame
            pose_on_frame = pose_processor.draw_pose(frame, keypoints, overlay=True)
            seg_on_frame = seg_inference.process_frame(frame, overlay=True)
            combined = cv2.addWeighted(pose_on_frame, 0.5, seg_on_frame, 0.5, 0)
            
        elif mode == 'blend':
            # Blend pose and segmentation together
            combined = cv2.addWeighted(pose_viz, 0.5, seg_viz, 0.5, 0)
        
        # Store keypoints for JSON output
        if keypoints:
            json_output.append({
                'frame': frame_count,
                'keypoints': keypoints,
                'num_keypoints': len(keypoints)
            })
        
        out.write(combined)
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Save keypoints JSON
    json_path = output_path.replace('.mp4', '_keypoints.json')
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\nOutput saved to: {output_path}")
    print(f"Keypoints saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Combined Sapiens pose + segmentation')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--pose-model', required=True, help='Path to pose model')
    parser.add_argument('--seg-model', required=True, help='Path to segmentation model')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--mode', choices=['side_by_side', 'overlay', 'blend'],
                       default='side_by_side',
                       help='How to combine outputs (default: side_by_side)')
    
    args = parser.parse_args()
    
    # Check models exist
    if not Path(args.pose_model).exists():
        print(f"Error: Pose model not found: {args.pose_model}")
        return
    if not Path(args.seg_model).exists():
        print(f"Error: Segmentation model not found: {args.seg_model}")
        return
    
    process_video_combined(args.input, args.output, 
                          args.pose_model, args.seg_model,
                          args.max_frames, args.mode)


if __name__ == '__main__':
    main()