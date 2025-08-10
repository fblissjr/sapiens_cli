#!/usr/bin/env python
"""
Sapiens Segmentation Inference - Colored body part visualization
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

# Goliath body part segmentation palette (28 classes)
# Based on official Sapiens segmentation model
BODY_PART_PALETTE = [
    [0, 0, 0],          # 0: Background
    [255, 128, 0],      # 1: Head  
    [255, 153, 51],     # 2: Torso
    [102, 178, 255],    # 3: Right Upper Arm
    [51, 153, 255],     # 4: Right Forearm
    [255, 51, 51],      # 5: Right Hand
    [230, 230, 0],      # 6: Left Upper Arm  
    [255, 153, 255],    # 7: Left Forearm
    [153, 255, 153],    # 8: Left Hand
    [255, 102, 255],    # 9: Right Thigh
    [255, 51, 255],     # 10: Right Calf
    [102, 255, 102],    # 11: Right Foot
    [51, 255, 51],      # 12: Left Thigh
    [26, 128, 0],       # 13: Left Calf
    [0, 255, 0],        # 14: Left Foot
    [0, 0, 255],        # 15: Right Upper Leg Clothing
    [255, 0, 0],        # 16: Left Upper Leg Clothing
    [102, 204, 0],      # 17: Right Lower Leg Clothing
    [204, 0, 102],      # 18: Left Lower Leg Clothing
    [0, 153, 153],      # 19: Upper Body Clothing
    [0, 102, 204],      # 20: Left Shoe
    [204, 102, 0],      # 21: Right Shoe
    [76, 153, 0],       # 22: Hair
    [102, 0, 204],      # 23: Face
    [255, 255, 0],      # 24: Left Sock
    [0, 255, 255],      # 25: Right Sock
    [255, 0, 255],      # 26: Gloves
    [192, 192, 192],    # 27: Other Accessories
]

BODY_PART_NAMES = [
    "Background", "Head", "Torso", "Right Upper Arm", "Right Forearm",
    "Right Hand", "Left Upper Arm", "Left Forearm", "Left Hand",
    "Right Thigh", "Right Calf", "Right Foot", "Left Thigh", 
    "Left Calf", "Left Foot", "Right Upper Leg Clothing", "Left Upper Leg Clothing",
    "Right Lower Leg Clothing", "Left Lower Leg Clothing", "Upper Body Clothing",
    "Left Shoe", "Right Shoe", "Hair", "Face", "Left Sock", "Right Sock",
    "Gloves", "Other Accessories"
]


class SegmentationInference:
    def __init__(self, model_path: str):
        """Initialize segmentation model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading segmentation model from {model_path}")
        if model_path.endswith('.pt2'):
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        self.model.eval()
        
        # Model expects 1024x768 input
        self.input_size = (768, 1024)  # (H, W)
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model"""
        # Resize to model input size
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def visualize_segmentation(self, seg_mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to colored visualization"""
        h, w = seg_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for idx in range(len(BODY_PART_PALETTE)):
            mask = seg_mask == idx
            if mask.any():
                colored[mask] = BODY_PART_PALETTE[idx]
        
        return colored
    
    def process_frame(self, frame: np.ndarray, overlay: bool = False) -> np.ndarray:
        """Process single frame"""
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Run inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = self.model(input_tensor)
        
        # Get segmentation mask
        if isinstance(output, dict):
            output = output['segmentation']
        
        # Convert to class predictions
        seg_logits = output[0]  # Remove batch dimension
        seg_mask = torch.argmax(seg_logits, dim=0).cpu().numpy()
        
        # Resize to original size
        seg_mask_resized = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Create colored visualization
        colored_seg = self.visualize_segmentation(seg_mask_resized)
        
        if overlay:
            # Blend with original image
            alpha = 0.5
            result = cv2.addWeighted(frame, 1-alpha, colored_seg, alpha, 0)
        else:
            # Return only colored segmentation
            result = colored_seg
        
        return result


def process_video(video_path: str, output_path: str, model_path: str, 
                  max_frames: Optional[int] = None, overlay: bool = False):
    """Process video with segmentation"""
    
    # Initialize model
    seg_model = SegmentationInference(model_path)
    
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
    print(f"Output mode: {'Overlay' if overlay else 'Features only'}")
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_count = 0
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = seg_model.process_frame(frame, overlay=overlay)
        
        # Write output
        out.write(result)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\nOutput saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Sapiens segmentation inference')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--model', required=True, help='Path to segmentation model')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--overlay', action='store_true', 
                       help='Overlay segmentation on original video')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return
    
    # Process video
    process_video(args.input, args.output, args.model, args.max_frames, args.overlay)


if __name__ == '__main__':
    main()