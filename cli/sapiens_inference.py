#!/usr/bin/env python
"""
Unified test script for Sapiens models - handles TorchScript and native PyTorch
Optimized for RTX 4090 with proper BF16/FP16 support
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import sys
from typing import Optional, Dict, List, Tuple

# Add local imports
sys.path.append(str(Path(__file__).parent))
from sapiens_constants import GOLIATH_KEYPOINTS, GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS


class SapiensModelLoader:
    """Smart model loader that handles different formats"""
    
    @staticmethod
    def load_model(model_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, str, Dict]:
        """
        Load model intelligently based on file type and content
        Returns: (model, format_type, metadata)
        """
        model_path = Path(model_path)
        metadata = {
            'path': str(model_path),
            'size': model_path.stat().st_size / 1e9,  # GB
        }
        
        print(f"Loading model: {model_path.name}")
        print(f"  Size: {metadata['size']:.2f} GB")
        
        # Determine model type
        if model_path.suffix == '.pt2':
            print("  Format: .pt2 file - detecting type...")
            
            # Check if it's an ExportedProgram
            import zipfile
            is_exported_program = False
            try:
                with zipfile.ZipFile(str(model_path), 'r') as zf:
                    if 'serialized_exported_program.json' in zf.namelist():
                        is_exported_program = True
            except:
                pass
            
            if is_exported_program:
                # This is an ExportedProgram format
                print("  Type: ExportedProgram (torch.export format)")
                print("  Loading as ExportedProgram...")
                
                try:
                    # Load the exported program
                    from torch import export as torch_export
                    exported_program = torch_export.load(str(model_path))
                    
                    # Create a wrapper module
                    class ExportedProgramWrapper(torch.nn.Module):
                        def __init__(self, exported_program):
                            super().__init__()
                            self.exported_program = exported_program
                        
                        def forward(self, x):
                            # Call the exported program
                            return self.exported_program(x)
                    
                    model = ExportedProgramWrapper(exported_program).to(device)
                    format_type = 'exported_program'
                    
                    if 'bfloat16' in str(model_path).lower():
                        metadata['exported_dtype'] = 'bfloat16'
                        format_type = 'exported_program_bf16'
                    
                    print("  ✓ Loaded as ExportedProgram")
                    
                except Exception as e:
                    print(f"  Failed to load as ExportedProgram: {e}")
                    print("  Falling back to regular loading...")
                    model = torch.load(str(model_path), map_location=device)
                    format_type = 'pytorch'
            else:
                # Not an ExportedProgram, try TorchScript
                print("  Type: TorchScript (.pt2)")
                try:
                    model = torch.jit.load(str(model_path), map_location=device)
                    format_type = 'torchscript'
                    metadata['original_format'] = 'torchscript'
                    
                    # Check if it's actually a BF16 export
                    if 'bfloat16' in str(model_path).lower() or 'bf16' in str(model_path).lower():
                        print("  Note: This is a BF16-exported TorchScript model")
                        metadata['exported_dtype'] = 'bfloat16'
                        format_type = 'torchscript_bf16'
                    
                    print("  ✓ Loaded as TorchScript")
                        
                except Exception as e:
                    print(f"  Failed to load as TorchScript: {e}")
                    print("  Attempting to load as regular PyTorch...")
                    model = torch.load(str(model_path), map_location=device)
                    format_type = 'pytorch'
                
        elif model_path.suffix in ['.pth', '.pt']:
            # Regular PyTorch model
            print("  Format: PyTorch (.pth/.pt)")
            checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Need to create model and load state dict
                    # This would require knowing the architecture
                    raise NotImplementedError("State dict loading requires model architecture")
                else:
                    model = checkpoint
            else:
                model = checkpoint
                
            format_type = 'pytorch'
            metadata['original_format'] = 'pytorch'
            
        else:
            raise ValueError(f"Unknown model format: {model_path.suffix}")
        
        # Move to device
        if hasattr(model, 'to'):
            model = model.to(device)
        
        # Set eval mode
        model.eval()
        
        # Detect precision capabilities
        if device == 'cuda' and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            metadata['cuda_capability'] = f"{capability[0]}.{capability[1]}"
            metadata['native_bf16'] = capability[0] >= 8
            metadata['native_fp16'] = capability[0] >= 7
        
        print(f"  ✓ Model loaded successfully")
        return model, format_type, metadata


class OptimizedInference:
    """Optimized inference for different model formats and hardware"""
    
    def __init__(self, model, format_type: str, metadata: Dict):
        self.model = model
        self.format_type = format_type
        self.metadata = metadata
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda')
        
        # Determine optimal precision
        self.precision = self._determine_precision()
        print(f"  Using precision: {self.precision}")
        
    def _determine_precision(self) -> str:
        """Determine optimal precision based on hardware and model"""
        if not torch.cuda.is_available():
            return 'fp32'
        
        # For TorchScript BF16 exports
        if 'bf16' in self.format_type:
            if self.metadata.get('native_bf16'):
                return 'bf16'
            else:
                # Fall back to FP16 if no native BF16
                return 'fp16'
        
        # For regular models
        if self.metadata.get('native_bf16'):
            return 'bf16'
        elif self.metadata.get('native_fp16'):
            return 'fp16'
        else:
            return 'fp32'
    
    def preprocess(self, image: np.ndarray, size: Tuple[int, int] = (1024, 768)) -> torch.Tensor:
        """Preprocess image for Sapiens models"""
        # Resize
        img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float()
        img = img[[2, 1, 0], ...]  # BGR -> RGB
        
        # Normalize (Sapiens standard)
        mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).view(-1, 1, 1)
        std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).view(-1, 1, 1)
        img = (img - mean) / std
        
        return img
    
    @torch.no_grad()
    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run optimized inference based on format and precision"""
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Add batch dimension if needed
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Run inference with appropriate precision
        if self.precision == 'bf16' and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = self.model(input_tensor)
        elif self.precision == 'fp16' and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = self.model(input_tensor)
        else:
            output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        return output


class PoseProcessor:
    """Process pose estimation results with proper refinement"""
    
    def __init__(self, keypoint_threshold: float = 0.3, 
                 refinement: str = 'dark_udp',
                 blur_kernel_size: int = 11):
        self.keypoint_threshold = keypoint_threshold
        self.refinement = refinement
        self.blur_kernel_size = blur_kernel_size
        
        # Import refinement functions
        try:
            from sapiens_refinement import SapiensKeypointDecoder
            self.use_refinement = True
            self.decoder = None  # Will initialize per image
        except ImportError:
            print("Warning: sapiens_refinement not found, using basic decoding")
            self.use_refinement = False
        
    def decode_heatmaps(self, heatmaps: torch.Tensor, bbox: List[int]) -> Dict:
        """Decode keypoints from heatmaps with refinement"""
        x1, y1, x2, y2 = bbox
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Convert to float32 first if BFloat16 (numpy doesn't support BFloat16)
        if heatmaps.dtype == torch.bfloat16:
            heatmaps = heatmaps.float()
        
        heatmaps_np = heatmaps.cpu().numpy()
        
        if self.use_refinement:
            # Use proper Sapiens refinement
            from sapiens_refinement import SapiensKeypointDecoder
            
            # Initialize decoder for this crop size
            H, W = heatmaps_np.shape[1:]
            decoder = SapiensKeypointDecoder(
                input_size=(crop_width, crop_height),
                heatmap_size=(W, H),
                refinement=self.refinement,
                blur_kernel_size=self.blur_kernel_size,
                confidence_threshold=self.keypoint_threshold
            )
            
            # Decode with refinement
            keypoints_array, scores = decoder.decode(heatmaps_np, bbox=(x1, y1, x2, y2))
            
            # Convert to dictionary format
            keypoints = {}
            if keypoints_array.shape[0] == 1:
                keypoints_array = keypoints_array[0]
                scores = scores[0]
            
            for i in range(min(len(keypoints_array), len(GOLIATH_KEYPOINTS))):
                if scores[i] > self.keypoint_threshold:
                    keypoints[GOLIATH_KEYPOINTS[i]] = {
                        'x': float(keypoints_array[i, 0]),
                        'y': float(keypoints_array[i, 1]),
                        'confidence': float(scores[i])
                    }
        else:
            # Fallback to simple decoding
            keypoints = {}
            for i, heatmap in enumerate(heatmaps_np):
                if i >= len(GOLIATH_KEYPOINTS):
                    break
                
                # Find peak
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                confidence = float(heatmap[y, x])
                
                if confidence > self.keypoint_threshold:
                    # Scale to original coordinates
                    kpt_x = x * crop_width / heatmap.shape[1] + x1
                    kpt_y = y * crop_height / heatmap.shape[0] + y1
                    
                    keypoints[GOLIATH_KEYPOINTS[i]] = {
                        'x': float(kpt_x),
                        'y': float(kpt_y),
                        'confidence': confidence
                    }
        
        return keypoints
    
    def draw_pose(self, image: np.ndarray, keypoints: Dict, overlay: bool = False) -> np.ndarray:
        """Draw pose skeleton on image or black background
        
        Args:
            image: Input image
            keypoints: Detected keypoints
            overlay: If True, draw on original image. If False, draw on black background.
        """
        if overlay:
            img = image.copy()
        else:
            # Create black background of same size
            img = np.zeros_like(image)
        
        # Draw skeleton connections
        for link_info in GOLIATH_SKELETON_INFO.values():
            pt1_name, pt2_name = link_info['link']
            color = link_info['color'][::-1]  # RGB to BGR
            
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                
                if pt1['confidence'] > self.keypoint_threshold and pt2['confidence'] > self.keypoint_threshold:
                    cv2.line(img,
                            (int(pt1['x']), int(pt1['y'])),
                            (int(pt2['x']), int(pt2['y'])),
                            color, 2)
        
        # Draw keypoints
        for i, (kpt_name, kpt_data) in enumerate(keypoints.items()):
            if i < len(GOLIATH_KPTS_COLORS) and kpt_data['confidence'] > self.keypoint_threshold:
                color = GOLIATH_KPTS_COLORS[i][::-1]  # RGB to BGR
                cv2.circle(img,
                          (int(kpt_data['x']), int(kpt_data['y'])),
                          3, color, -1)
        
        return img


def process_video(video_path: str, output_path: str, model_path: str, 
                  max_frames: Optional[int] = None, overlay: bool = False):
    """Process video with pose estimation"""
    
    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    loader = SapiensModelLoader()
    model, format_type, metadata = loader.load_model(model_path)
    
    # Create inference engine
    inference = OptimizedInference(model, format_type, metadata)
    
    # Create pose processor
    pose_processor = PoseProcessor()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print("\n" + "="*60)
    print("PROCESSING VIDEO")
    print("="*60)
    print(f"Input: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print(f"Frames to process: {total_frames}")
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process timing
    inference_times = []
    json_output = []
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing frames")
    frame_count = 0
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use full frame as person bbox
        h, w = frame.shape[:2]
        bbox = [0, 0, w, h]
        
        # Preprocess
        input_tensor = inference.preprocess(frame)
        
        # Run inference
        start_time = time.time()
        output = inference.infer(input_tensor)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Decode keypoints
        if output.dim() == 4:  # Remove batch dimension
            output = output[0]
        keypoints = pose_processor.decode_heatmaps(output, bbox)
        
        # Store results
        json_output.append({
            'frame': frame_count,
            'keypoints': keypoints,
            'num_keypoints': len(keypoints),
            'inference_time': inference_time
        })
        
        # Draw and save
        if keypoints:
            frame = pose_processor.draw_pose(frame, keypoints, overlay=overlay)
        else:
            # If no keypoints, write black frame or original based on overlay setting
            if not overlay:
                frame = np.zeros_like(frame)
        out.write(frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Save JSON
    json_path = output_path.replace('.mp4', '_keypoints.json')
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    # Print statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if inference_times:
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print(f"Model format: {format_type}")
        print(f"Precision used: {inference.precision}")
        print(f"Native BF16: {metadata.get('native_bf16', False)}")
        print(f"\nInference Performance:")
        print(f"  Average: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Min: {min_time*1000:.2f} ms")
        print(f"  Max: {max_time*1000:.2f} ms")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"\nTotal:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total time: {sum(inference_times):.1f}s")
        print(f"  Output video: {output_path}")
        print(f"  Keypoints JSON: {json_path}")


def process_image(image_path: str, output_path: str, model_path: str):
    """Process single image with pose estimation"""
    
    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    loader = SapiensModelLoader()
    model, format_type, metadata = loader.load_model(model_path)
    
    # Create inference engine
    inference = OptimizedInference(model, format_type, metadata)
    
    # Create pose processor
    pose_processor = PoseProcessor()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"\nProcessing image: {w}x{h}")
    
    # Use full image as bbox
    bbox = [0, 0, w, h]
    
    # Preprocess
    input_tensor = inference.preprocess(image)
    
    # Run inference
    start_time = time.time()
    output = inference.infer(input_tensor)
    inference_time = time.time() - start_time
    
    # Decode keypoints
    if output.dim() == 4:
        output = output[0]
    keypoints = pose_processor.decode_heatmaps(output, bbox)
    
    print(f"  Detected {len(keypoints)} keypoints")
    print(f"  Inference time: {inference_time*1000:.2f} ms")
    
    # Draw and save
    if keypoints:
        image = pose_processor.draw_pose(image, keypoints)
    
    cv2.imwrite(output_path, image)
    print(f"\n✓ Saved to: {output_path}")
    
    # Save keypoints
    json_path = output_path.replace('.jpg', '.json').replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            'image': image_path,
            'keypoints': keypoints,
            'num_keypoints': len(keypoints),
            'inference_time': inference_time
        }, f, indent=2)
    print(f"✓ Keypoints saved to: {json_path}")


def benchmark_model(model_path: str, num_iterations: int = 100):
    """Benchmark model performance"""
    
    print("\n" + "="*60)
    print("BENCHMARKING MODEL")
    print("="*60)
    
    # Load model
    loader = SapiensModelLoader()
    model, format_type, metadata = loader.load_model(model_path)
    
    # Create inference engine
    inference = OptimizedInference(model, format_type, metadata)
    
    # Create dummy input
    batch_sizes = [1, 2, 4, 8]
    # Sapiens models expect 1024x768 (H x W) input
    input_sizes = [(1024, 768)]
    
    results = {}
    
    for size in input_sizes:
        for batch_size in batch_sizes:
            # Skip large batches for large inputs
            if size[0] * size[1] * batch_size > 8000000:  # Memory limit
                continue
            
            print(f"\nTesting {size[0]}x{size[1]} with batch_size={batch_size}")
            
            # Create dummy input
            dummy_input = torch.randn(batch_size, 3, size[0], size[1])
            
            # Warmup
            for _ in range(10):
                _ = inference.infer(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(num_iterations):
                _ = inference.infer(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            total_time = time.time() - start
            avg_time = total_time / num_iterations
            throughput = batch_size / avg_time
            
            key = f"{size[0]}x{size[1]}_batch{batch_size}"
            results[key] = {
                'avg_time_ms': avg_time * 1000,
                'throughput_fps': throughput,
                'batch_size': batch_size
            }
            
            print(f"  Time: {avg_time*1000:.2f} ms")
            print(f"  Throughput: {throughput:.1f} images/sec")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {Path(model_path).name}")
    print(f"Format: {format_type}")
    print(f"Precision: {inference.precision}")
    
    print("\nResults:")
    for key, result in results.items():
        print(f"  {key}:")
        print(f"    Time: {result['avg_time_ms']:.2f} ms")
        print(f"    Throughput: {result['throughput_fps']:.1f} fps")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Sapiens Model Tester (TorchScript/PyTorch/BF16)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with automatic model detection
  python test_sapiens_unified.py video.mp4 output.mp4 --model checkpoints/pose/1b/model.pt2
  
  # Process image
  python test_sapiens_unified.py image.jpg output.jpg --model checkpoints/pose/1b/model.pth
  
  # Benchmark model
  python test_sapiens_unified.py --benchmark --model checkpoints/pose/1b/model.pt2
  
  # Process with frame limit (for testing)
  python test_sapiens_unified.py video.mp4 output.mp4 --model model.pt2 --max-frames 100
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input video or image path')
    parser.add_argument('output', nargs='?', help='Output path')
    parser.add_argument('--model', required=True, help='Path to model file (.pt2, .pth, .pt)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (for testing)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
    parser.add_argument('--benchmark-iterations', type=int, default=100, 
                       help='Number of iterations for benchmark')
    parser.add_argument('--overlay', action='store_true', 
                       help='Overlay pose on original video (default: features only on black background)')
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("="*60)
        print("SYSTEM INFO")
        print("="*60)
        print(f"GPU: {gpu_name}")
        print(f"Compute capability: {capability[0]}.{capability[1]}")
        print(f"Memory: {memory_gb:.1f} GB")
        
        if capability[0] >= 8:
            print("✓ Native BF16 support (Ampere or newer)")
        elif capability[0] >= 7:
            print("⚠ BF16 emulation available (Volta/Turing)")
        else:
            print("✗ No BF16 support")
    else:
        print("⚠ No GPU detected, using CPU (slow)")
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"\nError: Model not found: {args.model}")
        print("\nDownload models using:")
        print("  python download_models_flexible.py --tasks pose --sizes 1b --formats torchscript bfloat16")
        return
    
    # Run appropriate mode
    if args.benchmark:
        benchmark_model(args.model, args.benchmark_iterations)
    elif args.input and args.output:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input not found: {args.input}")
            return
        
        # Process based on input type
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            process_video(args.input, args.output, args.model, args.max_frames, args.overlay)
        else:
            process_image(args.input, args.output, args.model)
    else:
        print("Error: Please provide input and output paths, or use --benchmark")
        parser.print_help()


if __name__ == "__main__":
    main()