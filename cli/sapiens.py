#!/usr/bin/env python
"""
Unified Sapiens CLI - All pipelines in one command
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Sapiens - Unified pipeline for human pose and segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single person, pose only
  sapiens input.mp4 output.mp4 --pose checkpoints/pose/torchscript/1b/*.pt2
  
  # Multi-person with tracking, pose + segmentation
  sapiens input.mp4 output.mp4 --pose model.pt2 --seg model.pt2 --multi --track
  
  # Test on 5 frames with all features
  sapiens input.mp4 test.mp4 --pose model.pt2 --seg model.pt2 --multi --track --max-frames 5
  
  # Features on original video (overlay)
  sapiens input.mp4 output.mp4 --pose model.pt2 --overlay
  
  # Download models
  sapiens --download pose seg --size 1b
        """
    )
    
    # Main arguments
    parser.add_argument('input', nargs='?', help='Input video/image path')
    parser.add_argument('output', nargs='?', help='Output video/image path')
    
    # Model selection
    parser.add_argument('--pose', help='Path to pose model (.pt2)')
    parser.add_argument('--seg', help='Path to segmentation model (.pt2)')
    
    # Pipeline options
    parser.add_argument('--multi', action='store_true',
                       help='Enable multi-person detection (default: single person)')
    parser.add_argument('--track', action='store_true',
                       help='Enable tracking (requires --multi)')
    parser.add_argument('--overlay', action='store_true',
                       help='Overlay on original video (default: black background)')
    parser.add_argument('--refinement', choices=['dark_udp', 'dark', 'none'],
                       default='dark_udp',
                       help='Keypoint refinement method')
    
    # Processing options
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process')
    parser.add_argument('--detector', default='yolov8m',
                       help='Detector model for multi-person (default: yolov8m)')
    parser.add_argument('--side-by-side', action='store_true',
                       help='Show pose and segmentation side by side')
    
    # Download command
    parser.add_argument('--download', nargs='+',
                       choices=['pose', 'seg', 'depth', 'normal', 'all'],
                       help='Download models')
    parser.add_argument('--size', default='1b',
                       choices=['0.3b', '0.6b', '1b', '2b'],
                       help='Model size for download')
    
    # Benchmark
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark mode')
    
    args = parser.parse_args()
    
    # Handle download
    if args.download:
        from download_models import main as download_main
        sys.argv = ['download_models.py']
        
        # Map 'all' to individual tasks
        tasks = args.download
        if 'all' in tasks:
            tasks = ['pose', 'seg', 'depth', 'normal']
        
        sys.argv.extend(['--tasks'] + tasks)
        sys.argv.extend(['--sizes', args.size])
        sys.argv.extend(['--formats', 'torchscript'])  # Always use torchscript
        return download_main()
    
    # Check input/output provided
    if not args.input or not args.output:
        print("Error: Please provide input and output paths")
        parser.print_help()
        return 1
    
    # Check model provided
    if not args.pose and not args.seg:
        print("Error: Please provide at least one model (--pose or --seg)")
        return 1
    
    # Route to appropriate pipeline
    if args.multi:
        # Multi-person pipeline
        if not args.track:
            print("Note: Multi-person mode works best with tracking (add --track)")
        
        from sapiens_multi_person import MultiPersonSapiens
        
        pipeline = MultiPersonSapiens(
            pose_model_path=args.pose if args.pose else None,
            seg_model_path=args.seg if args.seg else None,
            detector=args.detector,
            use_tracking=args.track
        )
        
        # Note: overlay not supported in multi-person mode yet
        if args.overlay:
            print("Warning: Overlay not supported in multi-person mode, using black background")
        
        pipeline.process_video(
            args.input,
            args.output,
            max_frames=args.max_frames,
            show_seg=(args.seg is not None)
        )
        
    elif args.side_by_side and args.pose and args.seg:
        # Side-by-side visualization
        from sapiens_combined import process_video_combined
        
        process_video_combined(
            args.input,
            args.output,
            args.pose,
            args.seg,
            max_frames=args.max_frames,
            mode='side_by_side'
        )
        
    elif args.pose and not args.seg:
        # Pose only
        from sapiens_inference import process_video, process_image
        from pathlib import Path
        
        input_path = Path(args.input)
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            process_video(args.input, args.output, args.pose, 
                         args.max_frames, args.overlay)
        else:
            process_image(args.input, args.output, args.pose)
            
    elif args.seg and not args.pose:
        # Segmentation only
        from sapiens_segmentation import process_video
        
        process_video(args.input, args.output, args.seg,
                     args.max_frames, args.overlay)
                     
    else:
        # Both pose and seg but not side-by-side
        from sapiens_combined import process_video_combined
        
        mode = 'overlay' if args.overlay else 'blend'
        process_video_combined(
            args.input,
            args.output,
            args.pose,
            args.seg,
            max_frames=args.max_frames,
            mode=mode
        )
    
    print(f"\nProcessing complete! Output saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())