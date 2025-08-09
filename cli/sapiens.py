#!/usr/bin/env python
"""
Sapiens CLI - Human pose, depth, segmentation, and normal estimation
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sapiens pipeline for human understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download models
  python sapiens.py download --tasks pose --sizes 1b --formats torchscript
  
  # Process video with full pipeline (detection + tracking + pose)
  python sapiens.py process video.mp4 output/ --model pose_model.pt2
  
  # Simple inference without detection (single person)
  python sapiens.py infer video.mp4 output.mp4 --model pose_model.pt2
  
  # Benchmark model
  python sapiens.py benchmark --model pose_model.pt2
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download Sapiens models')
    download_parser.add_argument('--tasks', nargs='+', 
                                 choices=['pose', 'seg', 'depth', 'normal', 'all'],
                                 help='Tasks to download')
    download_parser.add_argument('--sizes', nargs='+',
                                 choices=['0.3b', '0.6b', '1b', '2b', 'all'],
                                 help='Model sizes')
    download_parser.add_argument('--formats', nargs='+',
                                 choices=['pytorch', 'torchscript', 'all'],
                                 help='Model formats')
    
    # Process command (full pipeline)
    process_parser = subparsers.add_parser('process', 
                                           help='Process with full pipeline (detection + tracking + pose)')
    process_parser.add_argument('input', help='Input video or image')
    process_parser.add_argument('output', help='Output directory')
    process_parser.add_argument('--model', required=True, help='Pose model path')
    process_parser.add_argument('--detector', default='yolov8',
                               choices=['yolov8', 'rtmdet'],
                               help='Person detector')
    process_parser.add_argument('--tracker', default='iou',
                               choices=['iou', 'bytetrack'],
                               help='Tracking method')
    process_parser.add_argument('--max-people', type=int, default=20,
                               help='Maximum people to track')
    process_parser.add_argument('--refinement', default='dark_udp',
                               choices=['dark_udp', 'dark', 'simple', 'none'],
                               help='Keypoint refinement method')
    
    # Infer command (simple, no detection)
    infer_parser = subparsers.add_parser('infer',
                                         help='Simple inference without detection')
    infer_parser.add_argument('input', help='Input video or image')
    infer_parser.add_argument('output', help='Output path')
    infer_parser.add_argument('--model', required=True, help='Model path')
    infer_parser.add_argument('--max-frames', type=int, help='Max frames to process')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    bench_parser.add_argument('--model', required=True, help='Model path')
    bench_parser.add_argument('--iterations', type=int, default=100,
                             help='Number of iterations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'download':
        from download_models import main as download_main
        sys.argv = ['download_models.py']
        if args.tasks:
            sys.argv.extend(['--tasks'] + args.tasks)
        if args.sizes:
            sys.argv.extend(['--sizes'] + args.sizes)
        if args.formats:
            sys.argv.extend(['--formats'] + args.formats)
        return download_main()
    
    elif args.command == 'process':
        from sapiens_full_pipeline import main as pipeline_main
        sys.argv = ['sapiens_full_pipeline.py', args.input, args.output,
                   '--model', args.model,
                   '--detector', args.detector,
                   '--tracker', args.tracker,
                   '--max-people', str(args.max_people),
                   '--refinement', args.refinement]
        return pipeline_main()
    
    elif args.command == 'infer':
        from sapiens_inference import main as infer_main
        sys.argv = ['sapiens_inference.py', args.input, args.output,
                   '--model', args.model]
        if args.max_frames:
            sys.argv.extend(['--max-frames', str(args.max_frames)])
        return infer_main()
    
    elif args.command == 'benchmark':
        from sapiens_inference import main as bench_main
        sys.argv = ['sapiens_inference.py', '--benchmark',
                   '--model', args.model,
                   '--benchmark-iterations', str(args.iterations)]
        return bench_main()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())