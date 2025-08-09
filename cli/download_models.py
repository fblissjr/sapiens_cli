#!/usr/bin/env python
"""
Flexible Sapiens model downloader
Choose models, formats, and sizes interactively or via command line
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
import json

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
    from huggingface_hub import HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed")
    print("Install with: pip install huggingface_hub")


# Model catalog with all available options
MODEL_CATALOG = {
    'pose': {
        'repo_pattern': 'facebook/sapiens-pose-{size}{format_suffix}',
        'sizes': ['0.3b', '0.6b', '1b', '2b'],
        'formats': {
            'pytorch': '',
            'torchscript': '-torchscript',
            'bfloat16': '-bfloat16'
        },
        'keypoint_sets': {
            'goliath': 'goliath_best_goliath_AP',  # 308 keypoints
            'coco_wholebody': 'coco_wholebody_best_coco_wholebody_AP',  # 133 keypoints
            'coco': 'coco_best_coco_AP'  # 17 keypoints
        }
    },
    'seg': {
        'repo_pattern': 'facebook/sapiens-seg-{size}{format_suffix}',
        'sizes': ['0.3b', '0.6b', '1b', '2b'],
        'formats': {
            'pytorch': '',
            'torchscript': '-torchscript',
            'bfloat16': '-bfloat16'
        }
    },
    'depth': {
        'repo_pattern': 'facebook/sapiens-depth-{size}{format_suffix}',
        'sizes': ['0.3b', '0.6b', '1b', '2b'],
        'formats': {
            'pytorch': '',
            'torchscript': '-torchscript',
            'bfloat16': '-bfloat16'
        }
    },
    'normal': {
        'repo_pattern': 'facebook/sapiens-normal-{size}{format_suffix}',
        'sizes': ['0.3b', '0.6b', '1b', '2b'],
        'formats': {
            'pytorch': '',
            'torchscript': '-torchscript',
            'bfloat16': '-bfloat16'
        }
    }
}

# File patterns for different formats
FILE_PATTERNS = {
    'pytorch': '.pth',
    'torchscript': '.pt2',
    'bfloat16': '_bfloat16.pt'
}


class ModelDownloader:
    def __init__(self, base_dir='checkpoints'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.api = HfApi() if HF_AVAILABLE else None
        
    def list_available_models(self):
        """List all available models on HuggingFace"""
        print("\n" + "="*60)
        print("AVAILABLE SAPIENS MODELS")
        print("="*60)
        
        for task, info in MODEL_CATALOG.items():
            print(f"\n{task.upper()}:")
            print(f"  Sizes: {', '.join(info['sizes'])}")
            print(f"  Formats: {', '.join(info['formats'].keys())}")
            
            if task == 'pose':
                print(f"  Keypoint sets: {', '.join(info['keypoint_sets'].keys())}")
    
    def check_model_exists(self, repo_id):
        """Check if a model repository exists"""
        if not self.api:
            return False
        
        try:
            files = list_repo_files(repo_id)
            return len(files) > 0
        except:
            return False
    
    def download_model(self, task, size, format_type, keypoint_set=None):
        """Download a specific model"""
        if task not in MODEL_CATALOG:
            print(f"Error: Unknown task '{task}'")
            return False
        
        info = MODEL_CATALOG[task]
        
        # Build repo ID
        format_suffix = info['formats'].get(format_type, '')
        repo_id = info['repo_pattern'].format(size=size, format_suffix=format_suffix)
        
        # Check if exists
        if not self.check_model_exists(repo_id):
            print(f"  ⚠ Model not found: {repo_id}")
            # Try alternative format
            if format_type == 'bfloat16':
                print(f"    Note: BF16 models may not be available for all sizes")
            return False
        
        # Determine local directory
        local_dir = self.base_dir / task / format_type / size
        
        print(f"\nDownloading: {repo_id}")
        print(f"  To: {local_dir}")
        
        try:
            # For pose models with specific keypoint sets
            if task == 'pose' and keypoint_set:
                # Download specific checkpoint
                pattern = info['keypoint_sets'].get(keypoint_set, 'goliath')
                files = list_repo_files(repo_id)
                
                # Find matching file
                target_files = [f for f in files if pattern in f]
                
                if target_files:
                    for file in target_files:
                        print(f"  Downloading: {file}")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=file,
                            local_dir=local_dir,
                            local_dir_use_symlinks=False
                        )
                else:
                    # Download all if specific not found
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        ignore_patterns=["*.md", "*.txt"]
                    )
            else:
                # Download entire repository
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.md", "*.txt"]
                )
            
            print(f"  ✓ Downloaded successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            return False
    
    def download_batch(self, selections):
        """Download multiple models based on selections"""
        success = []
        failed = []
        
        for selection in selections:
            task = selection['task']
            size = selection['size']
            format_type = selection['format']
            keypoint_set = selection.get('keypoint_set')
            
            if self.download_model(task, size, format_type, keypoint_set):
                success.append(f"{task}-{size}-{format_type}")
            else:
                failed.append(f"{task}-{size}-{format_type}")
        
        return success, failed
    
    def interactive_mode(self):
        """Interactive model selection"""
        selections = []
        
        print("\nINTERACTIVE MODEL SELECTION")
        print("="*60)
        
        # Select tasks
        print("\nWhich tasks do you need? (comma-separated or 'all')")
        print("Options: pose, seg, depth, normal")
        task_input = input("Tasks: ").strip()
        
        if task_input.lower() == 'all':
            tasks = list(MODEL_CATALOG.keys())
        else:
            tasks = [t.strip() for t in task_input.split(',')]
        
        # Select format
        print("\nWhich format do you want?")
        print("1. pytorch (original, most flexible)")
        print("2. torchscript (optimized, faster)")
        print("3. bfloat16 (memory efficient, requires Ampere+)")
        print("4. all formats")
        format_choice = input("Choice (1-4): ").strip()
        
        format_map = {
            '1': ['pytorch'],
            '2': ['torchscript'],
            '3': ['bfloat16'],
            '4': ['pytorch', 'torchscript', 'bfloat16']
        }
        formats = format_map.get(format_choice, ['pytorch'])
        
        # Select size
        print("\nWhich model size?")
        print("1. 0.3b (fastest, lowest quality)")
        print("2. 0.6b (fast, good quality)")
        print("3. 1b (balanced, recommended)")
        print("4. 2b (slowest, best quality)")
        print("5. all sizes")
        size_choice = input("Choice (1-5): ").strip()
        
        size_map = {
            '1': ['0.3b'],
            '2': ['0.6b'],
            '3': ['1b'],
            '4': ['2b'],
            '5': ['0.3b', '0.6b', '1b', '2b']
        }
        sizes = size_map.get(size_choice, ['1b'])
        
        # Special handling for pose keypoints
        keypoint_set = None
        if 'pose' in tasks:
            print("\nFor pose estimation, which keypoint set?")
            print("1. goliath (308 keypoints - most detailed)")
            print("2. coco_wholebody (133 keypoints - face+body+hands)")
            print("3. coco (17 keypoints - basic body)")
            kp_choice = input("Choice (1-3): ").strip()
            
            kp_map = {
                '1': 'goliath',
                '2': 'coco_wholebody',
                '3': 'coco'
            }
            keypoint_set = kp_map.get(kp_choice, 'goliath')
        
        # Build selection list
        for task in tasks:
            if task not in MODEL_CATALOG:
                print(f"Warning: Unknown task '{task}', skipping")
                continue
                
            for format_type in formats:
                for size in sizes:
                    selections.append({
                        'task': task,
                        'size': size,
                        'format': format_type,
                        'keypoint_set': keypoint_set if task == 'pose' else None
                    })
        
        # Confirm
        print(f"\nAbout to download {len(selections)} models:")
        for sel in selections[:5]:  # Show first 5
            print(f"  - {sel['task']}-{sel['size']}-{sel['format']}")
        if len(selections) > 5:
            print(f"  ... and {len(selections)-5} more")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return
        
        # Download
        success, failed = self.download_batch(selections)
        
        # Report
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Successfully downloaded: {len(success)} models")
        if failed:
            print(f"Failed: {len(failed)} models")
            for f in failed:
                print(f"  - {f}")
    
    def create_config_file(self, selections, output_path='model_config.json'):
        """Create a configuration file for downloaded models"""
        config = {
            'base_dir': str(self.base_dir),
            'models': {}
        }
        
        for sel in selections:
            task = sel['task']
            size = sel['size']
            format_type = sel['format']
            
            key = f"{task}_{size}_{format_type}"
            local_dir = self.base_dir / task / format_type / size
            
            config['models'][key] = {
                'task': task,
                'size': size,
                'format': format_type,
                'path': str(local_dir),
                'keypoint_set': sel.get('keypoint_set')
            }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Created config file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Flexible Sapiens model downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python download_models_flexible.py
  
  # Download specific models
  python download_models_flexible.py --tasks pose seg --sizes 1b --formats torchscript
  
  # Download all BF16 models
  python download_models_flexible.py --tasks all --formats bfloat16
  
  # Download best models for production
  python download_models_flexible.py --tasks pose depth seg --sizes 1b --formats torchscript bfloat16
        """
    )
    
    parser.add_argument('--tasks', nargs='+', 
                       choices=['pose', 'seg', 'depth', 'normal', 'all'],
                       help='Tasks to download')
    parser.add_argument('--sizes', nargs='+',
                       choices=['0.3b', '0.6b', '1b', '2b', 'all'],
                       help='Model sizes')
    parser.add_argument('--formats', nargs='+',
                       choices=['pytorch', 'torchscript', 'bfloat16', 'all'],
                       help='Model formats')
    parser.add_argument('--keypoints', 
                       choices=['goliath', 'coco_wholebody', 'coco'],
                       default='goliath',
                       help='Keypoint set for pose models')
    parser.add_argument('--base-dir', default='checkpoints',
                       help='Base directory for downloads')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    parser.add_argument('--config', help='Save model paths to config file')
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("Error: huggingface_hub is required")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    
    downloader = ModelDownloader(args.base_dir)
    
    # List mode
    if args.list:
        downloader.list_available_models()
        sys.exit(0)
    
    # Interactive mode if no arguments
    if not args.tasks:
        downloader.interactive_mode()
    else:
        # Command line mode
        # Process 'all' keywords
        tasks = list(MODEL_CATALOG.keys()) if 'all' in args.tasks else args.tasks
        sizes = ['0.3b', '0.6b', '1b', '2b'] if args.sizes and 'all' in args.sizes else (args.sizes or ['1b'])
        formats = ['pytorch', 'torchscript', 'bfloat16'] if args.formats and 'all' in args.formats else (args.formats or ['pytorch'])
        
        # Build selections
        selections = []
        for task in tasks:
            for size in sizes:
                for format_type in formats:
                    selections.append({
                        'task': task,
                        'size': size,
                        'format': format_type,
                        'keypoint_set': args.keypoints if task == 'pose' else None
                    })
        
        print(f"Downloading {len(selections)} models...")
        success, failed = downloader.download_batch(selections)
        
        # Save config if requested
        if args.config:
            downloader.create_config_file(selections, args.config)
        
        # Summary
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"✓ Successfully downloaded: {len(success)}")
        if failed:
            print(f"✗ Failed: {len(failed)}")
        
        print(f"\nModels saved to: {args.base_dir}/")
        print("\nDirectory structure:")
        print(f"  {args.base_dir}/")
        print(f"    ├── pose/")
        print(f"    │   ├── pytorch/")
        print(f"    │   ├── torchscript/")
        print(f"    │   └── bfloat16/")
        print(f"    ├── seg/")
        print(f"    ├── depth/")
        print(f"    └── normal/")


if __name__ == "__main__":
    main()