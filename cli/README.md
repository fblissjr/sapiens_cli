# Sapiens CLI - Multi-Person Video Pipeline

Complete pipeline for extracting human pose (308 keypoints) and segmentation (28 body parts) from videos with multi-person tracking.

## What's Working ✅

- **Multi-person detection** using YOLO
- **Multi-person tracking** using DeepSORT (handles moving cameras)
- **308 keypoint pose estimation** (full Goliath skeleton with fingers, face details)
- **28 body part segmentation** (colored body regions)
- **Features-only output** (no original video overlay)
- **TorchScript model support** (fastest inference)
- **BF16 optimization** for RTX 30/40 series GPUs

## What's NOT Working ❌

- **ByteTrack** - Installation issues, but DeepSORT works better anyway
- **RTMDet detector** - Requires mmdet setup, YOLO works fine
- **Depth/Normal models** - Not integrated yet (models available but no pipeline)

## Installation

```bash
cd /home/fbliss/workspace/sapiens_cli/cli
pip install -r requirements.txt
```

## Quick Start

### 1. Download Models

```bash
# Download all models
python sapiens.py --download all --size 1b

# Or download specific models
python sapiens.py --download pose seg --size 1b
```

### 2. Process Video - One Command, All Options

```bash
# Everything: multi-person + tracking + pose + segmentation
python sapiens.py input.mp4 output.mp4 \
  --pose checkpoints/pose/torchscript/1b/*.pt2 \
  --seg checkpoints/seg/torchscript/1b/*.pt2 \
  --multi --track

# Single person, pose only (simplest)
python sapiens.py input.mp4 output.mp4 \
  --pose checkpoints/pose/torchscript/1b/*.pt2

# Test on 5 frames first
python sapiens.py input.mp4 test.mp4 \
  --pose checkpoints/pose/torchscript/1b/*.pt2 \
  --seg checkpoints/seg/torchscript/1b/*.pt2 \
  --multi --track --max-frames 5
```

This gives you:
- Automatic detection of all people in frame
- Tracking with consistent IDs across frames
- 308 keypoint skeleton per person
- 28 colored body parts per person
- Features drawn on black background (no original video)

## Usage Examples - All with One Command!

### Single Person (default, fastest)
```bash
# Pose only
python sapiens.py input.mp4 output.mp4 --pose checkpoints/pose/torchscript/1b/*.pt2

# Segmentation only  
python sapiens.py input.mp4 output.mp4 --seg checkpoints/seg/torchscript/1b/*.pt2

# Both pose and segmentation
python sapiens.py input.mp4 output.mp4 --pose model.pt2 --seg model.pt2
```

### Multi-Person with Tracking
```bash
# Multi-person pose with tracking
python sapiens.py input.mp4 output.mp4 --pose model.pt2 --multi --track

# Multi-person pose + segmentation with tracking
python sapiens.py input.mp4 output.mp4 --pose model.pt2 --seg model.pt2 --multi --track
```

### Display Options
```bash
# Overlay on original video (instead of black background)
python sapiens.py input.mp4 output.mp4 --pose model.pt2 --overlay

# Side-by-side pose and segmentation
python sapiens.py input.mp4 output.mp4 --pose model.pt2 --seg model.pt2 --side-by-side
```

### Processing Options
```bash
# Test on just 5 frames
python sapiens.py input.mp4 test.mp4 --pose model.pt2 --max-frames 5

# Disable keypoint refinement (faster but less accurate)
python sapiens.py input.mp4 output.mp4 --pose model.pt2 --refinement none
```

## Output Format

The pipeline generates:
- **Video file**: Visualization with features only on black background
- **JSON file**: Frame-by-frame tracking data with:
  - Person bounding boxes
  - Track IDs (consistent across frames)
  - 308 keypoints per person (x, y, confidence)

## File Structure

```
cli/
├── sapiens.py                 # ⭐ UNIFIED CLI - Use this for everything!
├── sapiens_multi_person.py    # Multi-person pipeline (called by sapiens.py)
├── sapiens_inference.py       # Single-person pipeline (called by sapiens.py)
├── sapiens_segmentation.py    # Segmentation pipeline (called by sapiens.py)
├── sapiens_combined.py        # Combined visualization (called by sapiens.py)
├── sapiens_constants.py       # Keypoint definitions (308 points)
├── sapiens_refinement.py      # Sub-pixel keypoint refinement
├── download_models.py         # Model downloader
├── requirements.txt           # Dependencies
├── checkpoints/              # Downloaded models go here
│   ├── pose/
│   └── seg/
└── archive/                  # Old/unused code
```

## Performance

On RTX 4090 with 1B models:
- Single person: ~40 FPS
- 5 people with tracking: ~15 FPS
- 10 people with tracking: ~8 FPS

## Troubleshooting

### "YOLO not available"
```bash
pip install ultralytics
```

### "DeepSORT not available"
DeepSORT is already installed at `/home/fbliss/workspace/deep_sort`

### Out of Memory
- Use `--max-frames` to process fewer frames
- Or use smaller 0.6B models instead of 1B

### Want Original Video Overlay?
Add `--overlay` flag to `sapiens_inference.py` or `sapiens_segmentation.py`

## What Each Script Does

- **sapiens_multi_person.py**: Complete pipeline with detection, tracking, pose, and segmentation
- **sapiens_inference.py**: Simple pose estimation without detection (single person or full frame)
- **sapiens_segmentation.py**: Body part segmentation only
- **sapiens_combined.py**: Run pose and segmentation together with visualization options
- **download_models.py**: Download Sapiens models from HuggingFace

## Credits

Based on Meta's Sapiens: https://github.com/facebookresearch/sapiens