# Sapiens CLI

Command-line interface for Sapiens human pose estimation with proper two-stage processing, tracking, and refinement.

## Features

- Two-stage processing: detection then pose estimation
- Multi-person tracking (20+ people)
- Sub-pixel keypoint refinement (Dark/UDP)
- Temporal smoothing
- 308 Goliath keypoints
- Support for TorchScript and PyTorch models
- RTX 4090 optimized with BF16 support

## Installation

```bash
cd cli
pip install -r requirements.txt
```

## Quick Start

### 1. Download Models

```bash
# Download TorchScript pose model (recommended)
python sapiens.py download --tasks pose --sizes 1b --formats torchscript

# Download all models
python sapiens.py download --tasks all --sizes 1b --formats torchscript
```

### 2. Process Video (Full Pipeline)

Full pipeline with person detection, tracking, and pose estimation:

```bash
python sapiens.py process input.mp4 output_dir/ \
    --model checkpoints/pose/torchscript/1b/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 \
    --detector yolov8 \
    --tracker iou \
    --max-people 20
```

### 3. Simple Inference

For single person or when you don't need detection:

```bash
python sapiens.py infer input.mp4 output.mp4 \
    --model checkpoints/pose/torchscript/1b/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2
```

### 4. Benchmark

Test model performance:

```bash
python sapiens.py benchmark --model path/to/model.pt2 --iterations 100
```

## Pipeline Architecture

The complete pipeline follows these stages:

1. **Detection**: Detect all people using YOLOv8 or RTMDet
2. **Tracking**: Track people across frames using IoU or ByteTrack
3. **Cropping**: Extract person crops with padding
4. **Inference**: Run Sapiens model on each crop
5. **Refinement**: Apply Dark/UDP refinement for sub-pixel accuracy
6. **Filtering**: Filter by confidence and visibility
7. **Smoothing**: Apply temporal smoothing for stability
8. **Visualization**: Draw skeleton with track IDs

## File Structure

```
cli/
├── sapiens.py              # Main CLI entry point
├── sapiens_full_pipeline.py   # Complete two-stage pipeline
├── sapiens_inference.py    # Simple inference without detection
├── sapiens_refinement.py   # Keypoint refinement (Dark/UDP)
├── sapiens_constants.py    # Keypoint definitions
├── download_models.py      # Model downloader
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Advanced Usage

### Using Different Detectors

```bash
# YOLOv8 (default, faster)
python sapiens.py process video.mp4 output/ --model model.pt2 --detector yolov8

# RTMDet (more accurate, requires mmdet)
python sapiens.py process video.mp4 output/ --model model.pt2 --detector rtmdet
```

### Refinement Methods

```bash
# Dark UDP (default, most accurate)
python sapiens.py process video.mp4 output/ --model model.pt2 --refinement dark_udp

# Dark Pose
python sapiens.py process video.mp4 output/ --model model.pt2 --refinement dark

# No refinement (fastest)
python sapiens.py process video.mp4 output/ --model model.pt2 --refinement none
```

### Processing Options

```bash
# Limit number of people
python sapiens.py process video.mp4 output/ --model model.pt2 --max-people 10

# Process specific frames
python sapiens.py infer video.mp4 output.mp4 --model model.pt2 --max-frames 100
```

## Model Formats

| Format | Extension | Usage |
|--------|-----------|-------|
| TorchScript | .pt2 | Production (fastest) |
| PyTorch | .pth | Development (requires architecture) |
| ExportedProgram | .pt2 | Not recommended (version issues) |

## Performance

On RTX 4090 with 1B TorchScript model:
- Single person: ~40 FPS
- 5 people: ~15 FPS
- 10 people: ~8 FPS
- 20 people: ~4 FPS

## Output Format

The pipeline outputs:
- Visualization video with skeletons and track IDs
- JSON file with frame-by-frame keypoints
- Each person has 308 Goliath keypoints with x, y, confidence

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or use smaller model (0.6b instead of 1b)

### Missing People
Lower detection threshold or increase max_people

### Jittery Keypoints
Enable temporal smoothing (enabled by default)

### Model Loading Error
Ensure you're using TorchScript models (.pt2) downloaded with the correct format

## Credits

Based on Meta's Sapiens: https://github.com/facebookresearch/sapiens