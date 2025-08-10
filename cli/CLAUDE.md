# Sapiens Video Pipeline - Project Status

## Current State (December 2024)

### What's Working ✅
- **Multi-person detection** with YOLO (yolov8m.pt)
- **Multi-person tracking** with DeepSORT (from ~/workspace/deep_sort)
- **308 keypoint pose estimation** (Goliath skeleton with fingers, face, ears, eyes)
- **28 body part segmentation** (colored regions for each body part)
- **Features-only output** (black background, no original video)
- **TorchScript models** (fastest inference at ~40 FPS single person)
- **BF16 optimization** on RTX 4090

### What's NOT Working ❌
- **ByteTrack** - pip install fails due to torch dependency issues
- **RTMDet detector** - mmdet import issues despite local installation
- **Depth models** - Downloaded but not integrated into pipeline
- **Normal models** - Downloaded but not integrated into pipeline
- **Unified CLI** - Currently using separate scripts for different modes

## Code Organization

### Active Scripts
```
sapiens_multi_person.py    # Complete multi-person pipeline with tracking
sapiens_inference.py       # Single person/frame pose estimation  
sapiens_segmentation.py    # Segmentation only
sapiens_combined.py        # Side-by-side pose + segmentation
sapiens_constants.py       # 308 keypoint definitions
sapiens_refinement.py      # Sub-pixel refinement (Dark/UDP)
download_models.py         # Model downloader
```

### Archived Scripts
```
archive/
├── sapiens_full_pipeline.py  # Old pipeline, replaced by multi_person
├── sapiens.py                # Old wrapper, not needed
└── test_*.mp4                # Test outputs
```

## Technical Details

### Model Formats
- **TorchScript (.pt2)**: Working perfectly, best performance
- **PyTorch (.pth)**: Working but slower
- **ExportedProgram**: Version compatibility issues

### Tracking Methods
- **DeepSORT**: Working, best for moving cameras
- **IoU Tracker**: Working, simple but effective
- **ByteTrack**: Not working due to installation issues

### Detection Methods
- **YOLO**: Working perfectly
- **RTMDet**: Not working due to mmdet issues

## Performance Benchmarks (RTX 4090)

### Single Person
- Pose only: ~40 FPS
- Pose + Segmentation: ~25 FPS

### Multi-Person with Tracking
- 5 people: ~15 FPS
- 10 people: ~8 FPS
- 20 people: ~4 FPS

## Known Issues

1. **ByteTrack Installation**
   - Error: "No module named 'torch'" during pip install
   - Workaround: Use DeepSORT instead

2. **RTMDet/mmdet**
   - Local installation at ../det/mmdet not importing correctly
   - Workaround: Use YOLO instead

3. **Multiple Scripts**
   - Currently need different scripts for different modes
   - TODO: Create unified CLI with all options

## Future Improvements

### High Priority
- [ ] Create single unified CLI script with all options
- [ ] Integrate depth estimation
- [ ] Integrate normal estimation

### Low Priority
- [ ] Fix ByteTrack installation
- [ ] Fix mmdet import for RTMDet
- [ ] Add TensorRT optimization
- [ ] Add real-time streaming support

## Usage Patterns

### Complete Pipeline (Everything)
```bash
python sapiens_multi_person.py input.mp4 output.mp4 \
  --pose-model checkpoints/pose/torchscript/1b/*.pt2 \
  --seg-model checkpoints/seg/torchscript/1b/*.pt2
```

### Quick Testing
```bash
# Test on 5 frames
python sapiens_multi_person.py input.mp4 test.mp4 \
  --pose-model checkpoints/pose/torchscript/1b/*.pt2 \
  --max-frames 5
```

## Important Notes

- Models expect 1024x768 (H x W) input resolution
- DeepSORT uses placeholder features (could improve with ReID network)
- Segmentation has 28 classes, pose has 308 keypoints
- All outputs are features-only on black background by default
- JSON output includes frame-by-frame tracking with all keypoints

## File Paths

- Models: `checkpoints/pose/` and `checkpoints/seg/`
- DeepSORT: `/home/fbliss/workspace/deep_sort`
- YOLO: `yolov8m.pt` in current directory
- Output: Video + JSON with tracking data