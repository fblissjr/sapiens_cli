import argparse
import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import sys
import os
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image

# Constants (SKELETON, COLORS)
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# Preprocessing Transform ---
def get_sapiens_transform(target_height, target_width):
    return transforms.Compose([
        transforms.Resize((target_height, target_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4843, 0.4569, 0.4059], std=[0.2294, 0.2235, 0.2255])
    ])

# Stage 1: Person Detectorr
def get_person_detector(device):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.to(device)
    model.eval()
    return model, weights.transforms()

# Stage 2: Pose Estimation on a Single Crop
def estimate_pose_on_crop(crop, pose_model, sapiens_transform, device):
    input_tensor = sapiens_transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        # The model returns heatmaps; we need to find the max index to get coordinates
        heatmaps = pose_model(input_tensor)[0]

    num_kpts = heatmaps.shape[0]
    model_h, model_w = heatmaps.shape[-2:]
    keypoints = np.zeros((num_kpts, 2), dtype=np.float32)

    for i in range(num_kpts):
        heatmap = heatmaps[i]
        max_val_y, max_val_x = np.unravel_index(np.argmax(heatmap.cpu().numpy()), (model_h, model_w))
        keypoints[i, 0] = max_val_x
        keypoints[i, 1] = max_val_y

    return keypoints

# --- Drawing function with filtering
def draw_poses(img, all_poses, min_keypoints_to_draw):
    # This function receives a list of pose arrays
    num_drawn = 0
    for pose in all_poses:
        # todo: add a basic quality check here
        if np.sum((pose[:, 0] > 0) | (pose[:, 1] > 0)) < min_keypoints_to_draw:
            continue
        num_drawn+=1
        for p1_idx, p2_idx in SKELETON:
            p1 = pose[p1_idx]
            p2 = pose[p2_idx]
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
        for point in pose:
             cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    return img, num_drawn

def main():
    parser = argparse.ArgumentParser(description="A CORRECT two-stage pipeline for SAPIENS pose estimation.")
    parser.add_argument("input_video", help="Path to input video.")
    parser.add_argument("output_path", help="Path for output video or debug image.")
    parser.add_argument("--min-keypoints", type=int, default=7)
    parser.add_argument("--debug-frame", type=int, help="Process a single frame for debugging.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load both models
    person_detector, detector_transform = get_person_detector(device)
    pose_model_path = hf_hub_download("facebook/sapiens-pose-1b-torchscript", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")
    pose_model = torch.jit.load(pose_model_path, map_location=device).eval()
    sapiens_transform = get_sapiens_transform(1024, 768)

    # Open video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error opening video: {args.input_video}", file=sys.stderr)
        return

    # Route to debug or full video mode
    if args.debug_frame is not None:
        run_debug_mode(args, cap, person_detector, pose_model, detector_transform, sapiens_transform, device)
    else:
        run_video_mode(args, cap, person_detector, pose_model, detector_transform, sapiens_transform, device)

def run_video_mode(args, cap, person_detector, pose_model, detector_transform, sapiens_transform, device):
    # Setup video writer
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        frame_with_poses, num_drawn = process_frame_pipeline(frame, person_detector, pose_model, detector_transform, sapiens_transform, device, args.min_keypoints)
        out.write(frame_with_poses)
        sys.stdout.write(f"\rProcessing frame {frame_count+1}/{total_frames} | People Drawn: {num_drawn}")
        sys.stdout.flush()

    cap.release()
    out.release()
    print(f"\nComplete. Video saved to {args.output_path}")

def run_debug_mode(args, cap, person_detector, pose_model, detector_transform, sapiens_transform, device):
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.debug_frame)
    ret, frame = cap.read()
    if not ret: return

    frame_with_poses, num_drawn = process_frame_pipeline(frame, person_detector, pose_model, detector_transform, sapiens_transform, device, args.min_keypoints)

    output_filename = f"{os.path.splitext(args.output_path)[0]}_debug_frame_{args.debug_frame}.jpg"
    cv2.imwrite(output_filename, frame_with_poses)
    print(f"\nDebug image with {num_drawn} poses saved to {output_filename}")
    cap.release()

def process_frame_pipeline(frame, person_detector, pose_model, detector_transform, sapiens_transform, device, min_keypoints):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # --- STAGE 1: Detect People ---
    detector_input = detector_transform(pil_img).to(device)
    with torch.no_grad():
        detections = person_detector([detector_input])[0]

    all_poses_in_frame = []

    # STAGE 2: Estimate Pose for each person
    for i in range(len(detections['boxes'])):
        if detections['labels'][i] == 1: # Label 1 is 'person' in coco dataset
            bbox = detections['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)

            # Crop the person from the PIL image
            person_crop = pil_img.crop((x1, y1, x2, y2))
            crop_w, crop_h = person_crop.size

            if crop_w == 0 or crop_h == 0: continue

            # get keypoints in the coordinate system of the SAPIENS model (ex: 192x256)
            model_kpts = estimate_pose_on_crop(person_crop, pose_model, sapiens_transform, device)

            # scale keypoints from model space to crop space, then shift to full image space
            model_h, model_w = (256, 192) # heatmap size from their app
            scaled_kpts = np.zeros_like(model_kpts)
            scaled_kpts[:, 0] = (model_kpts[:, 0] / model_w) * crop_w + x1
            scaled_kpts[:, 1] = (model_kpts[:, 1] / model_h) * crop_h + y1

            all_poses_in_frame.append(scaled_kpts)

    frame_with_poses, num_drawn = draw_poses(frame, all_poses_in_frame, min_keypoints)
    return frame_with_poses, num_drawn

if __name__ == '__main__':
    main()
