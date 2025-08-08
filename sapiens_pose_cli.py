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

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def get_sapiens_transform(h, w):
    return transforms.Compose([transforms.Resize((h, w), antialias=True), transforms.ToTensor(), transforms.Normalize(mean=[0.4843, 0.4569, 0.4059], std=[0.2294, 0.2235, 0.2255])])

def get_person_detector(device):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7).to(device).eval()
    return model, weights.transforms()

def estimate_pose_on_crop(crop, pose_model, transform, device):
    heatmaps = pose_model(transform(crop).unsqueeze(0).to(device))[0]
    kpts = np.zeros((heatmaps.shape[0], 2), dtype=np.float32)
    for i, hmap in enumerate(heatmaps.cpu().detach().numpy()):
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        kpts[i] = [x, y]
    return kpts

def draw_poses(img, all_pose_data, min_kpts):
    details = []
    for data in all_pose_data:
        kps = data['keypoints']
        valid_kps = np.sum((kps[:, 0] > 0) | (kps[:, 1] > 0))
        if valid_kps < min_kpts: continue
        details.append({'score': data['score'], 'kps_count': valid_kps})
        for p1_idx, p2_idx in SKELETON:
            p1, p2 = kps[p1_idx], kps[p2_idx]
            if np.any(p1==0) or np.any(p2==0): continue
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
        for pt in kps:
            if np.any(pt==0): continue
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
    return img, details

class SimpleTracker:
    def __init__(self, max_people=1, iou_threshold=0.3):
        self.max_people = max_people
        self.iou_threshold = iou_threshold
        self.tracks = []  # list of dictionaries: {'box': [x1,y1,x2,y2], 'id': int}

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, detections):
        if not self.tracks:  # first frame or lost all tracks
            self.tracks = [{'box': det['box'], 'score': det['score']} for det in detections[:self.max_people]]
            return self.tracks

        updated_tracks = []
        used_det_indices = set()

        for track in self.tracks:
            best_match_iou = -1
            best_match_idx = -1
            for i, det in enumerate(detections):
                if i in used_det_indices: continue
                iou = self._calculate_iou(track['box'], det['box'])
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = i

            if best_match_iou > self.iou_threshold:
                updated_tracks.append(detections[best_match_idx])
                used_det_indices.add(best_match_idx)

        self.tracks = updated_tracks
        return self.tracks

def process_frame_pipeline(frame, person_detector, pose_model, detector_transform, sapiens_transform, device, min_keypoints, tracker):
    black_canvas = np.zeros_like(frame)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        detections = person_detector([detector_transform(pil_img).to(device)])[0]

    person_detections = sorted(
        [{'box': b.cpu().numpy(), 'score': s.cpu().numpy()} for b, s, l in zip(detections['boxes'], detections['scores'], detections['labels']) if l == 1],
        key=lambda p: p['score'], reverse=True
    )

    # use the tracker to get the stable list of people to process
    tracked_detections = tracker.update(person_detections)

    all_pose_data = []
    for det in tracked_detections:
        x1, y1, x2, y2 = map(int, det['box'])
        crop = pil_img.crop((x1, y1, x2, y2))
        if crop.width == 0 or crop.height == 0: continue

        kpts = estimate_pose_on_crop(crop, pose_model, sapiens_transform, device)

        h, w = (256, 192) # Heatmap size
        kpts[:, 0] = (kpts[:, 0] / w) * crop.width + x1
        kpts[:, 1] = (kpts[:, 1] / h) * crop.height + y1
        all_pose_data.append({'keypoints': kpts, 'score': det['score']})

    pose_frame, details = draw_poses(black_canvas, all_pose_data, min_keypoints)
    return pose_frame, details, tracked_detections

def main():
    parser = argparse.ArgumentParser(description="A stable, two-stage tracking pipeline for SAPIENS pose estimation.")
    parser.add_argument("input_video", help="Path to input video.")
    parser.add_argument("output_path", help="Path for output video or debug image.")
    parser.add_argument("--min-keypoints", type=int, default=7, help="Minimum valid keypoints to draw a skeleton.")
    parser.add_argument("--max-people", type=int, default=1, help="Number of primary people to track (0 for no limit).")
    parser.add_argument("--enable-tracking", action="store_true", help="Enable tracking to prevent flickering (recommended when max-people > 0).")
    parser.add_argument("--debug-frame", type=int, help="Process a single frame for debugging (tracking is disabled in debug).")
    parser.add_argument("--verbose-progress", action="store_true", help="Show detailed per-person stats in the progress bar.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    person_detector, dt = get_person_detector(device)
    pose_model = torch.jit.load(hf_hub_download("facebook/sapiens-pose-1b-torchscript", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"), map_location=device).eval()
    st = get_sapiens_transform(1024, 768)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened(): sys.exit(f"Error opening video: {args.input_video}")

    tracker = SimpleTracker(args.max_people) if args.enable_tracking and args.max_people > 0 else None

    if args.debug_frame is not None:
        run_debug_mode(args, cap, person_detector, pose_model, dt, st, device)
    else:
        run_video_mode(args, cap, person_detector, pose_model, dt, st, device, tracker)

def run_video_mode(args, cap, person_detector, pose_model, dt, st, device, tracker):
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        # If tracking is disabled, create a dummy tracker for this frame only
        frame_tracker = tracker if tracker else SimpleTracker(args.max_people)

        pose_frame, details, _ = process_frame_pipeline(frame, person_detector, pose_model, dt, st, device, args.min_keypoints, frame_tracker)
        out.write(pose_frame)

        p_str = f"\rProcessing frame {i+1}/{total_frames} | "
        p_str += f"Tracking: {'ON' if tracker else 'OFF'} | "
        if args.verbose_progress:
            d_str = ", ".join([f"P{j+1}({d['score']:.2f}): {d['kps_count']} kpts" for j, d in enumerate(details)])
            p_str += f"Drawn: {len(details)} [{d_str if d_str else 'None'}]"
        else:
            p_str += f"People Drawn: {len(details)}"
        sys.stdout.write(p_str.ljust(120)); sys.stdout.flush()

    cap.release(); out.release(); print(f"\nComplete. Video saved to {args.output_path}")

def run_debug_mode(args, cap, person_detector, pose_model, dt, st, device):
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.debug_frame)
    ret, frame = cap.read()
    if not ret: return

    # In debug mode, we always just get the top N by score (no tracking)
    debug_tracker = SimpleTracker(args.max_people)
    pose_frame, details, detections = process_frame_pipeline(frame, person_detector, pose_model, dt, st, device, args.min_keypoints, debug_tracker)

    print("\n--- Detection & Pose Summary (Top Detections by Score) ---")
    if not detections:
        print("No people detected.")
    else:
        for i, det in enumerate(detections):
            info = next((d for d in details if np.isclose(d['score'], det['score'])), None)
            kps_info = f"{info['kps_count']} kpts" if info else "Filtered Out"
            print(f"Person {i+1}: Score={det['score']:.3f} | BBox={np.int32(det['box'])} | Pose Quality: {kps_info}")

    fname = f"{os.path.splitext(args.output_path)[0]}_debug_frame_{args.debug_frame}.jpg"
    cv2.imwrite(fname, pose_frame); print(f"\nDebug image saved to {fname}")
    cap.release()

if __name__ == '__main__':
    main()
