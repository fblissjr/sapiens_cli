import argparse
import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import sys
import os
import json
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
from sapiens_constants import GOLIATH_KEYPOINTS, GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS

# for --outline-by-person feature
PERSON_OUTLINE_COLORS = [
    (255, 255, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (128, 0, 128), (0, 165, 255)
]

def get_sapiens_transform(h, w):
    return transforms.Compose([
        transforms.Resize((h, w), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4843, 0.4569, 0.4059], std=[0.2294, 0.2235, 0.2255])
    ])

def get_person_detector(device):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7).to(device).eval()
    return model, weights.transforms()

def estimate_pose_on_crop(crop, pose_model, transform, device):
    with torch.no_grad():
        heatmaps = pose_model(transform(crop).unsqueeze(0).to(device))[0]
    kpts = np.zeros((heatmaps.shape[0], 2), dtype=np.float32)
    for i, hmap in enumerate(heatmaps.cpu().detach().numpy()):
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        kpts[i] = [x, y]
    return kpts

def draw_poses(img, all_pose_data, min_kpts, outline_by_person=False):
    for i, data in enumerate(all_pose_data):
        if data['kps_count'] < min_kpts: continue

        person_outline_color = PERSON_OUTLINE_COLORS[i % len(PERSON_OUTLINE_COLORS)] if outline_by_person else None

        for link_info in GOLIATH_SKELETON_INFO.values():
            pt1_name, pt2_name = link_info['link']
            feature_color = link_info['color'][::-1]
            pt1, pt2 = data['keypoints'].get(pt1_name), data['keypoints'].get(pt2_name)

            if pt1 and pt2:
                if person_outline_color:
                    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), person_outline_color, 4)
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), feature_color, 2)

        for name, pt in data['keypoints'].items():
            try:
                idx = GOLIATH_KEYPOINTS.index(name)
                feature_color = GOLIATH_KPTS_COLORS[idx][::-1]
                if person_outline_color:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 6, person_outline_color, -1)
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, feature_color, -1)
            except (ValueError, IndexError):
                continue # safety check if a keypoint name is not in the list

    return img, len([p for p in all_pose_data if p['kps_count'] >= min_kpts])

class SimpleTracker:
    def __init__(self, max_people=1, iou_threshold=0.3):
        self.max_people, self.iou_threshold, self.tracks = max_people, iou_threshold, []
    def _calculate_iou(self, bA, bB):
        xA,yA,xB,yB=max(bA[0],bB[0]),max(bA[1],bB[1]),min(bA[2],bB[2]),min(bA[3],bB[3])
        iA=max(0,xB-xA)*max(0,yB-yA);bAA,bBA=(bA[2]-bA[0])*(bA[3]-bA[1]),(bB[2]-bB[0])*(bB[3]-bB[1])
        return iA/float(bAA+bBA-iA) if (bAA+bBA-iA)>0 else 0
    def update(self, dets):
        if not self.tracks: self.tracks=[{'box':d['box'],'score':d['score']} for d in dets[:self.max_people]]; return self.tracks
        u_t,u_d=[],set()
        for t in self.tracks:
            b_m_i,b_m_idx=-1,-1
            for i,d in enumerate(dets):
                if i in u_d: continue
                iou=self._calculate_iou(t['box'],d['box'])
                if iou>b_m_i: b_m_i,b_m_idx=iou,i
            if b_m_i>self.iou_threshold: u_t.append(dets[b_m_idx]); u_d.add(b_m_idx)
        self.tracks=u_t; return self.tracks

def process_frame_pipeline(frame, person_detector, pose_model, detector_transform, sapiens_transform, device, tracker):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad(): dets = person_detector([detector_transform(pil_img).to(device)])[0]
    person_dets = sorted([{'box': b.cpu().numpy(), 'score': s.cpu().numpy()} for b, s, l in zip(dets['boxes'], dets['scores'], dets['labels']) if l == 1], key=lambda p: p['score'], reverse=True)
    tracked_dets = tracker.update(person_dets)
    frame_pose_data = []
    for det in tracked_dets:
        x1, y1, x2, y2 = map(int, det['box'])
        crop = pil_img.crop((x1, y1, x2, y2))
        if crop.width == 0 or crop.height == 0: continue
        kpts_array = estimate_pose_on_crop(crop, pose_model, sapiens_transform, device)
        h, w = (256, 192)
        kpts_dict = {}
        for i, pt in enumerate(kpts_array):
            if np.any(pt > 0) and i < len(GOLIATH_KEYPOINTS):
                kpt_name = GOLIATH_KEYPOINTS[i]
                scaled_x = (pt[0] / w) * crop.width + x1
                scaled_y = (pt[1] / h) * crop.height + y1
                kpts_dict[kpt_name] = [float(scaled_x), float(scaled_y)]
        frame_pose_data.append({'detection_score': float(det['score']),'bounding_box': [x1, y1, x2, y2],'kps_count': len(kpts_dict),'keypoints': kpts_dict})
    return frame_pose_data

def run_video_mode(args, cap, person_detector, pose_model, dt, st, device, tracker):
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_json_data = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        frame_pose_data = process_frame_pipeline(frame, person_detector, pose_model, dt, st, device, tracker)
        pose_canvas = np.zeros_like(frame)
        pose_canvas, num_drawn = draw_poses(pose_canvas, frame_pose_data, args.min_keypoints, args.outline_by_person)
        out.write(pose_canvas)
        if args.output_json: video_json_data.append({'frame': i, 'persons': frame_pose_data})

        p_str = f"\rProcessing frame {i+1}/{total_frames} | "
        if args.verbose_progress:
            details = [d for d in frame_pose_data if d['kps_count'] >= args.min_keypoints]
            d_str = ", ".join([f"P{j+1}({d['detection_score']:.2f}): {d['kps_count']} kpts" for j, d in enumerate(details)])
            p_str += f"Drawn: {len(details)} [{d_str if d_str else 'None'}]"
        else:
            p_str += f"People Drawn: {num_drawn}"
        sys.stdout.write(p_str.ljust(120)); sys.stdout.flush()

    cap.release(); out.release()
    if args.output_json:
        json_path = os.path.splitext(args.output_path)[0] + ".json"
        with open(json_path, 'w') as f: json.dump(video_json_data, f, indent=2)
        print(f"\nDetailed keypoint data saved to {json_path}")
    print(f"\nComplete. Video saved to {args.output_path}")

def run_debug_mode(args, cap, person_detector, pose_model, dt, st, device):
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.debug_frame)
    ret, frame = cap.read()
    if not ret: return

    debug_tracker = SimpleTracker(args.max_people)
    frame_pose_data = process_frame_pipeline(frame, person_detector, pose_model, dt, st, device, debug_tracker)

    print("\n--- Detection & Pose Summary (Top Detections by Score) ---")
    if not frame_pose_data:
        print("No people were detected that met the tracking criteria for this frame.")
    else:
        for i, person_data in enumerate(frame_pose_data):
            score = person_data['detection_score']
            box = person_data['bounding_box']
            kps_count = person_data['kps_count']
            status = f"{kps_count} kpts" if kps_count >= args.min_keypoints else "Filtered Out"
            print(f"Person {i+1}: Score={score:.3f} | BBox={np.int32(box)} | Pose Quality: {status}")

    pose_canvas = np.zeros_like(frame)
    pose_canvas, num_drawn = draw_poses(pose_canvas, frame_pose_data, args.min_keypoints, args.outline_by_person)
    fname = f"{os.path.splitext(args.output_path)[0]}_debug_frame_{args.debug_frame}.jpg"
    cv2.imwrite(fname, pose_canvas); print(f"\nDebug image saved to {fname}")
    cap.release()

def main():
    parser = argparse.ArgumentParser(description="SAPIENS two-stage pose estimation pipeline.")
    parser.add_argument("input_video"); parser.add_argument("output_path")
    parser.add_argument("--min-keypoints", type=int, default=10, help="Min keypoints to draw a skeleton.")
    parser.add_argument("--max-people", type=int, default=1, help="Number of people to track.")
    parser.add_argument("--enable-tracking", action="store_true", help="Enable tracking to stabilize IDs.")
    parser.add_argument("--output-json", action="store_true", help="Save detailed keypoint data to a .json file.")
    parser.add_argument("--outline-by-person", action="store_true", help="Draw a unique color outline for each person.")
    parser.add_argument("--debug-frame", type=int, help="Process a single frame for debugging.")
    parser.add_argument("--verbose-progress", action="store_true", help="Show detailed stats in the progress bar.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    person_detector, dt = get_person_detector(device)
    pose_model = torch.jit.load(hf_hub_download("facebook/sapiens-pose-1b-torchscript", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"), map_location=device).eval()
    st = get_sapiens_transform(1024, 768)
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened(): sys.exit(f"Error opening video: {args.input_video}")

    tracker = SimpleTracker(args.max_people) if args.enable_tracking and args.max_people > 0 else SimpleTracker(100)

    if args.debug_frame is not None:
        run_debug_mode(args, cap, person_detector, pose_model, dt, st, device)
    else:
        run_video_mode(args, cap, person_detector, pose_model, dt, st, device, tracker)

if __name__ == '__main__':
    main()
