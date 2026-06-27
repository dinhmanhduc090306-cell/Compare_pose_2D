"""
extract_ap3d_features.py
========================
Extract CPN image features for AP3D validation data.

Usage:
    python extract_ap3d_features.py

Extracts features for S1, S2, and S3 using the camera pairs chosen in
prepare_ap3d_dataset.py (saved in data/ap3d_cameras.pkl).
"""

import argparse
import pickle
import numpy as np
import os
import sys
import cv2
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from networks.network import CPN101

def parse_args():
    parser = argparse.ArgumentParser(description='Extract CPN features for AP3D')
    parser.add_argument('--ap3d_dir', type=str, default='ap3d', help='AP3D dataset root')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for CPN')
    parser.add_argument('--cpn_checkpoint', type=str, default='data/pretrained/cpn_101_384x288.pth.tar')
    return parser.parse_args()

def load_cpn_model(checkpoint_path, device):
    model = CPN101(out_size=(96, 72), num_class=17, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def preprocess_frame(frame):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = cv2.resize(frame, (288, 384))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    return image

def extract_features_from_video(video_path, frame_indices, model, device, batch_size=32):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    sorted_indices = sorted(list(frame_indices))
    if len(sorted_indices) == 0:
        cap.release()
        return np.zeros((0, 1024), dtype=np.float32)
        
    min_idx = sorted_indices[0]
    max_idx = sorted_indices[-1]
    
    # Seek once to the starting frame index
    if min_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_idx)
        
    current_idx = min_idx
    target_set = set(sorted_indices)
    extracted_frames = {}
    
    while current_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_set:
            extracted_frames[current_idx] = frame
        current_idx += 1
        
    cap.release()
    
    frames = []
    last_valid_frame = None
    for idx in sorted_indices:
        if idx in extracted_frames:
            frame = extracted_frames[idx]
            last_valid_frame = frame
        else:
            frame = last_valid_frame
            
        if frame is None:
            # Fallback if no frame was read
            frame = np.zeros((384, 288, 3), dtype=np.uint8)
            
        frames.append(frame)
        
    all_features = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_tensor = np.array([preprocess_frame(f) for f in batch_frames])
            batch_tensor = torch.tensor(batch_tensor).to(device)
            f_maps_list = model(batch_tensor)
            f_maps = f_maps_list[-1].mean(dim=[2, 3])
            all_features.append(f_maps.cpu().numpy())
            
    return np.concatenate(all_features, axis=0).astype(np.float32)


def get_unique_subaction(item):
    sub = item['subject']
    if sub == 'S1':
        return item['subaction']
    elif sub == 'S2':
        basename = os.path.basename(item['image_path'])
        parts = basename.split('_')
        vid_idx = int(parts[3])
        return f"Running_{vid_idx}"
    elif sub == 'S3':
        basename = os.path.basename(item['image_path'])
        parts = basename.split('_')
        action_name = '_'.join(parts[1:-2])
        return action_name
    return item['subaction']

def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Read chosen cameras
    cam_params_path = os.path.join(args.output_dir, 'ap3d_cameras.pkl')
    if not os.path.exists(cam_params_path):
        print("ERROR: Run prepare_ap3d_dataset.py first")
        return
        
    with open(cam_params_path, 'rb') as f:
        ap3d_cameras = pickle.load(f)
        
    print("Loading CPN model...")
    model = load_cpn_model(args.cpn_checkpoint, device)
    
    valid_file = os.path.join(args.ap3d_dir, 'pose_3d_v3', 'valid.pkl')
    with open(valid_file, 'rb') as f:
        data = pickle.load(f)
        
    groups = {}
    for item in data:
        sub = item['subject']
        unique_subact = get_unique_subaction(item)
        cam = item['cameraid']
        key = (sub, unique_subact, cam)
        if key not in groups:
            groups[key] = {}
        groups[key][item['imageid']] = item
        
    subactions = {'S1': set(), 'S2': set(), 'S3': set()}
    for (sub, unique_subact, cam) in groups.keys():
        subactions[sub].add(unique_subact)
        
    features_dict = {'S1': {}, 'S2': {}, 'S3': {}}
    
    for sub in ['S1', 'S2', 'S3']:
        print(f"\nExtracting features for {sub}...")
        
        for subact in tqdm(sorted(subactions[sub]), desc=f"{sub} Subactions"):
            if isinstance(ap3d_cameras[sub], dict):
                cam_a_name = ap3d_cameras[sub][subact][0]['id']
                cam_b_name = ap3d_cameras[sub][subact][1]['id']
            else:
                cam_a_name = ap3d_cameras[sub][0]['id']
                cam_b_name = ap3d_cameras[sub][1]['id']
                
            key_a = (sub, subact, cam_a_name)
            key_b = (sub, subact, cam_b_name)
            
            if key_a not in groups or key_b not in groups:
                continue
                
            frames_a = groups[key_a]
            frames_b = groups[key_b]
            
            sync_ids = sorted(set(frames_a.keys()).intersection(set(frames_b.keys())))
            if len(sync_ids) == 0:
                continue
                
            # Get the first item to deduce video path
            item_a = frames_a[sync_ids[0]]
            item_b = frames_b[sync_ids[0]]
            
            def get_video_path(item, sub, cam_name):
                # deduce from image_path
                # e.g. pose_3d/valid_img/S3_Spin_discus_13_1_0.jpg
                # e.g. pose_3d/valid_img/S2_vid_00_0000_0.jpg
                img_path = item['image_path']
                basename = os.path.basename(img_path)
                
                cam_idx = cam_name.split('_')[-1]
                
                if sub == 'S1':
                    # Axel_1 -> Axel_1_cam_9.mp4
                    # Or check what's in S1 folder. Usually just {subaction}_cam_{idx}.mp4
                    return os.path.join(args.ap3d_dir, 'data', 'valid_set', sub, f"{item['subaction']}_cam_{cam_idx}.mp4")
                elif sub == 'S2':
                    # S2_vid_00_0000_0.jpg -> Running_0_cam_1.mp4
                    parts = basename.split('_')
                    vid_idx = int(parts[3])
                    return os.path.join(args.ap3d_dir, 'data', 'valid_set', sub, f"Running_{vid_idx}_cam_{cam_idx}.mp4")
                elif sub == 'S3':
                    # S3_Spin_discus_13_1_0.jpg -> Spin_discus_13_cam_1.mp4
                    parts = basename.split('_')
                    # S3 (0) _ Spin (1) _ discus (2) _ 13 (3) _ 1 (4) _ 0 (5)
                    # base is parts[1:-2] joined by '_'
                    base = '_'.join(parts[1:-2])
                    return os.path.join(args.ap3d_dir, 'data', 'valid_set', sub, f"{base}_cam_{cam_idx}.mp4")
                    
                return None
                
            vid_a = get_video_path(item_a, sub, cam_a_name)
            vid_b = get_video_path(item_b, sub, cam_b_name)
            
            if not os.path.exists(vid_a) or not os.path.exists(vid_b):
                print(f"  Warning: Video not found for {subact}: {vid_a} or {vid_b}, skipping")
                continue
                
            feat_a = extract_features_from_video(vid_a, sync_ids, model, device, args.batch_size)
            feat_b = extract_features_from_video(vid_b, sync_ids, model, device, args.batch_size)
            
            features_dict[sub][subact] = [feat_a, feat_b, feat_a.copy(), feat_b.copy()]
            
    out_path = os.path.join(args.output_dir, 'ap3d_img_features.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"\nSaved features to {out_path}")

if __name__ == '__main__':
    main()
