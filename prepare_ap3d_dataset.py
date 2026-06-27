"""
prepare_ap3d_dataset.py
========================
Generates data_3d_ap3d.npz and data_2d_ap3d_cpn.npz from ap3d/pose_3d_v3/valid.pkl.

Usage:
    python prepare_ap3d_dataset.py

This script processes S1, S2, and S3. For each subject, it dynamically finds the
pair of cameras with a relative azimuth closest to 90 degrees. It then extracts
the synchronized 2D/3D data for that pair and pads it to 4 views [A, B, A, B].
"""

import argparse
import pickle
import numpy as np
import os
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare AP3D dataset for 2-view evaluation')
    parser.add_argument('--ap3d_dir', type=str, default='ap3d', help='Path to AP3D dataset root')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    return parser.parse_args()

def load_valid_pkl(ap3d_dir):
    valid_file = os.path.join(ap3d_dir, 'pose_3d_v3', 'valid.pkl')
    with open(valid_file, 'rb') as f:
        data = pickle.load(f)
    return data

def get_azimuth(R):
    view_dir = R.T @ np.array([0, 0, 1])
    azimuth = math.degrees(math.atan2(view_dir[1], view_dir[0]))
    return azimuth

def find_best_camera_pair(cam_params_dict):
    """Find the pair of cameras with relative azimuth closest to 90 degrees."""
    cams = list(cam_params_dict.keys())
    azimuths = {}
    for cam in cams:
        R = np.array(cam_params_dict[cam]['R'])
        azimuths[cam] = get_azimuth(R)
        
    best_pair = None
    min_diff_to_90 = float('inf')
    
    for i in range(len(cams)):
        for j in range(i+1, len(cams)):
            a1 = azimuths[cams[i]]
            a2 = azimuths[cams[j]]
            diff = abs(a1 - a2)
            if diff > 180:
                diff = 360 - diff
                
            diff_to_90 = abs(diff - 90)
            if diff_to_90 < min_diff_to_90:
                min_diff_to_90 = diff_to_90
                best_pair = (cams[i], cams[j])
                
    return best_pair

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

def prepare_dataset(args):
    print("Preparing AP3D dataset...")
    data = load_valid_pkl(args.ap3d_dir)
    
    # 1. Gather all cameras per subject to find the best pair
    cam_params_by_sub = {'S1': {}, 'S2': {}, 'S3': {}}
    for item in data:
        sub = item['subject']
        cam = item['cameraid']
        if cam not in cam_params_by_sub[sub]:
            cam_params_by_sub[sub][cam] = item['camera_param']
            
    best_pairs = {}
    for sub in ['S1', 'S2', 'S3']:
        pair = find_best_camera_pair(cam_params_by_sub[sub])
        best_pairs[sub] = pair
        print(f"Subject {sub}: Best ~90° pair is {pair[0]} and {pair[1]}")
        
    # 2. Group data by (subject, unique_subaction, cameraid) -> {imageid: item}
    groups = {}
    for item in data:
        sub = item['subject']
        unique_subact = get_unique_subaction(item)
        cam = item['cameraid']
        
        key = (sub, unique_subact, cam)
        if key not in groups:
            groups[key] = {}
        groups[key][item['imageid']] = item
        
    # Find all base subactions per subject
    subactions = {'S1': set(), 'S2': set(), 'S3': set()}
    for (sub, base_subact, cam) in groups.keys():
        subactions[sub].add(base_subact)
        
    import csv
    from collections import defaultdict
    
    csv_counts = defaultdict(list)
    with open('camera_choice.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = row['Tập']
            motion = row['Motion']
            parts = motion.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                prefix = '_'.join(parts[:-1])
            else:
                prefix = motion
            csv_counts[(subj, prefix)].append((motion, row['Cam 1'], row['Cam 2']))

    for k in csv_counts:
        def sort_key(item):
            m = item[0]
            parts = m.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
            return m
        csv_counts[k].sort(key=sort_key)

    pkl_counts = defaultdict(list)
    for sub in subactions:
        for subact in subactions[sub]:
            parts = subact.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                prefix = '_'.join(parts[:-1])
            else:
                prefix = subact
            pkl_counts[(sub, prefix)].append(subact)

    for k in pkl_counts:
        def sort_key2(m):
            parts = m.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
            return m
        pkl_counts[k].sort(key=sort_key2)

    subact_to_cameras = {}
    for sub in subactions:
        subact_to_cameras[sub] = {}
        for subact in subactions[sub]:
            parts = subact.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                prefix = '_'.join(parts[:-1])
            else:
                prefix = subact
            idx = pkl_counts[(sub, prefix)].index(subact)
            csv_list = csv_counts[(sub, prefix)]
            if idx < len(csv_list):
                mapped_motion, cam1, cam2 = csv_list[idx]
            else:
                mapped_motion, cam1, cam2 = csv_list[-1]
                
            if sub == 'S1':
                c1_name, c2_name = f"fs_camera_{cam1}", f"fs_camera_{cam2}"
            elif sub == 'S2':
                c1_name, c2_name = f"rm_camera_{cam1}", f"rm_camera_{cam2}"
            else:
                c1_name, c2_name = f"tnf_camera_{cam1}", f"tnf_camera_{cam2}"
            subact_to_cameras[sub][subact] = (c1_name, c2_name)
    
    positions_3d = {'S1': {}, 'S2': {}, 'S3': {}}
    positions_2d = {'S1': {}, 'S2': {}, 'S3': {}}
    ap3d_cameras = {'S1': {}, 'S2': {}, 'S3': {}}
    
    def format_cam(cp, cam_name):
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(cp['R'])
        quat_xyzw = rot.as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        res_w = cp.get('res_w', 1920)
        res_h = cp.get('res_h', 1088)
        return {
            'id': cam_name,
            'center': [cp['cx'], cp['cy']],
            'focal_length': [cp['fx'], cp['fy']],
            'radial_distortion': [0.0, 0.0, 0.0],
            'tangential_distortion': [0.0, 0.0],
            'res_w': res_w,
            'res_h': res_h,
            'orientation': quat_wxyz,
            'translation': cp['T']
        }
    
    for sub in ['S1', 'S2', 'S3']:
        skipped = 0
        total = 0
        for subact in sorted(subactions[sub]):
            cam_a_name, cam_b_name = subact_to_cameras[sub][subact]
            
            cam_a_param = cam_params_by_sub[sub][cam_a_name]
            cam_b_param = cam_params_by_sub[sub][cam_b_name]
            c_a = format_cam(cam_a_param, cam_a_name)
            c_b = format_cam(cam_b_param, cam_b_name)
            
            ap3d_cameras[sub][subact] = [c_a, c_b, c_a, c_b]
            
            total += 1
            key_a = (sub, subact, cam_a_name)
            key_b = (sub, subact, cam_b_name)
            
            if key_a not in groups or key_b not in groups:
                skipped += 1
                continue
                
            frames_a = groups[key_a]
            frames_b = groups[key_b]
            
            # Find synchronized frame indices
            sync_ids = sorted(set(frames_a.keys()).intersection(set(frames_b.keys())))
            
            if len(sync_ids) == 0:
                skipped += 1
                continue
                
            kp2d_a = []
            kp2d_b = []
            gt3d = []
            
            for fid in sync_ids:
                item_a = frames_a[fid]
                item_b = frames_b[fid]
                
                kp2d_a.append(item_a['joint_3d_image'][:, :2])
                kp2d_b.append(item_b['joint_3d_image'][:, :2])
                
                gt3d_frame = item_a['joint_3d_camera'].copy()
                gt3d.append(gt3d_frame)
                
            kp2d_a = np.array(kp2d_a, dtype=np.float32)
            kp2d_b = np.array(kp2d_b, dtype=np.float32)
            gt3d = np.array(gt3d, dtype=np.float32)
            
            # Pad 2D keypoints
            positions_2d[sub][subact] = [kp2d_a, kp2d_b, kp2d_a.copy(), kp2d_b.copy()]
            positions_3d[sub][subact] = gt3d
            
            print(f"  {sub} {subact}: {len(sync_ids)} synchronized frames")
            
        print(f"  Processed {total - skipped}/{total} subactions ({skipped} skipped)")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    out_3d_path = os.path.join(args.output_dir, 'data_3d_ap3d.npz')
    np.savez_compressed(out_3d_path, positions_3d=positions_3d)
    print(f"Saved 3D data to {out_3d_path}")
    
    metadata = {
        'keypoints_symmetry': [
            [4, 5, 6, 11, 12, 13],
            [1, 2, 3, 14, 15, 16],
        ]
    }
    out_2d_path = os.path.join(args.output_dir, 'data_2d_ap3d_cpn.npz')
    np.savez_compressed(out_2d_path, positions_2d=positions_2d, metadata=metadata)
    print(f"Saved 2D data to {out_2d_path}")
    
    cam_params_path = os.path.join(args.output_dir, 'ap3d_cameras.pkl')
    with open(cam_params_path, 'wb') as f:
        pickle.dump(ap3d_cameras, f)
    print(f"Saved padded camera parameters to {cam_params_path}")
    
if __name__ == '__main__':
    args = parse_args()
    prepare_dataset(args)
