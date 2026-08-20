import json
import os
import glob
import numpy as np

# --- CONFIGURATION ---
JSON_DIR = '/home/duc090306/belief_data/annotations' # Change this to your directory
OUT_DIR = 'belief_data' # Change this to where you want the .npz files

# Camera IDs typically used in H36M (ensure these match your JSON keys if needed)
# CAM_IDS = [0, 1, 2, 3]
CAM_IDS = [0, 1, 0, 1] 

def project_world_to_2d(joints_3d_world, R, t, f, c):
    """
    Transforms 3D world coordinates to 2D pixel coordinates.
    joints_3d_world: (N, 17, 3)
    R: (3, 3) rotation matrix
    t: (3,) translation vector
    f: (2,) focal length [fx, fy]
    c: (2,) principal point [cx, cy]
    """
    # 1. World to Camera Coordinate System (X_cam = X_world * R^T + t)
    # Using np.tensordot to handle the (N, 17, 3) batch cleanly
    joints_3d_cam = np.tensordot(joints_3d_world, R.T, axes=([2], [1])) + t

    # 2. Camera 3D to 2D Pixels
    X = joints_3d_cam[..., 0]
    Y = joints_3d_cam[..., 1]
    Z = joints_3d_cam[..., 2]

    # Prevent division by zero
    Z = np.where(Z == 0, 1e-10, Z)

    u = (X / Z) * f[0] + c[0]
    v = (Y / Z) * f[1] + c[1]

    # Stack to get (N, 17, 2)
    joints_2d = np.stack((u, v), axis=-1)
    return joints_2d

def main():
    positions_3d = {}
    positions_2d = {}

    # Find all 3D joint JSON files
    joint_files = glob.glob(os.path.join(JSON_DIR, 'Human36M_subject*_joint_3d.json'))

    for j_file in joint_files:
        # Extract subject ID from filename (e.g., 'S1')
        subject_id = os.path.basename(j_file).split('_')[1]
        print(f"Processing {subject_id}...")

        # Load 3D joints
        with open(j_file, 'r') as f:
            j3d_data = json.load(f)

        # Load corresponding camera parameters
        cam_file = os.path.join(JSON_DIR, f'Human36M_{subject_id}_camera.json')
        with open(cam_file, 'r') as f:
            cam_data = json.load(f)
            
        # --- NEW: Load corresponding data file to map official action names ---
        data_file = os.path.join(JSON_DIR, f'Human36M_{subject_id}_data.json')
        action_name_map = {}
        
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                meta_data = json.load(f)
            
            # Scan the images list to link (action_idx, subaction_idx) to action_name
            for item in meta_data.get('images', []):
                a_idx = str(item['action_idx'])
                s_idx = str(item['subaction_idx'])
                name = item['action_name']
                
                # Add to mapping if we haven't mapped this specific pair yet
                if (a_idx, s_idx) not in action_name_map:
                    action_name_map[(a_idx, s_idx)] = name
        else:
            print(f"Warning: {data_file} not found. Will use fallback Act_Sub naming.")

        subject_id = subject_id[7:]
        positions_3d[subject_id] = {}
        positions_2d[subject_id] = {}

        subject_data = j3d_data
        
        # Iterate over actions
        for action_id, subactions in subject_data.items():
            for subaction_id, frames in subactions.items():
                # DSVTformer usually groups by a string action name (e.g., "Walking 1")
                # Combine action and subaction to create a unique sequence name
                action_base = action_name_map.get(
                    (str(action_id), str(subaction_id)), 
                    f"Act{action_id}_Sub{subaction_id}"
                )
                
                # --- FIX: Match the pkl script format ---
                action_name = f"{action_base} {subaction_id}"
                
                # --- ADD THIS: Synchronize with the .pkl filter ---
                if subject_id == '11' and action_name == 'Directions 2':
                    print("Skipping corrupted S11 Directions 2 to match .pkl file...")
                    continue
                # --------------------------------------------------

                # Sort frames by frame_id to ensure sequential order
                sorted_frame_ids = sorted(frames.keys(), key=lambda x: int(x))
                
                # Extract 3D joints and convert to numpy array (N, 17, 3)
                seq_3d_joints = []
                for fid in sorted_frame_ids:
                    seq_3d_joints.append(frames[fid])
                
                seq_3d_joints = np.array(seq_3d_joints, dtype=np.float32)
                
                # Save 3D sequence
                positions_3d[subject_id][action_name] = seq_3d_joints
                
                # Prepare 2D sequence list (one array per camera)
                positions_2d[subject_id][action_name] = []
                
                # Project 3D joints to 2D for each camera view
                for cam_idx in CAM_IDS:
                    cam_info = cam_data[str(cam_idx + 1)]
                    R = np.array(cam_info['R'], dtype=np.float32)
                    t = np.array(cam_info['t'], dtype=np.float32).flatten()
                    f = np.array(cam_info['f'], dtype=np.float32).flatten()
                    c = np.array(cam_info['c'], dtype=np.float32).flatten()
                    
                    # Project
                    seq_2d_joints = project_world_to_2d(seq_3d_joints, R, t, f, c)
                    positions_2d[subject_id][action_name].append(seq_2d_joints)

    # --- SAVE THE .NPZ FILES ---
    
    # 1. Save 3D Data
    out_3d_path = os.path.join(OUT_DIR, 'data_3d_h36m.npz')
    np.savez(out_3d_path, positions_3d=positions_3d)
    print(f"Saved 3D data to {out_3d_path}")

    # 2. Save 2D Data with required symmetry metadata
    out_2d_path = os.path.join(OUT_DIR, 'data_2d_h36m_cpn_ft_h36m_dbb.npz')
    
    # DSVTformer uses this metadata dictionary to perform left-right flip augmentations
    metadata = {
        'keypoints_symmetry': (
            [4, 5, 6, 10, 11, 12], # Left side joints
            [1, 2, 3, 13, 14, 15]  # Right side joints
        )
    }
    np.savez(out_2d_path, positions_2d=positions_2d, metadata=metadata)
    print(f"Saved 2D data to {out_2d_path}")

if __name__ == "__main__":
    main()
