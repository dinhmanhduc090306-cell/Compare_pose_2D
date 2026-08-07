"""
evaluate_ap3d.py
================
Standalone evaluation of the H36M-trained DSVTformer model on the AP3D dataset.

Usage:
    python evaluate_ap3d.py --checkpoint checkpoint/2026-05-29_09-39-3827_dsvtformer_0.000200_64_torch_2.4.1+cu124_python_3.11/model_6_3728243.pth

This evaluates using 2 camera views padded to 4: [cam_A, cam_B, cam_A, cam_B]
"""

import argparse
import os
import sys
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

from model.dsvtformer import Model
from common.h36m_dataset import AP3DDataset
from common.Mydataset_img import Fusion
from common.utils import (
    test_calculation, define_error_list, print_error,
    get_varialbe, AccumLoss, p_mpjpe
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DSVTformer on AP3D')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoint/2026-05-29_09-39-3827_dsvtformer_0.000200_64_torch_2.4.1+cu124_python_3.11/model_6_3728243.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--frames', type=int, default=27, help='Number of input frames')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--depth', type=int, default=2, help='Model depth')
    parser.add_argument('--embed_dim_ratio', type=int, default=32)
    parser.add_argument('--img_embed_dim_ratio', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


class OptProxy:
    """Minimal opt object that mimics the training options for the Fusion dataset."""
    pass


def build_opt(args):
    """Build an opt-compatible object for the Fusion dataset class."""
    opt = OptProxy()
    opt.dataset = 'ap3d'
    opt.keypoints = 'cpn'
    opt.data_augmentation = False
    opt.reverse_augmentation = False
    opt.test_augmentation = True
    opt.crop_uv = 0
    opt.root_path = './data/'
    opt.actions = '*'
    opt.downsample = 1
    opt.subset = 1
    opt.stride = 1
    opt.frames = args.frames
    opt.pad = (args.frames - 1) // 2
    opt.batch_size = args.batch_size
    opt.out_all = 1
    opt.subjects_train = ''  # No training
    opt.subjects_test = 'S1,S2,S3'
    opt.train = 0
    opt.workers = args.workers
    return opt


def define_ap3d_actions(dataset):
    """
    Extract unique action categories from the AP3D dataset.
    Action names are derived from subaction names (e.g., 'Axel_1' -> 'Axel',
    'Discus_error_7' -> 'Discus', 'Running_0' -> 'Running').
    """
    actions = set()
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            parts = action.split('_')
            # Strip trailing numbers
            if len(parts) > 1 and parts[-1].isdigit():
                parts = parts[:-1]
            category = '_'.join(parts)
            actions.add(category)
    return sorted(list(actions))


def evaluate(args):
    print(f"=" * 60)
    print(f"Evaluating AP3D (S1, S2, S3) with optimal pairs")
    print(f"=" * 60)

    # Seed
    random.seed(1)
    torch.manual_seed(1)

    # Build opt
    opt = build_opt(args)

    # Load AP3D dataset
    dataset_path = os.path.join(opt.root_path, 'data_3d_ap3d.npz')
    if not os.path.exists(dataset_path):
        print(f"ERROR: {dataset_path} not found. Run prepare_ap3d_dataset.py first.")
        sys.exit(1)

    dataset = AP3DDataset(dataset_path, opt)

    # Get actions
    actions = define_ap3d_actions(dataset)
    print(f"Actions: {actions}")

    # Build test dataset
    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=opt.root_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=opt.batch_size,
        shuffle=False, num_workers=int(opt.workers), pin_memory=True
    )

    # Build model
    model = Model(
        num_frame=opt.frames, num_joints=17, in_chans=2,
        embed_dim_ratio=args.embed_dim_ratio,
        img_embed_dim_ratio=args.img_embed_dim_ratio,
        depth=args.depth, num_heads=8, mlp_ratio=2.,
        qkv_bias=True, qk_scale=None, drop_path_rate=0.
    )

    # Load checkpoint
    gpu_ids = list(map(int, args.gpu.split(',')))
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device(f"cuda:{gpu_ids[0]}")
    model = model.to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        pre_dict = torch.load(args.checkpoint, map_location=device)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(state_dict)}/{len(model_dict)} parameters")
    else:
        print(f"WARNING: Checkpoint not found: {args.checkpoint}")
        print("Running with random weights (for testing only)")

    model.eval()

    mapping_path = 'data/ap3d_mapping.pkl'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'rb') as f:
            ap3d_mapping = pickle.load(f)
    else:
        ap3d_mapping = None

    # Evaluate
    action_error_sum = define_error_list(actions)
    key_frames_cache = {}

    with torch.no_grad():
        TQDM = tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=100)
        for i, data in TQDM:
            batch_cam, gt_3D, input_2D, image, action, subject, scale, bb_box, start, end = data

            [input_2D, image, gt_3D, batch_cam, scale, bb_box] = get_varialbe(
                'test', [input_2D, image, gt_3D, batch_cam, scale, bb_box]
            )

            # Forward pass (no test augmentation for simplicity)
            input_2D_non_flip = input_2D[:, 0]
            output_3D = model(input_2D_non_flip, image)

            if output_3D.shape[1] != 1:
                output_3D = output_3D[:, opt.pad].unsqueeze(1)

            # Root-relative
            output_3D[:, :, 1:, :] -= output_3D[:, :, :1, :]

            out_target = gt_3D.clone()
            out_target[:, :, 0] = 0

            # Calculate errors
            valid_mask = []
            for b in range(len(action)):
                act = action[b]
                sub = subject[b]
                frame_idx = start[b].item()
                if ap3d_mapping is not None and sub in ap3d_mapping and act in ap3d_mapping[sub]:
                    mapping_info = ap3d_mapping[sub][act]
                    csv_motion = mapping_info['csv_motion']
                    cam1 = mapping_info['cam1']
                    cam2 = mapping_info['cam2']
                    
                    kf_key = (sub, csv_motion, cam1, cam2)
                    if kf_key not in key_frames_cache:
                        kfs = set()
                        kf1_path = f'key_frames/{sub}/{csv_motion}_cam_{cam1[-1]}_key_frames.npy'
                        kf2_path = f'key_frames/{sub}/{csv_motion}_cam_{cam2[-1]}_key_frames.npy'
                        if os.path.exists(kf1_path):
                            kfs.update(np.load(kf1_path).tolist())
                        if os.path.exists(kf2_path):
                            kfs.update(np.load(kf2_path).tolist())
                        key_frames_cache[kf_key] = kfs
                        
                    kfs = key_frames_cache[kf_key]
                    if len(kfs) > 0:
                        is_valid = frame_idx in kfs
                    else:
                        is_valid = True
                else:
                    is_valid = True
                valid_mask.append(is_valid)

            action_error_sum = test_calculation(
                output_3D, out_target, action, action_error_sum, opt.dataset, subject, valid_mask
            )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Results for AP3D Evaluation (Optimal Camera Pairs)")
    print(f"{'=' * 60}")
    p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)

    return p1, p2


if __name__ == '__main__':
    args = parse_args()
    p1, p2 = evaluate(args)
    print(f"\nFinal: P1={p1:.2f}mm, P2={p2:.2f}mm")
