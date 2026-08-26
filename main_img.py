import os
import torch
import logging
import random
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.utils import *
from common.opt import opts
from common.h36m_dataset import Human36mDataset
from common.Mydataset_img import Fusion

from fvcore.nn import FlopCountAnalysis
from datetime import datetime
import pytz


# ============================================================
# OPTIONS
# ============================================================

opt = opts().parse()


# ============================================================
# TRAIN ONLY ON HUMAN3.6M SUBJECT S8
# ============================================================

opt.subjects_train = '8'

# Keep standard H36M validation subjects
opt.subjects_test = '9,11'

print(f'Training subjects: {opt.subjects_train}')
print(f'Testing subjects: {opt.subjects_test}')


# ============================================================
# MODEL IMPORT
# ============================================================

exec('from model.' + opt.model + ' import Model')


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.cuda.set_device(f'cuda:{opt.gpu}')


# ============================================================
# TRAIN / VALIDATION FUNCTIONS
# ============================================================

def train(
    opt,
    actions,
    train_loader,
    model,
    optimizer,
    epoch,
    writer,
    adaptive_weight=None
):
    return step(
        'train',
        opt,
        actions,
        train_loader,
        model,
        optimizer,
        epoch,
        writer,
        adaptive_weight
    )


def val(
    opt,
    actions,
    val_loader,
    model,
    writer
):
    with torch.no_grad():
        return step(
            'test',
            opt,
            actions,
            val_loader,
            model,
            writer
        )


# ============================================================
# TRAIN / TEST STEP
# ============================================================

def step(
    split,
    opt,
    actions,
    dataLoader,
    model,
    optimizer=None,
    epoch=None,
    writer=None,
    adaptive_weight=None
):

    loss_all = {
        'loss': AccumLoss()
    }

    action_error_sum = define_error_list(actions)

    import pickle
    import numpy as np

    # ========================================================
    # AP3D MAPPING
    # ========================================================

    mapping_path = 'data/ap3d_mapping.pkl'

    if os.path.exists(mapping_path):

        with open(mapping_path, 'rb') as f:
            ap3d_mapping = pickle.load(f)

    else:

        ap3d_mapping = None

    key_frames_cache = {}


    # ========================================================
    # TRAIN / EVAL MODE
    # ========================================================

    if split == 'train':
        model.train()
    else:
        model.eval()


    # ========================================================
    # DATA LOOP
    # ========================================================

    TQDM = tqdm(
        enumerate(dataLoader),
        total=len(dataLoader),
        ncols=100
    )

    for i, data in TQDM:

        batch_cam, gt_3D, input_2D, image, action, subject, scale, bb_box, start, end = data


        # ====================================================
        # MOVE DATA TO DEVICE
        # ====================================================

        [
            input_2D,
            image,
            gt_3D,
            batch_cam,
            scale,
            bb_box
        ] = get_varialbe(
            split,
            [
                input_2D,
                image,
                gt_3D,
                batch_cam,
                scale,
                bb_box
            ]
        )


        # ====================================================
        # FORWARD PASS
        # ====================================================

        if split == 'train':

            output_3D = model(
                input_2D,
                image
            )

        elif split == 'test':

            input_2D, output_3D = input_augmentation(
                input_2D,
                image,
                model
            )


            # ================================================
            # SAVE RAW 3D POSES
            # ================================================

            import numpy as np

            os.makedirs(
                'output',
                exist_ok=True
            )

            np.save(
                f'output/predicted_poses_{subject[0]}_{action[0]}.npy',
                output_3D.cpu().numpy()
            )


        # ====================================================
        # TARGET
        # ====================================================

        out_target = gt_3D.clone()

        out_target[:, :, 0] = 0


        # ====================================================
        # TRAINING
        # ====================================================

        if split == 'train':

            loss = mpjpe_cal(
                output_3D,
                out_target
            )

            TQDM.set_description(
                f'Epoch [{epoch}/{opt.nepoch}]'
            )

            TQDM.set_postfix(
                {
                    "l": loss.item()
                }
            )

            N = input_2D.size(0)

            loss_all['loss'].update(
                loss.detach().cpu().numpy() * N,
                N
            )


            # ================================================
            # BACKPROPAGATION
            # ================================================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # ================================================
            # TENSORBOARD
            # ================================================

            if writer is not None:

                step_num = (
                    epoch * len(dataLoader) + i
                )

                writer.add_scalar(
                    "Loss/Total_Loss",
                    loss.item(),
                    step_num
                )


        # ====================================================
        # VALIDATION
        # ====================================================

        elif split == 'test':

            if output_3D.shape[1] != 1:

                output_3D = output_3D[
                    :,
                    opt.pad
                ].unsqueeze(1)


            # Root-relative coordinates
            output_3D[:, :, 1:, :] -= (
                output_3D[:, :, :1, :]
            )


            # ================================================
            # VALID FRAME MASK
            # ================================================

            valid_mask = []


            if opt.dataset == 'ap3d':

                for b in range(len(action)):

                    act = action[b]

                    sub = subject[b]

                    frame_idx = start[b].item()


                    if (
                        ap3d_mapping is not None
                        and sub in ap3d_mapping
                        and act in ap3d_mapping[sub]
                    ):

                        mapping_info = (
                            ap3d_mapping[sub][act]
                        )

                        csv_motion = (
                            mapping_info['csv_motion']
                        )

                        cam1 = (
                            mapping_info['cam1']
                        )

                        cam2 = (
                            mapping_info['cam2']
                        )


                        kf_key = (
                            sub,
                            csv_motion,
                            cam1,
                            cam2
                        )


                        # ====================================
                        # LOAD KEY FRAMES
                        # ====================================

                        if kf_key not in key_frames_cache:

                            kfs = set()


                            kf1_path = (
                                f'key_frames/{sub}/'
                                f'{csv_motion}_cam_'
                                f'{cam1[-1]}_key_frames.npy'
                            )


                            kf2_path = (
                                f'key_frames/{sub}/'
                                f'{csv_motion}_cam_'
                                f'{cam2[-1]}_key_frames.npy'
                            )


                            if os.path.exists(kf1_path):

                                kfs.update(
                                    np.load(
                                        kf1_path
                                    ).tolist()
                                )


                            if os.path.exists(kf2_path):

                                kfs.update(
                                    np.load(
                                        kf2_path
                                    ).tolist()
                                )


                            key_frames_cache[kf_key] = kfs


                        kfs = key_frames_cache[kf_key]


                        if len(kfs) > 0:

                            is_valid = (
                                frame_idx in kfs
                            )

                        else:

                            is_valid = True


                    else:

                        is_valid = True


                    valid_mask.append(
                        is_valid
                    )


            else:

                valid_mask = None


            # ================================================
            # CALCULATE TEST ERROR
            # ================================================

            action_error_sum = test_calculation(
                output_3D,
                out_target,
                action,
                action_error_sum,
                opt.dataset,
                subject,
                valid_mask=valid_mask
            )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    if split == 'train':

        return loss_all['loss'].avg

    elif split == 'test':

        p1, p2 = print_error(
            opt.dataset,
            action_error_sum,
            opt.train
        )

        return p1, p2


# ============================================================
# INPUT AUGMENTATION
# ============================================================

def input_augmentation(
    input_2D,
    image,
    model
):

    input_2D_non_flip = input_2D[:, 0]

    output_3D_non_flip = model(
        input_2D_non_flip,
        image
    )

    return (
        input_2D_non_flip,
        output_3D_non_flip
    )


# ============================================================
# MODEL STATISTICS
# ============================================================

def count_parameters_in_M(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ) / 1e6


def count_flops_in_G(
    model,
    input_2d,
    input_img
):

    flops = FlopCountAnalysis(
        model,
        (
            input_2d,
            input_img
        )
    )

    return flops.total() / 1e9


def count_used_parameters_cuda(
    model,
    example_input_2d,
    example_img
):

    model.eval()

    used_params = set()


    def hook_fn(
        module,
        input,
        output
    ):

        for name, param in module.named_parameters(
            recurse=False
        ):

            if param.requires_grad:

                used_params.add(
                    param.data_ptr()
                )


    hooks = []


    for module in model.modules():

        if any(
            p.requires_grad
            for p in module.parameters(
                recurse=False
            )
        ):

            hooks.append(
                module.register_forward_hook(
                    hook_fn
                )
            )


    with torch.no_grad():

        _ = model(
            example_input_2d,
            example_img
        )


    for h in hooks:

        h.remove()


    used_param_count = sum(
        p.numel()
        for p in model.parameters()
        if p.data_ptr() in used_params
    )

    return used_param_count / 1e6


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    import time

    start_time = time.time()


    # ========================================================
    # RANDOM SEED
    # ========================================================

    root_path = opt.root_path

    opt.manualSeed = 1

    random.seed(
        opt.manualSeed
    )

    torch.manual_seed(
        opt.manualSeed
    )


    # ========================================================
    # LOGGING
    # ========================================================

    if opt.train:

        logging.basicConfig(
            format='%(asctime)s %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            filename=os.path.join(
                opt.checkpoint,
                'train.log'
            ),
            level=logging.INFO
        )


    # ========================================================
    # DATASET
    # ========================================================

    root_path = opt.root_path

    dataset_path = (
        root_path
        + 'data_3d_'
        + opt.dataset
        + '.npz'
    )


    if opt.dataset == 'ap3d':

        from common.h36m_dataset import AP3DDataset

        from evaluate_ap3d import (
            define_ap3d_actions
        )

        dataset = AP3DDataset(
            dataset_path,
            opt
        )

        actions = define_ap3d_actions(
            dataset
        )

    else:

        dataset = Human36mDataset(
            dataset_path,
            opt
        )

        actions = define_actions(
            opt.actions
        )


    # ========================================================
    # TRAIN DATA
    # ========================================================

    if opt.train:

        train_data = Fusion(
            opt=opt,
            train=True,
            dataset=dataset,
            root_path=root_path
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            pin_memory=True
        )


    # ========================================================
    # TEST DATA
    # ========================================================

    test_data = Fusion(
        opt=opt,
        train=False,
        dataset=dataset,
        root_path=root_path
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        pin_memory=True
    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = Model(
        num_frame=opt.frames,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=opt.embed_dim_ratio,
        img_embed_dim_ratio=opt.img_embed_dim_ratio,
        depth=opt.depth,
        num_heads=8,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.
    )


    # ========================================================
    # GPU
    # ========================================================

    gpu_ids = list(
        map(
            int,
            opt.gpu.split(',')
        )
    )


    if len(gpu_ids) == 1:

        torch.cuda.set_device(
            gpu_ids[0]
        )

        device = torch.device(
            f"cuda:{gpu_ids[0]}"
        )

        model = model.to(device)


    else:

        print(
            f"Let's use {len(gpu_ids)} GPUs: {gpu_ids}"
        )

        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids
        ).to(
            f"cuda:{gpu_ids[0]}"
        )


    # ========================================================
    # LOAD PRETRAINED MODEL
    # ========================================================

    model_dict = model.state_dict()


    if opt.previous_dir != '':

        print(
            'pretrained model path:',
            opt.previous_dir
        )

        model_path = opt.previous_dir


        pre_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )


        model_dict = model.state_dict()


        state_dict = {
            k: v
            for k, v in pre_dict.items()
            if k in model_dict.keys()
        }


        model_dict.update(
            state_dict
        )


        model.load_state_dict(
            model_dict
        )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    all_param = []

    lr = opt.lr

    all_param += list(
        model.parameters()
    )


    optimizer = optim.AdamW(
        all_param,
        lr=lr,
        weight_decay=0.1
    )


    # ========================================================
    # TENSORBOARD
    # ========================================================

    local_tz = pytz.timezone(
        "Asia/Shanghai"
    )

    current_time = datetime.now(
        local_tz
    ).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


    log_dir = (
        f'runs/'
        f'{opt.model}_'
        f'{current_time}'
    )


    writer = SummaryWriter(
        log_dir
    )


    print(
        f"TensorBoard log directory: {log_dir}"
    )


    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    flag = 0

    best_epoch = 0


    # ========================================================
    # CHECKPOINT DIRECTORY
    # ========================================================

    resume_dir = (
        "/content/drive/MyDrive/"
        "Pose_compare_checkpoints"
    )


    os.makedirs(
        resume_dir,
        exist_ok=True
    )


    latest_checkpoint = os.path.join(
        resume_dir,
        "latest_checkpoint.pth"
    )


    start_epoch = 1


    # ========================================================
    # RESUME CHECKPOINT
    # ========================================================

    if os.path.exists(
        latest_checkpoint
    ):

        print("=" * 60)

        print(
            "FOUND PREVIOUS CHECKPOINT"
        )

        print(
            latest_checkpoint
        )


        checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
            weights_only=False
        )


        # ================================================
        # RESTORE MODEL
        # ================================================

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        # ================================================
        # RESTORE OPTIMIZER
        # ================================================

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )


        # ================================================
        # RESTORE EPOCH
        # ================================================

        start_epoch = (
            checkpoint["epoch"] + 1
        )


        # ================================================
        # RESTORE LEARNING RATE
        # ================================================

        if "lr" in checkpoint:

            lr = checkpoint["lr"]

            for param_group in optimizer.param_groups:

                param_group["lr"] = lr


        # ================================================
        # RESTORE BEST EPOCH
        # ================================================

        if "best_epoch" in checkpoint:

            best_epoch = (
                checkpoint["best_epoch"]
            )


        # ================================================
        # RESTORE BEST P1
        # ================================================

        if (
            "previous_best_threshold"
            in checkpoint
        ):

            opt.previous_best_threshold = (
                checkpoint[
                    "previous_best_threshold"
                ]
            )


        print(
            f"Resuming training from "
            f"Epoch {start_epoch}"
        )

        print(
            f"Previous epoch: "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Previous loss: "
            f"{checkpoint.get('loss', 'N/A')}"
        )

        print(
            f"Previous P1: "
            f"{checkpoint.get('p1', 'N/A')}"
        )

        print("=" * 60)


    else:

        print("=" * 60)

        print(
            "NO PREVIOUS CHECKPOINT FOUND"
        )

        print(
            "Starting training from Epoch 1"
        )

        print("=" * 60)


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        opt.nepoch + 1
    ):


        # ====================================================
        # TRAIN
        # ====================================================

        if opt.train:

            loss = train(
                opt,
                actions,
                train_dataloader,
                model,
                optimizer,
                epoch,
                writer
            )


        # ====================================================
        # VALIDATION
        # ====================================================

        p1, p2 = val(
            opt,
            actions,
            test_dataloader,
            model,
            writer
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            opt.train
            and p1 < opt.previous_best_threshold
        ):

            best_epoch = epoch


            opt.previous_name = save_model(
                opt.previous_name,
                opt.checkpoint,
                epoch,
                p1,
                model
            )


            opt.previous_best_threshold = p1


        # ====================================================
        # FULL CHECKPOINT
        # ====================================================

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                loss if opt.train else None,

            "p1":
                p1,

            "p2":
                p2,

            "lr":
                lr,

            "best_epoch":
                best_epoch,

            "previous_best_threshold":
                opt.previous_best_threshold
        }


        # ====================================================
        # SAVE INDIVIDUAL EPOCH
        # ====================================================

        epoch_checkpoint = os.path.join(
            resume_dir,
            f"epoch_{epoch}.pth"
        )


        torch.save(
            checkpoint,
            epoch_checkpoint
        )


        # ====================================================
        # SAVE LATEST
        # ====================================================

        torch.save(
            checkpoint,
            latest_checkpoint
        )


        print(
            f"Checkpoint saved: "
            f"{epoch_checkpoint}"
        )


        # ====================================================
        # RESULTS
        # ====================================================

        if opt.train == 0:

            print(
                'p1: %.2f, p2: %.2f'
                % (
                    p1,
                    p2
                )
            )

            break


        else:

            logging.info(
                'epoch: %d, '
                'lr: %.7f, '
                'loss: %.4f, '
                'p1: %.2f, '
                'p2: %.2f, '
                '%d: %.2f'
                % (
                    epoch,
                    lr,
                    loss,
                    p1,
                    p2,
                    best_epoch,
                    opt.previous_best_threshold
                )
            )


            print(
                'e: %d, '
                'lr: %.7f, '
                'loss: %.4f, '
                'p1: %.2f, '
                'p2: %.2f, '
                '%d: %.2f'
                % (
                    epoch,
                    lr,
                    loss,
                    p1,
                    p2,
                    best_epoch,
                    opt.previous_best_threshold
                )
            )


        # ====================================================
        # LEARNING RATE DECAY
        # ====================================================

        if (
            epoch
            % opt.large_decay_epoch
            == 0
        ):

            for param_group in optimizer.param_groups:

                param_group['lr'] *= (
                    opt.lr_decay_large
                )


            lr *= opt.lr_decay_large


        else:

            for param_group in optimizer.param_groups:

                param_group['lr'] *= (
                    opt.lr_decay
                )


            lr *= opt.lr_decay


    # ========================================================
    # FINISHED
    # ========================================================

    print(
        opt.checkpoint
    )


    end_time = time.time()


    print(
        "\n" + "=" * 40
    )

    print(
        "Process Completed"
    )

    print(
        f"Time Cost: "
        f"{(end_time - start_time) / 60:.2f} minutes"
    )


    print(
        "Hardware Info:"
    )


    if torch.cuda.is_available():

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        gpu_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )


        print(
            f"  GPU: "
            f"{gpu_name} "
            f"({gpu_memory:.2f} GB)"
        )


    else:

        print(
            "  GPU: Not available "
            "(running on CPU)"
        )


    print(
        "=" * 40
    )
