import os, pickle, cv2, sys, glob, json

import torch, faulthandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# from model.backbones import hrnet, cpn3, sh, simcc
from networks.network import CPN101
from einops import rearrange

import multiprocessing
import threading
from pathlib import Path

folder_list = {'train': ['S1', 'S5', 'S6', 'S7', 'S8'],
               'test': ['S9', 'S11']}

camera_ids = {
    0: '.54138969',
    1: '.55011271',
    2: '.58860488',
    3: '.60457274'
}
camera_ids2 = {
    '54138969': 0,
    '55011271': 1,
    '58860488': 2,
    '60457274': 3
}
out_data = {}

class GenSequenceDataset(Dataset):
    def __init__(self,
                 data_dir='/home/xxxxxx/MotionAGFormer/data/motion3d/3DHPImages_81_81',
                 images_dir='/home/xxxxxx/data-home/MPI_INF_3DHP/images_384_288',
                 detector_type='hrnet'):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.detector_type = detector_type

        self.hrnet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.hrnet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.sh_mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float32)
        self.sh_std = np.array([0.2458, 0.2410, 0.2468], dtype=np.float32)
        self.simcc_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.simcc_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.image_info = []
        self.last_valid_img = None

        # keypoints = np.load("/data2/zhangwr/stcformer/dataset/data_2d_h36m_cpn_ft_h36m_dbb.npz", allow_pickle=True)
        # self.dst_data = keypoints['positions_2d'].item()
        # for subject in folder_list['train'] + folder_list['test']:
        #     for action in self.dst_data[subject].keys():
        #         for cam_idx in range(4):
        #             frame_dst = len(self.dst_data[subject][action][cam_idx])
        #             for f in range(frame_dst):

        #                 self.image_info.append({
        #                     'subject': subject,
        #                     'action': action,
        #                     'camera': cam_idx,
        #                     'frame_idx': f
        #                 })
        
        # Find all your subject JSON files
        json_files = glob.glob('/home/levantuananh/h36m/annotations/Human36M_subject*_data.json')

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for item in data['images']:
                subject = str(item['subject'])
                
                action_base = item['action_name']
                subaction = item['subaction_idx']
                action = f"{action_base} {subaction}" 
                
                camera = item['cam_idx'] - 1
                
                # --- ADD THIS: Skip cameras 1 and 3 entirely! ---
                if camera not in [0, 1]:
                    continue
                # ------------------------------------------------
                
                frame_idx = item['frame_idx']
                
                self.image_info.append({
                    'subject': subject,
                    'action': action,
                    'camera': camera,
                    'frame_idx': frame_idx,
                    'file_name': item['file_name']
                })

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        info = self.image_info[index]
        subject = info['subject']
        action = info['action']
        camera = info['camera']
        frame_idx = info['frame_idx']

        # img_rel_path = f"{subject}/{action}{camera_ids[camera]}/{frame_idx + 1:04d}.jpg"
        img_rel_path = info['file_name']
        image_full_path = os.path.join(self.images_dir, img_rel_path)

        img = None
        if os.path.exists(image_full_path):
            img = cv2.imread(image_full_path, cv2.IMREAD_COLOR)

        if img is None:

            if self.last_valid_img is not None:

                img = self.last_valid_img.copy()
            else:
                raise ValueError(f"[FATAL] 第一次图像读取就失败，没有上一张可替代: {image_full_path}")

        else:
            self.last_valid_img = img.copy()

        image = cv2.resize(img, (288, 384))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        if self.detector_type == 'hrnet':
            image = (image - self.hrnet_mean) / self.hrnet_std
        elif self.detector_type == 'sh':
            image = (image - self.sh_mean) / self.sh_std
        elif self.detector_type == 'simcc':
            image = (image - self.simcc_mean) / self.simcc_std
        image = image.transpose((2, 0, 1))

        return image, info, index

if __name__ == '__main__':
    faulthandler.enable()

    detector_type = 'cpn'
    write_dir = 'data/'
    pretrained_dir = 'data/pretrained'

    data_dir = '/home/levantuananh/h36m/images'

    os.makedirs(write_dir, exist_ok=True)

    # if detector_type == 'hrnet':
    #     model = hrnet.build_model('hrnet_w32_256_192')
    #     model = hrnet.load_model(model, f'{pretrained_dir}/coco/pose_hrnet_w32_256x192.pth')
    # elif detector_type == 'cpn':

    #     model = cpn3.build_model('cpn_101_384_288')

    #     model = cpn3.load_model(model, f'{pretrained_dir}/cpn_101_384x288.pth.tar')
    # elif detector_type == 'simcc':
    #     model = simcc.build_model('pose_resnet_101_384_288')
    #     model = simcc.load_model(model, f'{pretrained_dir}/pose_resnet_101_384x288.pth')
    # else:
    #     raise ValueError

    model = CPN101(out_size=(96, 72), num_class=17, pretrained=False) 
    checkpoint = torch.load(f'{pretrained_dir}/cpn_101_384x288.pth.tar', map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')
    else:
        # print("GPU not available, using CPU instead.")
        # device = torch.device('cpu')
        exit()

    model = model.to(device)
    model.eval()

    dataset = GenSequenceDataset(
        data_dir=data_dir,
        images_dir=data_dir,
        detector_type=detector_type)

    dataloader = DataLoader(dataset, num_workers=8, persistent_workers=True, shuffle=False, batch_size=16)
    features_dict = {}
    for batch in tqdm(dataloader):
        images, infos, index = batch
        images = images.to(device)

        with torch.no_grad():
            # f_maps = model(images)
            # f_maps = f_maps[0]
            
            f_maps_list = model(images)
            f_maps = f_maps_list[-1]
            f_maps = f_maps.mean(dim=[2, 3])

        f_maps = f_maps.cpu().numpy()

        for i in range(len(infos['subject'])):
            subject = str(infos['subject'][i])
            action = str(infos['action'][i])
            if action == "Directions 2" and subject == '11':
                break
            camera = infos['camera'][i].item()
            if subject not in features_dict.keys():
                features_dict[subject] = {}
            if action not in features_dict[subject].keys():
                features_dict[subject][action] = [[], [], [], []]
            # features_dict[subject][action][camera].append(f_maps[i])
            
            # --- VIEW PADDING LOGIC ---
            # 0 duplicates to 0 and 1
            if camera == 0:
                features_dict[subject][action][0].append(f_maps[i])
                features_dict[subject][action][2].append(f_maps[i])
            # 1 duplicates to 2 and 3
            elif camera == 1:
                features_dict[subject][action][1].append(f_maps[i])
                features_dict[subject][action][3].append(f_maps[i])
            # --------------------------

    for subject in features_dict:
        for action in features_dict[subject]:
            for cam in range(4):
                features_dict[subject][action][cam] = np.stack(features_dict[subject][action][cam], axis=0)

    out_file = os.path.join(write_dir, 'img_features.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Features saved to {out_file}")
