import csv
from collections import defaultdict
import os, pickle

def get_mapping():
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

    valid_pkl = os.path.join('ap3d/pose_3d_v3/valid.pkl')
    with open(valid_pkl, 'rb') as f:
        valid_data = pickle.load(f)

    def get_unique_subaction(item):
        image_path = item['image_path']
        parts = image_path.split('/')
        if 'S3' in image_path:
            filename = parts[-1]
            name_parts = filename.split('_')
            idx = 1
            while idx < len(name_parts) and not name_parts[idx].isdigit():
                idx += 1
            if idx < len(name_parts):
                return '_'.join(name_parts[1:idx+1])
        elif 'S2' in image_path:
            vid_idx = int(parts[-1].split('_')[3])
            return f'Running_{vid_idx}'
        return item['subaction']

    subactions = {'S1': set(), 'S2': set(), 'S3': set()}
    for item in valid_data:
        sub = 'S1' if 'S1' in item['image_path'] else ('S2' if 'S2' in item['image_path'] else 'S3')
        subact = get_unique_subaction(item)
        subactions[sub].add(subact)

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

    mapping = {}
    for sub in subactions:
        mapping[sub] = {}
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
            mapping[sub][subact] = {'csv_motion': mapped_motion, 'cam1': cam1, 'cam2': cam2}
            
    with open('data/ap3d_mapping.pkl', 'wb') as f:
        pickle.dump(mapping, f)

get_mapping()
