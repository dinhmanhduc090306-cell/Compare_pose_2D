import os
import copy
import pickle
import numpy as np

from common.cameras import (
    h36m_cameras_intrinsic_params,
    h36m_cameras_extrinsic_params,
    normalize_screen_coordinates,
)


# ============================================================
# SKELETON
# ============================================================

class Skeleton:

    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents, dtype=np.int64)
        self._joints_left = list(joints_left)
        self._joints_right = list(joints_right)

        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def remove_joints(self, joints_to_remove):

        joints_to_remove = set(joints_to_remove)

        valid_joints = [
            i for i in range(len(self._parents))
            if i not in joints_to_remove
        ]

        # Update parents
        for i in range(len(self._parents)):
            while (
                self._parents[i] in joints_to_remove
                and self._parents[i] != -1
            ):
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(
            len(self._parents),
            dtype=np.int64
        )

        new_parents = []

        for i, parent in enumerate(self._parents):

            if i not in joints_to_remove:

                if parent == -1:
                    new_parents.append(-1)
                else:
                    new_parents.append(
                        parent - index_offsets[parent]
                    )

            else:
                index_offsets[i:] += 1

        self._parents = np.array(
            new_parents,
            dtype=np.int64
        )

        if self._joints_left is not None:

            new_joints_left = []

            for joint in self._joints_left:

                if joint in valid_joints:

                    new_joints_left.append(
                        joint - index_offsets[joint]
                    )

            self._joints_left = new_joints_left

        if self._joints_right is not None:

            new_joints_right = []

            for joint in self._joints_right:

                if joint in valid_joints:

                    new_joints_right.append(
                        joint - index_offsets[joint]
                    )

            self._joints_right = new_joints_right

        self._compute_metadata()

        return valid_joints

    def _compute_metadata(self):

        self._has_children = np.zeros(
            len(self._parents),
            dtype=bool
        )

        for i, parent in enumerate(self._parents):

            if parent != -1:

                self._has_children[parent] = True

        self._children = [
            [] for _ in range(len(self._parents))
        ]

        for i, parent in enumerate(self._parents):

            if parent != -1:

                self._children[parent].append(i)


# ============================================================
# HUMAN3.6M SKELETON
# ============================================================

h36m_skeleton = Skeleton(
    parents=[
        -1, 0, 1, 2, 3, 4,
        0, 6, 7, 8, 9,
        0, 11, 12, 13, 14,
        12, 16, 17, 18, 19, 20,
        19, 22, 12, 24, 25, 26, 27,
        28, 27, 30
    ],

    joints_left=[
        6, 7, 8, 9, 10,
        16, 17, 18, 19, 20, 21,
        22, 23
    ],

    joints_right=[
        1, 2, 3, 4, 5,
        24, 25, 26, 27, 28, 29,
        30, 31
    ]
)


# ============================================================
# BASE MOCAP DATASET
# ============================================================

class MocapDataset:

    def __init__(self, fps, skeleton):

        self._skeleton = skeleton
        self._fps = fps

        self._data = {}
        self._cameras = {}

    def remove_joints(self, joints_to_remove):

        kept_joints = self._skeleton.remove_joints(
            joints_to_remove
        )

        for subject in self._data.keys():

            for action in self._data[subject].keys():

                sequence = self._data[subject][action]

                positions = sequence['positions']

                sequence['positions'] = positions[
                    :,
                    kept_joints
                ]

    def __getitem__(self, key):

        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def subjects(self):

        return self._data.keys()

    def fps(self):

        return self._fps

    def skeleton(self):

        return self._skeleton

    def cameras(self):

        return self._cameras

    def supports_semi_supervised(self):

        return False


# ============================================================
# CAMERA PROCESSING
# ============================================================

def process_h36m_cameras(opt):

    cameras = copy.deepcopy(
        h36m_cameras_extrinsic_params
    )

    for subject, subject_cameras in cameras.items():

        for i, cam in enumerate(subject_cameras):

            cam.update(
                h36m_cameras_intrinsic_params[i]
            )

            for k, v in cam.items():

                if k not in [
                    'id',
                    'res_w',
                    'res_h'
                ]:

                    cam[k] = np.array(
                        v,
                        dtype=np.float32
                    )

            if opt.crop_uv == 0:

                cam['center'] = normalize_screen_coordinates(
                    cam['center'],
                    w=cam['res_w'],
                    h=cam['res_h']
                ).astype(np.float32)

                cam['focal_length'] = (
                    cam['focal_length']
                    / cam['res_w']
                    * 2
                )

            if 'translation' in cam:

                cam['translation'] = (
                    cam['translation'] / 1000
                )

            cam['intrinsic'] = np.concatenate(
                (
                    cam['focal_length'],
                    cam['center'],
                    cam['radial_distortion'],
                    cam['tangential_distortion']
                )
            )

    return cameras


# ============================================================
# HUMAN36M DATASET
# ============================================================

class Human36mDataset(MocapDataset):

    """
    Dataset loader for the converted belief_data dataset.

    Expected NPZ structure:

        positions_3d
            video_0_seg_1
                camera0 -> (T,17,3)
                camera1 -> (T,17,3)
                ...

            video_0_seg_2
                ...

    IMPORTANT:

    These are NOT Human3.6M subjects.

    Therefore we must NOT do:

        self._cameras[subject]

    because:

        subject = 'video_0_seg_1'

    does not exist inside the original H36M camera dictionary.
    """

    def __init__(
        self,
        path,
        opt,
        remove_static_joints=False
    ):

        super().__init__(
            fps=50,
            skeleton=h36m_skeleton
        )

        # ----------------------------------------------------
        # Original H36M train/test lists
        #
        # Keep these because main_img.py expects them.
        # ----------------------------------------------------

        self.train_list = [
            '1',
            '5',
            '6',
            '7',
            '8'
        ]

        self.test_list = [
            '9',
            '11'
        ]

        # ----------------------------------------------------
        # Load original H36M cameras
        # ----------------------------------------------------

        self._cameras = process_h36m_cameras(opt)

        # ----------------------------------------------------
        # Load converted belief dataset
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("Loading converted belief_data dataset")
        print("=" * 70)

        print(
            f"Dataset path:\n{path}"
        )

        loaded = np.load(
            path,
            allow_pickle=True
        )

        if 'positions_3d' not in loaded:

            raise KeyError(
                "NPZ file does not contain "
                "'positions_3d'."
            )

        data = loaded[
            'positions_3d'
        ].item()

        print(
            f"Loaded sequences: {len(data)}"
        )

        # ----------------------------------------------------
        # Detect custom sequence format
        # ----------------------------------------------------

        custom_sequences = False

        for key in data.keys():

            if str(key).startswith(
                'video_'
            ):

                custom_sequences = True
                break

        # ====================================================
        # CUSTOM BELIEF DATA
        # ====================================================

        if custom_sequences:

            print(
                "Detected custom "
                "video_*_seg_* dataset."
            )

            print(
                "Using sequence-specific "
                "camera placeholders."
            )

            self._data = {}

            for sequence_name, camera_dict in data.items():

                sequence_name = str(
                    sequence_name
                )

                self._data[
                    sequence_name
                ] = {}

                # ------------------------------------------------
                # Each sequence becomes one "action"
                #
                # Example:
                #
                # self._data['video_0_seg_1']['default']
                # ------------------------------------------------

                self._data[
                    sequence_name
                ]['default'] = {
                    'positions': camera_dict,
                    'cameras': self._make_custom_cameras(
                        camera_dict
                    )
                }

            # ----------------------------------------------------
            # Important:
            #
            # main_img.py/Fusion may use dataset.subjects()
            # to determine train/test.
            #
            # We therefore expose ALL converted sequences.
            # ----------------------------------------------------

            self.train_list = list(
                self._data.keys()
            )

            self.test_list = []

            print(
                f"Custom sequences loaded: "
                f"{len(self.train_list)}"
            )

        # ====================================================
        # ORIGINAL H36M FORMAT
        # ====================================================

        else:

            print(
                "Detected standard "
                "Human3.6M dataset format."
            )

            self._data = {}

            for subject, actions in data.items():

                subject = str(subject)

                self._data[
                    subject
                ] = {}

                for action_name, positions in actions.items():

                    self._data[
                        subject
                    ][action_name] = {
                        'positions': positions,
                        'cameras': self._cameras[
                            subject
                        ]
                    }

        # ----------------------------------------------------
        # Optional static joint removal
        # ----------------------------------------------------

        if remove_static_joints:

            self.remove_joints(
                [
                    4, 5,
                    9, 10,
                    11,
                    16,
                    20, 21, 22, 23, 24,
                    28, 29, 30, 31
                ]
            )

            if self._skeleton.num_joints() > 14:

                self._skeleton._parents[11] = 8
                self._skeleton._parents[14] = 8

        print()
        print(
            "Dataset initialization complete."
        )

        print(
            f"Number of subjects/sequences: "
            f"{len(self._data)}"
        )

        print(
            "=" * 70
        )
        print()

    # ========================================================
    # CUSTOM CAMERA GENERATOR
    # ========================================================

    def _make_custom_cameras(
        self,
        camera_dict
    ):

        """
        Create camera entries for custom
        video_*_seg_* sequences.

        The actual 3D coordinates are already
        camera-space coordinates.

        Therefore these cameras are mainly
        placeholders required by the DSVTformer
        dataset interface.
        """

        cameras = []

        camera_names = sorted(
            camera_dict.keys()
        )

        for i, camera_name in enumerate(
            camera_names
        ):

            positions = camera_dict[
                camera_name
            ]

            # ------------------------------------------------
            # Default image resolution.
            #
            # This can be changed if your actual
            # image resolution is known.
            # ------------------------------------------------

            res_w = 288
            res_h = 384

            cam = {

                'id':
                    camera_name,

                'res_w':
                    res_w,

                'res_h':
                    res_h,

                'center':
                    np.array(
                        [
                            res_w / 2,
                            res_h / 2
                        ],
                        dtype=np.float32
                    ),

                'focal_length':
                    np.array(
                        [
                            res_w,
                            res_w
                        ],
                        dtype=np.float32
                    ),

                'radial_distortion':
                    np.zeros(
                        3,
                        dtype=np.float32
                    ),

                'tangential_distortion':
                    np.zeros(
                        2,
                        dtype=np.float32
                    ),

                'translation':
                    np.zeros(
                        3,
                        dtype=np.float32
                    ),

                'rotation':
                    np.eye(
                        3,
                        dtype=np.float32
                    ),

                'intrinsic':
                    np.concatenate(
                        [
                            np.array(
                                [
                                    res_w,
                                    res_w
                                ],
                                dtype=np.float32
                            ),

                            np.array(
                                [
                                    res_w / 2,
                                    res_h / 2
                                ],
                                dtype=np.float32
                            ),

                            np.zeros(
                                3,
                                dtype=np.float32
                            ),

                            np.zeros(
                                2,
                                dtype=np.float32
                            )
                        ]
                    )
            }

            cameras.append(
                cam
            )

        return cameras

    # ========================================================
    # SEMI-SUPERVISED
    # ========================================================

    def supports_semi_supervised(self):

        return True


# ============================================================
# AP3D DATASET
# ============================================================

class AP3DDataset(MocapDataset):

    """
    AP3D dataset loader.

    Uses pre-generated camera parameters:

        ap3d_cameras.pkl
    """

    ap3d_skeleton = Skeleton(

        parents=[
            -1, 0, 1, 2, 3, 4,
            0, 6, 7, 8, 9,
            0, 11, 12, 13, 14,
            12, 16, 17, 18, 19, 20,
            19, 22, 12, 24, 25, 26, 27,
            28, 27, 30
        ],

        joints_left=[
            6, 7, 8, 9, 10,
            16, 17, 18, 19, 20, 21,
            22, 23
        ],

        joints_right=[
            1, 2, 3, 4, 5,
            24, 25, 26, 27, 28, 29,
            30, 31
        ]
    )

    def __init__(
        self,
        path,
        opt,
        remove_static_joints=False
    ):

        super().__init__(
            fps=50,
            skeleton=AP3DDataset.ap3d_skeleton
        )

        self.train_list = []

        self.test_list = [
            'S1',
            'S2',
            'S3'
        ]

        # ----------------------------------------------------
        # Camera parameters
        # ----------------------------------------------------

        cam_params_path = os.path.join(
            os.path.dirname(path),
            'ap3d_cameras.pkl'
        )

        if not os.path.exists(
            cam_params_path
        ):

            raise FileNotFoundError(
                "AP3D camera file not found:\n"
                f"{cam_params_path}"
            )

        with open(
            cam_params_path,
            'rb'
        ) as f:

            ap3d_cameras = pickle.load(f)

        self._cameras = copy.deepcopy(
            ap3d_cameras
        )

        # ----------------------------------------------------
        # Process cameras
        # ----------------------------------------------------

        for subject, subject_cameras in self._cameras.items():

            if isinstance(
                subject_cameras,
                dict
            ):

                for action_name, cameras in subject_cameras.items():

                    for i, cam in enumerate(cameras):

                        self._process_ap3d_camera(
                            cam,
                            opt
                        )

            else:

                for i, cam in enumerate(
                    subject_cameras
                ):

                    self._process_ap3d_camera(
                        cam,
                        opt
                    )

        # ----------------------------------------------------
        # Load positions
        # ----------------------------------------------------

        loaded = np.load(
            path,
            allow_pickle=True
        )

        data = loaded[
            'positions_3d'
        ].item()

        self._data = {}

        for subject, actions in data.items():

            subject = str(subject)

            self._data[
                subject
            ] = {}

            for action_name, positions in actions.items():

                if isinstance(
                    self._cameras.get(subject),
                    dict
                ):

                    cameras = self._cameras[
                        subject
                    ][action_name]

                else:

                    cameras = self._cameras[
                        subject
                    ]

                self._data[
                    subject
                ][action_name] = {

                    'positions':
                        positions,

                    'cameras':
                        cameras
                }

        if remove_static_joints:

            self.remove_joints(
                [
                    4, 5,
                    9, 10,
                    11,
                    16,
                    20, 21, 22, 23, 24,
                    28, 29, 30, 31
                ]
            )

    # ========================================================
    # AP3D CAMERA PROCESSING
    # ========================================================

    def _process_ap3d_camera(
        self,
        cam,
        opt
    ):

        for k, v in cam.items():

            if k not in [
                'id',
                'res_w',
                'res_h'
            ]:

                cam[k] = np.array(
                    v,
                    dtype=np.float32
                )

        if opt.crop_uv == 0:

            cam['center'] = normalize_screen_coordinates(
                cam['center'],
                w=cam['res_w'],
                h=cam['res_h']
            ).astype(np.float32)

            cam['focal_length'] = (
                cam['focal_length']
                / cam['res_w']
                * 2
            )

        if 'translation' in cam:

            cam['translation'] = (
                cam['translation'] / 1000
            )

        cam['intrinsic'] = np.concatenate(
            (
                cam['focal_length'],
                cam['center'],
                cam['radial_distortion'],
                cam['tangential_distortion']
            )
        )

    def supports_semi_supervised(self):

        return True
