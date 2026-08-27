import os
import pickle
import numpy as np
import torch
import torch.utils.data as data

from common.cameras import normalize_screen_coordinates


# ============================================================
# CONFIGURATION
# ============================================================

# Your belief dataset has 9 cameras:
#
# camera0
# camera1
# ...
# camera8
#
# DSVTformer was originally designed around 4 views.
#
# Change this if you want another subset.
BELIEF_CAMERA_IDS = [0, 1, 2, 3]

# Dummy image-feature dimension.
#
# IMPORTANT:
# Your belief dataset contains pose coordinates but NO RGB images
# and therefore no CPN image features.
#
# If the model expects CPN features, this dimension may need to
# match the feature dimension expected by your model.
#
# 2048 is the common final feature dimension for many ResNet/CPN
# configurations.
DUMMY_IMAGE_FEATURE_DIM = 2048

# If True, create zero-valued image features.
#
# This allows the existing main_img.py interface to continue
# working without RGB images.
USE_DUMMY_IMAGE_FEATURES = True


# ============================================================
# CHUNKED GENERATOR
# ============================================================

class ChunkedGenerator:

    def __init__(
        self,
        batch_size,
        cameras,
        poses_3d,
        poses_2d,
        images,
        chunk_length=1,
        pad=0,
        causal_shift=0,
        shuffle=False,
        random_seed=1234,
        augment=False,
        reverse_aug=False,
        kps_left=None,
        kps_right=None,
        joints_left=None,
        joints_right=None,
        endless=False,
        out_all=False
    ):

        assert poses_3d is None or (
            len(poses_3d) == len(poses_2d)
            and len(poses_3d) == len(images)
        ), (
            len(poses_3d),
            len(poses_2d),
            len(images)
        )

        assert cameras is None or (
            len(cameras) == len(poses_2d)
            and len(cameras) == len(images)
        ), (
            len(cameras),
            len(poses_2d),
            len(images)
        )

        pairs = []

        self.saved_index = {}

        start_index = 0

        # ----------------------------------------------------
        # Build sequence chunks
        # ----------------------------------------------------

        for key in poses_2d.keys():

            seq_2d = poses_2d[key]

            if poses_3d is not None:
                seq_3d = poses_3d[key]

                assert seq_2d.shape[0] == seq_3d.shape[0], (
                    f"Frame mismatch for {key}: "
                    f"2D={seq_2d.shape[0]}, "
                    f"3D={seq_3d.shape[0]}"
                )

            n_chunks = (
                seq_2d.shape[0] + chunk_length - 1
            ) // chunk_length

            offset = (
                n_chunks * chunk_length
                - seq_2d.shape[0]
            ) // 2

            bounds = (
                np.arange(n_chunks + 1)
                * chunk_length
                - offset
            )

            augment_vector = np.full(
                len(bounds) - 1,
                False,
                dtype=bool
            )

            reverse_augment_vector = np.full(
                len(bounds) - 1,
                False,
                dtype=bool
            )

            # key is (subject, action)
            keys = np.tile(
                np.array(key, dtype=object).reshape(1, 2),
                (len(bounds) - 1, 1)
            )

            pairs += list(
                zip(
                    keys,
                    bounds[:-1],
                    bounds[1:],
                    augment_vector,
                    reverse_augment_vector
                )
            )

            if reverse_aug:

                pairs += list(
                    zip(
                        keys,
                        bounds[:-1],
                        bounds[1:],
                        augment_vector,
                        ~reverse_augment_vector
                    )
                )

            if augment:

                if reverse_aug:

                    pairs += list(
                        zip(
                            keys,
                            bounds[:-1],
                            bounds[1:],
                            ~augment_vector,
                            ~reverse_augment_vector
                        )
                    )

                else:

                    pairs += list(
                        zip(
                            keys,
                            bounds[:-1],
                            bounds[1:],
                            ~augment_vector,
                            reverse_augment_vector
                        )
                    )

            if poses_3d is not None:
                end_index = (
                    start_index
                    + poses_3d[key].shape[0]
                )

                self.saved_index[key] = [
                    start_index,
                    end_index
                ]

                start_index = end_index

        # ----------------------------------------------------
        # Determine dimensions
        # ----------------------------------------------------

        first_key = next(iter(poses_2d.keys()))

        # cameras:
        #
        # Usually:
        #     cameras[(subject, action)]
        # = camera information
        #
        # We do not actually require camera parameters for the
        # belief dataset.
        # ----------------------------------------------------

        if cameras is not None:

            try:

                camera_dim = cameras[first_key].shape[-1]

            except Exception:

                camera_dim = 9

            self.batch_cam = np.empty(
                (
                    batch_size,
                    camera_dim
                ),
                dtype=np.float32
            )

        else:

            self.batch_cam = None

        # ----------------------------------------------------
        # 3D
        # ----------------------------------------------------

        if poses_3d is not None:

            self.batch_3d = np.empty(
                (
                    batch_size,
                    chunk_length,
                    poses_3d[first_key].shape[-2],
                    poses_3d[first_key].shape[-1]
                ),
                dtype=np.float32
            )

        else:

            self.batch_3d = None

        # ----------------------------------------------------
        # 2D
        #
        # Expected:
        #
        # (T, views, joints, 2)
        # ----------------------------------------------------

        self.batch_2d = np.empty(
            (
                batch_size,
                chunk_length + 2 * pad,
                poses_2d[first_key].shape[-3],
                poses_2d[first_key].shape[-2],
                poses_2d[first_key].shape[-1]
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Image features
        #
        # Expected:
        #
        # (T, views, feature_dim)
        # ----------------------------------------------------

        self.batch_images = np.empty(
            (
                batch_size,
                chunk_length + 2 * pad,
                images[first_key].shape[-2],
                images[first_key].shape[-1]
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # General state
        # ----------------------------------------------------

        self.num_batches = (
            len(pairs) + batch_size - 1
        ) // batch_size

        self.batch_size = batch_size

        self.random = np.random.RandomState(
            random_seed
        )

        self.pairs = pairs

        self.shuffle = shuffle

        self.pad = pad

        self.causal_shift = causal_shift

        self.endless = endless

        self.state = None

        self.cameras = cameras

        self.poses_3d = poses_3d

        self.poses_2d = poses_2d

        self.images = images

        self.augment = augment

        self.kps_left = kps_left

        self.kps_right = kps_right

        self.joints_left = joints_left

        self.joints_right = joints_right

        self.out_all = out_all

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    def num_frames(self):

        return (
            self.num_batches
            * self.batch_size
        )

    def random_state(self):

        return self.random

    def set_random_state(self, random):

        self.random = random

    def augment_enabled(self):

        return self.augment

    # --------------------------------------------------------
    # Pairs
    # --------------------------------------------------------

    def next_pairs(self):

        if self.state is None:

            if self.shuffle:

                pairs = self.random.permutation(
                    self.pairs
                )

            else:

                pairs = self.pairs

            return 0, pairs

        else:

            return self.state

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    def get_batch(
        self,
        seq_i,
        start_3d,
        end_3d,
        mask,
        flip,
        reverse
    ):

        subject, action = seq_i

        seq_name = (
            subject,
            action
        )

        start_2d = (
            start_3d
            - self.pad
            - self.causal_shift
        )

        end_2d = (
            end_3d
            + self.pad
            - self.causal_shift
        )

        seq_2d = self.poses_2d[
            seq_name
        ]

        seq_img = self.images[
            seq_name
        ]

        # ----------------------------------------------------
        # Ensure lengths match
        # ----------------------------------------------------

        assert len(seq_2d) == len(seq_img), (
            f"Length mismatch: "
            f"poses_2d={len(seq_2d)}, "
            f"images={len(seq_img)}, "
            f"sequence={seq_name}"
        )

        # ----------------------------------------------------
        # 2D range
        # ----------------------------------------------------

        low_2d = max(
            start_2d,
            0
        )

        high_2d = min(
            end_2d,
            seq_2d.shape[0]
        )

        pad_left_2d = (
            low_2d
            - start_2d
        )

        pad_right_2d = (
            end_2d
            - high_2d
        )

        if (
            pad_left_2d != 0
            or pad_right_2d != 0
        ):

            self.batch_2d = np.pad(
                seq_2d[
                    low_2d:high_2d
                ],
                (
                    (pad_left_2d, pad_right_2d),
                    (0, 0),
                    (0, 0),
                    (0, 0)
                ),
                mode='edge'
            )

            self.batch_images = np.pad(
                seq_img[
                    low_2d:high_2d
                ],
                (
                    (pad_left_2d, pad_right_2d),
                    (0, 0),
                    (0, 0)
                ),
                mode='edge'
            )

        else:

            self.batch_2d = seq_2d[
                low_2d:high_2d
            ]

            self.batch_images = seq_img[
                low_2d:high_2d
            ]

        # ----------------------------------------------------
        # 3D
        # ----------------------------------------------------

        if self.poses_3d is not None:

            seq_3d = self.poses_3d[
                seq_name
            ].copy()

            if self.out_all:

                low_3d = low_2d

                high_3d = high_2d

                pad_left_3d = pad_left_2d

                pad_right_3d = pad_right_2d

            else:

                low_3d = max(
                    start_3d,
                    0
                )

                high_3d = min(
                    end_3d,
                    seq_3d.shape[0]
                )

                pad_left_3d = (
                    low_3d
                    - start_3d
                )

                pad_right_3d = (
                    end_3d
                    - high_3d
                )

            if (
                pad_left_3d != 0
                or pad_right_3d != 0
            ):

                self.batch_3d = np.pad(
                    seq_3d[
                        low_3d:high_3d
                    ],
                    (
                        (pad_left_3d, pad_right_3d),
                        (0, 0),
                        (0, 0)
                    ),
                    mode='edge'
                )

            else:

                self.batch_3d = seq_3d[
                    low_3d:high_3d
                ]

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        # Belief dataset does not currently provide real
        # camera calibration parameters.
        #
        # Returning None is intentional.
        camera_output = None

        if (
            self.poses_3d is not None
        ):

            return (
                camera_output,
                self.batch_3d.copy(),
                self.batch_2d.copy(),
                self.batch_images.copy(),
                action,
                subject,
                low_2d,
                high_2d
            )

        return (
            camera_output,
            None,
            self.batch_2d.copy(),
            self.batch_images.copy(),
            action,
            subject
        )


# ============================================================
# BELIEF DATASET HELPERS
# ============================================================

def _extract_sequence_id(sequence_name):

    """
    Convert:

        video_0_seg_1

    into:

        ('video_0', 'seg_1')

    This keeps compatibility with the original DSVTformer
    (subject, action) indexing.
    """

    parts = sequence_name.split("_")

    if len(parts) >= 4:

        video_id = parts[1]

        segment_id = parts[3]

        return (
            f"video_{video_id}",
            f"seg_{segment_id}"
        )

    return (
        sequence_name,
        "default"
    )


def _load_npz_dict(path, key):

    print(
        f"\nLoading:\n{path}"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    data = np.load(
        path,
        allow_pickle=True
    )

    if key not in data:

        raise KeyError(
            f"'{key}' not found in {path}. "
            f"Available keys: {data.files}"
        )

    value = data[key].item()

    print(
        f"Loaded {len(value)} sequences."
    )

    return value


def _find_belief_file(root_path, filename):

    """
    Try several common root_path layouts.
    """

    candidates = [

        os.path.join(
            root_path,
            filename
        ),

        os.path.join(
            root_path,
            "belief_data",
            filename
        ),

        os.path.join(
            "/content/drive/MyDrive/belief_data",
            filename
        ),

        os.path.join(
            "/content/Compare_pose_2D",
            "data",
            filename
        )
    ]

    for path in candidates:

        if os.path.exists(path):

            return path

    raise FileNotFoundError(
        "\nCould not locate:\n"
        f"    {filename}\n\n"
        "Tried:\n"
        + "\n".join(
            f"    {p}"
            for p in candidates
        )
    )


def _convert_nested_camera_dict(
    data,
    camera_ids
):

    """
    Convert:

        data[sequence]['camera0']

    into the list:

        [camera0, camera1, camera2, camera3]

    """

    output = {}

    for sequence_name in data:

        sequence = data[
            sequence_name
        ]

        camera_arrays = []

        for cam_id in camera_ids:

            camera_name = (
                f"camera{cam_id}"
            )

            if camera_name not in sequence:

                raise KeyError(
                    f"{sequence_name} does not contain "
                    f"{camera_name}. Available cameras: "
                    f"{list(sequence.keys())}"
                )

            arr = np.asarray(
                sequence[camera_name],
                dtype=np.float32
            )

            camera_arrays.append(
                arr
            )

        output[
            sequence_name
        ] = camera_arrays

    return output


def _stack_views(
    camera_data
):

    """
    Convert:

        list of 4 arrays
        each = (T,J,C)

    into:

        (T,4,J,C)
    """

    if len(camera_data) == 0:

        raise ValueError(
            "No camera data."
        )

    lengths = [
        arr.shape[0]
        for arr in camera_data
    ]

    min_len = min(
        lengths
    )

    camera_data = [
        arr[:min_len]
        for arr in camera_data
    ]

    return np.stack(
        camera_data,
        axis=1
    ).astype(
        np.float32
    )


# ============================================================
# FUSION DATASET
# ============================================================

class Fusion(data.Dataset):

    """
    Fusion loader adapted for the belief dataset.

    Original DSVTformer expected:

        subject
            action
                4 cameras

    Belief dataset provides:

        video_X_seg_Y
            camera0
            ...
            camera8

    This implementation converts the selected cameras into:

        (T, 4, 17, 2)

    for 2D input.

    3D is:

        (T, 17, 3)

    Image features are dummy zero features because the belief
    dataset contains no RGB images.
    """

    def __init__(
        self,
        opt,
        dataset,
        root_path,
        train=True
    ):

        self.data_type = opt.dataset

        self.train = train

        self.keypoints_name = opt.keypoints

        self.root_path = root_path

        # ----------------------------------------------------
        # Determine split
        # ----------------------------------------------------

        self.train_list = (
            opt.subjects_train.split(",")
        )

        self.test_list = (
            opt.subjects_test.split(",")
        )

        self.action_filter = (
            None
            if opt.actions == '*'
            else opt.actions.split(',')
        )

        self.downsample = opt.downsample

        self.subset = opt.subset

        self.stride = opt.stride

        self.crop_uv = opt.crop_uv

        self.test_aug = opt.test_augmentation

        self.pad = opt.pad

        # ----------------------------------------------------
        # For belief data, the dataset passed by main_img.py
        # is already a Human36mDataset-compatible object after
        # we replace its loader.
        #
        # We nevertheless identify the actual sequences here.
        # ----------------------------------------------------

        if self.train:

            selected = self.train_list

        else:

            selected = self.test_list

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        self.keypoints, self.img = (
            self.prepare_data(
                dataset,
                selected
            )
        )

        (
            self.cameras_train,
            self.poses_train,
            self.poses_train_2d,
            self.images_train
        ) = self.fetch(
            dataset,
            selected,
            subset=self.subset
        )

        if self.train:

            self.generator = ChunkedGenerator(
                opt.batch_size // opt.stride,
                self.cameras_train,
                self.poses_train,
                self.poses_train_2d,
                self.images_train,
                self.stride,
                pad=self.pad,
                augment=opt.data_augmentation,
                reverse_aug=opt.reverse_augmentation,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right,
                out_all=opt.out_all
            )

            print(
                "INFO: Training on {} frames".format(
                    self.generator.num_frames()
                )
            )

        else:

            self.cameras_test = (
                self.cameras_train
            )

            self.poses_test = (
                self.poses_train
            )

            self.poses_test_2d = (
                self.poses_train_2d
            )

            self.images_test = (
                self.images_train
            )

            self.generator = ChunkedGenerator(
                opt.batch_size // opt.stride,
                self.cameras_test,
                self.poses_test,
                self.poses_test_2d,
                self.images_test,
                chunk_length=1,
                pad=self.pad,
                augment=False,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right
            )

            self.key_index = (
                self.generator.saved_index
            )

            print(
                "INFO: Testing on {} frames".format(
                    self.generator.num_frames()
                )
            )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_data(
        self,
        dataset,
        folder_list
    ):

        # ----------------------------------------------------
        # Keypoint symmetry
        # ----------------------------------------------------

        if (
            hasattr(dataset, "skeleton")
        ):

            self.kps_left = list(
                dataset.skeleton().joints_left()
            )

            self.kps_right = list(
                dataset.skeleton().joints_right()
            )

            self.joints_left = list(
                dataset.skeleton().joints_left()
            )

            self.joints_right = list(
                dataset.skeleton().joints_right()
            )

        else:

            # H36M 32-joint symmetry
            self.kps_left = [
                6, 7, 8, 9, 10,
                16, 17, 18, 19,
                20, 21, 22, 23
            ]

            self.kps_right = [
                1, 2, 3, 4, 5,
                24, 25, 26, 27,
                28, 29, 30, 31
            ]

            self.joints_left = (
                self.kps_left
            )

            self.joints_right = (
                self.kps_right
            )

        # ----------------------------------------------------
        # Load normalized 2D belief data
        # ----------------------------------------------------

        try:

            keypoints_path = (
                _find_belief_file(
                    self.root_path,
                    "data_2d_h36m_cpn_ft_h36m_dbb.npz"
                )
            )

        except FileNotFoundError:

            keypoints_path = (
                _find_belief_file(
                    self.root_path,
                    "data_2d_belief.npz"
                )
            )

        keypoints_raw = np.load(
            keypoints_path,
            allow_pickle=True
        )

        print(
            "\n2D NPZ keys:",
            keypoints_raw.files
        )

        keypoints = (
            keypoints_raw[
                "positions_2d"
            ].item()
        )

        # ----------------------------------------------------
        # Detect metadata
        # ----------------------------------------------------

        if "metadata" in keypoints_raw.files:

            try:

                metadata = (
                    keypoints_raw[
                        "metadata"
                    ].item()
                )

                if (
                    "keypoints_symmetry"
                    in metadata
                ):

                    symmetry = (
                        metadata[
                            "keypoints_symmetry"
                        ]
                    )

                    self.kps_left = list(
                        symmetry[0]
                    )

                    self.kps_right = list(
                        symmetry[1]
                    )

            except Exception as e:

                print(
                    "WARNING: Could not read 2D metadata:",
                    e
                )

        # ----------------------------------------------------
        # Convert belief sequence structure
        # ----------------------------------------------------

        converted_keypoints = {}

        for sequence_name in keypoints:

            if sequence_name not in dataset:

                continue

            camera_dict = (
                keypoints[
                    sequence_name
                ]
            )

            camera_arrays = []

            for cam_id in BELIEF_CAMERA_IDS:

                camera_name = (
                    f"camera{cam_id}"
                )

                if (
                    camera_name
                    not in camera_dict
                ):

                    raise KeyError(
                        f"Missing {camera_name} "
                        f"in {sequence_name}"
                    )

                arr = np.asarray(
                    camera_dict[
                        camera_name
                    ],
                    dtype=np.float32
                )

                camera_arrays.append(
                    arr
                )

            converted_keypoints[
                sequence_name
            ] = camera_arrays

        # ----------------------------------------------------
        # Build dummy image features
        # ----------------------------------------------------

        img = {}

        for sequence_name in converted_keypoints:

            camera_arrays = (
                converted_keypoints[
                    sequence_name
                ]
            )

            # All cameras must have the same number
            # of frames.
            frame_count = min(
                arr.shape[0]
                for arr in camera_arrays
            )

            img[
                sequence_name
            ] = []

            for _ in BELIEF_CAMERA_IDS:

                img[
                    sequence_name
                ].append(
                    np.zeros(
                        (
                            frame_count,
                            DUMMY_IMAGE_FEATURE_DIM
                        ),
                        dtype=np.float32
                    )
                )

        # ----------------------------------------------------
        # Synchronize with 3D data
        # ----------------------------------------------------

        for sequence_name in list(
            converted_keypoints.keys()
        ):

            if sequence_name not in dataset:

                del converted_keypoints[
                    sequence_name
                ]

                del img[
                    sequence_name
                ]

                continue

            positions_3d = (
                dataset[
                    sequence_name
                ][
                    "positions"
                ]
            )

            mocap_length = (
                positions_3d.shape[0]
            )

            # Determine shortest modality
            min_len = mocap_length

            for arr in converted_keypoints[
                sequence_name
            ]:

                min_len = min(
                    min_len,
                    arr.shape[0]
                )

            for arr in img[
                sequence_name
            ]:

                min_len = min(
                    min_len,
                    arr.shape[0]
                )

            # ------------------------------------------------
            # Truncate 3D
            # ------------------------------------------------

            dataset[
                sequence_name
            ][
                "positions"
            ] = dataset[
                sequence_name
            ][
                "positions"
            ][:min_len]

            # ------------------------------------------------
            # Truncate 2D
            # ------------------------------------------------

            for cam_idx in range(
                len(
                    converted_keypoints[
                        sequence_name
                    ]
                )
            ):

                converted_keypoints[
                    sequence_name
                ][cam_idx] = (
                    converted_keypoints[
                        sequence_name
                    ][cam_idx
                    ][:min_len]
                )

            # ------------------------------------------------
            # Truncate dummy image features
            # ------------------------------------------------

            for cam_idx in range(
                len(
                    img[
                        sequence_name
                    ]
                )
            ):

                img[
                    sequence_name
                ][cam_idx] = (
                    img[
                        sequence_name
                    ][cam_idx
                    ][:min_len]
                )

        # ----------------------------------------------------
        # Convert to DSVTformer format
        # ----------------------------------------------------

        for sequence_name in list(
            converted_keypoints.keys()
        ):

            # ----------------------------------------------
            # 2D:
            #
            # (T,4,17,2)
            # ----------------------------------------------

            positions_2d = np.stack(
                converted_keypoints[
                    sequence_name
                ],
                axis=1
            ).astype(
                np.float32
            )

            # ----------------------------------------------
            # Dummy features:
            #
            # (T,4,F)
            # ----------------------------------------------

            image_features = np.stack(
                img[
                    sequence_name
                ],
                axis=1
            ).astype(
                np.float32
            )

            converted_keypoints[
                sequence_name
            ] = positions_2d

            img[
                sequence_name
            ] = image_features

        # ----------------------------------------------------
        # Store converted structures
        # ----------------------------------------------------

        self._belief_keypoints = (
            converted_keypoints
        )

        self._belief_images = img

        # ----------------------------------------------------
        # Print diagnostic information
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "BELIEF DATASET PREPARATION"
        )

        print(
            "=" * 70
        )

        print(
            "Sequences available:",
            len(converted_keypoints)
        )

        print(
            "Selected cameras:",
            BELIEF_CAMERA_IDS
        )

        if len(converted_keypoints) > 0:

            first_seq = next(
                iter(
                    converted_keypoints
                )
            )

            print(
                "Example sequence:",
                first_seq
            )

            print(
                "2D shape:",
                converted_keypoints[
                    first_seq
                ].shape
            )

            print(
                "Image feature shape:",
                img[
                    first_seq
                ].shape
            )

        print(
            "=" * 70
        )

        return (
            converted_keypoints,
            img
        )

    # ========================================================
    # FETCH
    # ========================================================

    def fetch(
        self,
        dataset,
        subjects,
        subset=1
    ):

        out_poses_3d = {}

        out_poses_2d = {}

        out_images = {}

        out_camera_params = {}

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # For belief data, "subjects" from opt may contain
        # H36M IDs such as:
        #
        #     8
        #     9
        #     11
        #
        # Those do not exist in the belief dataset.
        #
        # Therefore we use all sequences available in the
        # prepared belief data.
        # ----------------------------------------------------

        sequence_names = (
            list(
                self._belief_keypoints.keys()
            )
        )

        # ----------------------------------------------------
        # Optional split handling
        #
        # If opt.subjects_train/test contains actual belief
        # sequence names, use them.
        #
        # Otherwise use all sequences.
        # ----------------------------------------------------

        requested = set(
            str(x).strip()
            for x in subjects
        )

        use_requested_filter = False

        for sequence_name in sequence_names:

            video_part = (
                sequence_name.split("_")[1]
                if sequence_name.startswith(
                    "video_"
                )
                else None
            )

            if (
                sequence_name in requested
                or video_part in requested
            ):

                use_requested_filter = True

                break

        # ----------------------------------------------------
        # Build outputs
        # ----------------------------------------------------

        for sequence_name in sequence_names:

            # ----------------------------------------------
            # Filter only if the user explicitly supplied
            # matching belief IDs.
            # ----------------------------------------------

            if use_requested_filter:

                video_id = (
                    sequence_name.split("_")[1]
                )

                if (
                    sequence_name not in requested
                    and video_id not in requested
                ):

                    continue

            # ----------------------------------------------
            # Convert sequence:
            #
            # video_0_seg_1
            #
            # ->
            #
            # subject = video_0
            # action  = seg_1
            # ----------------------------------------------

            subject, action = (
                _extract_sequence_id(
                    sequence_name
                )
            )

            seq_key = (
                subject,
                action
            )

            # ----------------------------------------------
            # 2D
            # ----------------------------------------------

            out_poses_2d[
                seq_key
            ] = self._belief_keypoints[
                sequence_name
            ]

            # ----------------------------------------------
            # Dummy image features
            # ----------------------------------------------

            out_images[
                seq_key
            ] = self._belief_images[
                sequence_name
            ]

            # ----------------------------------------------
            # 3D
            #
            # dataset structure may be either:
            #
            # dataset[sequence_name]
            #
            # or:
            #
            # dataset[subject][action]
            # ----------------------------------------------

            if sequence_name in dataset:

                positions = (
                    dataset[
                        sequence_name
                    ][
                        "positions"
                    ]
                )

            elif (
                subject in dataset
                and action in dataset[
                    subject
                ]
            ):

                positions = (
                    dataset[
                        subject
                    ][
                        action
                    ][
                        "positions"
                    ]
                )

            else:

                raise KeyError(
                    f"Could not find 3D data for "
                    f"{sequence_name}"
                )

            out_poses_3d[
                seq_key
            ] = np.asarray(
                positions,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Camera parameters
            #
            # Not available for belief dataset.
            # ------------------------------------------------

            # We intentionally leave this empty.

        # ----------------------------------------------------
        # Convert camera params
        # ----------------------------------------------------

        if len(out_camera_params) == 0:

            out_camera_params = None

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            "FETCHED DATA"
        )

        print(
            "Sequences:",
            len(out_poses_3d)
        )

        if len(out_poses_3d) > 0:

            first_key = next(
                iter(
                    out_poses_3d
                )
            )

            print(
                "First sequence:",
                first_key
            )

            print(
                "3D:",
                out_poses_3d[
                    first_key
                ].shape
            )

            print(
                "2D:",
                out_poses_2d[
                    first_key
                ].shape
            )

            print(
                "Image features:",
                out_images[
                    first_key
                ].shape
            )

        print(
            "-" * 70
        )

        return (
            out_camera_params,
            out_poses_3d,
            out_poses_2d,
            out_images
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.generator.pairs
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        (
            seq_name,
            start_3d,
            end_3d,
            flip,
            reverse
        ) = self.generator.pairs[
            index
        ]

        (
            cam,
            gt_3D,
            input_2D,
            images,
            action,
            subject,
            low_2d,
            high_2d
        ) = self.generator.get_batch(
            seq_name,
            start_3d,
            end_3d,
            False,
            False,
            False
        )

        # ----------------------------------------------------
        # Test augmentation
        # ----------------------------------------------------

        if (
            self.train == False
            and self.test_aug
        ):

            (
                _,
                _,
                input_2D_aug,
                images_aug,
                _,
                _,
                low_2d,
                high_2d
            ) = self.generator.get_batch(
                seq_name,
                start_3d,
                end_3d,
                mask=False,
                flip=False,
                reverse=False
            )

            input_2D = np.concatenate(
                (
                    np.expand_dims(
                        input_2D,
                        axis=0
                    ),
                    np.expand_dims(
                        input_2D_aug,
                        axis=0
                    )
                ),
                axis=0
            )

            images = np.concatenate(
                (
                    np.expand_dims(
                        images,
                        axis=0
                    ),
                    np.expand_dims(
                        images_aug,
                        axis=0
                    )
                ),
                axis=0
            )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bb_box = np.array(
            [0, 0, 1, 1],
            dtype=np.float32
        )

        input_2D_update = (
            input_2D
        )

        images_update = (
            images
        )

        scale = np.float64(
            1.0
        )

        # ----------------------------------------------------
        # Return EXACTLY the interface expected by main_img.py
        # ----------------------------------------------------

        return (
            cam,
            gt_3D,
            input_2D_update,
            images_update,
            action,
            subject,
            scale,
            bb_box,
            start_3d,
            high_2d
        )
