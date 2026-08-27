import os
import numpy as np
import torch
import torch.utils.data as data


# ============================================================
# CONFIGURATION
# ============================================================

# Your belief dataset contains:
#
# camera0 ... camera8
#
# DSVTformer uses 4 views in this configuration.
BELIEF_CAMERA_IDS = [0, 1, 2, 3]

# Camera used as the 3D supervision/reference coordinate system.
#
# Your JSON says:
#
# coordinate_system:
#     absolute_camera_space_m
#
# Therefore each camera has its own coordinate system.
#
# We use camera0 consistently as the GT reference.
GT_CAMERA_ID = 0

# Your belief dataset does not contain RGB image features.
#
# The original DSVTformer image branch expects image features.
# We therefore create zero features with this dimension.
#
# Change this only if your model expects another dimension.
DUMMY_IMAGE_FEATURE_DIM = 2048

USE_DUMMY_IMAGE_FEATURES = True


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _to_numpy_float32(x):
    """
    Safely convert an object to float32 numpy.
    """
    return np.asarray(x, dtype=np.float32)


def _get_sequence_names(dataset):
    """
    Get sequence names from the converted Human36mDataset.

    Expected:
        video_0_seg_1
        video_0_seg_2
        ...
    """

    try:
        return list(dataset.keys())
    except Exception:
        pass

    try:
        return list(dataset._data.keys())
    except Exception:
        pass

    raise TypeError(
        "Could not determine sequence names from dataset."
    )


def _get_sequence(dataset, sequence_name):
    """
    Get one sequence from Human36mDataset.

    Expected structure:

        dataset[sequence_name]
            -> {
                "default": {
                    "positions": {
                        "camera0": ndarray,
                        ...
                    },
                    "cameras": ...
                }
            }
    """

    seq = dataset[sequence_name]

    if not isinstance(seq, dict):
        raise TypeError(
            f"Sequence {sequence_name} must be dict, "
            f"got {type(seq)}"
        )

    return seq


def _get_action_data(dataset, sequence_name):
    """
    Get the actual action block.

    Your observed structure is:

        video_0_seg_1
            default
                positions
                cameras
    """

    seq = _get_sequence(dataset, sequence_name)

    if "default" in seq:
        action_data = seq["default"]

        if not isinstance(action_data, dict):
            raise TypeError(
                f"{sequence_name}['default'] must be dict."
            )

        return action_data

    # Fallback: if another action name exists, use the first one.
    if len(seq) == 1:
        action_name = next(iter(seq))
        action_data = seq[action_name]

        if isinstance(action_data, dict):
            return action_data

    # Some implementations may already expose positions.
    if "positions" in seq:
        return seq

    raise KeyError(
        f"Could not find action data for {sequence_name}. "
        f"Available keys: {list(seq.keys())}"
    )


def _get_positions(dataset, sequence_name):
    """
    Return the camera dictionary:

        {
            'camera0': ndarray,
            ...
            'camera8': ndarray
        }
    """

    action_data = _get_action_data(
        dataset,
        sequence_name
    )

    if "positions" not in action_data:
        raise KeyError(
            f"'positions' not found in {sequence_name}. "
            f"Available keys: {list(action_data.keys())}"
        )

    positions = action_data["positions"]

    if not isinstance(positions, dict):
        raise TypeError(
            f"{sequence_name} positions must be dict, "
            f"got {type(positions)}"
        )

    return positions


def _get_camera_positions(
    dataset,
    sequence_name,
    camera_id
):
    """
    Get one camera's 3D pose.

    Expected:

        (T, 17, 3)
    """

    positions = _get_positions(
        dataset,
        sequence_name
    )

    camera_name = f"camera{camera_id}"

    if camera_name not in positions:
        raise KeyError(
            f"{sequence_name} does not contain "
            f"{camera_name}.\n"
            f"Available cameras: {list(positions.keys())}"
        )

    arr = _to_numpy_float32(
        positions[camera_name]
    )

    if arr.ndim != 3:
        raise ValueError(
            f"{sequence_name}/{camera_name}: "
            f"expected 3D array (T,J,3), got {arr.shape}"
        )

    if arr.shape[-1] != 3:
        raise ValueError(
            f"{sequence_name}/{camera_name}: "
            f"expected last dimension 3, got {arr.shape}"
        )

    return arr


def _get_2d_sequence(
    keypoints,
    sequence_name,
    camera_ids
):
    """
    Extract selected 2D camera arrays.

    Supported input format:

        keypoints[sequence_name]["camera0"]
        keypoints[sequence_name]["camera1"]
        ...

    Also supports:

        keypoints[sequence_name] = [
            camera0,
            camera1,
            ...
        ]
    """

    if sequence_name not in keypoints:
        raise KeyError(
            f"2D keypoints do not contain sequence "
            f"{sequence_name}"
        )

    seq = keypoints[sequence_name]

    output = []

    # --------------------------------------------------------
    # Dictionary format
    # --------------------------------------------------------

    if isinstance(seq, dict):

        for camera_id in camera_ids:

            camera_name = f"camera{camera_id}"

            if camera_name not in seq:
                raise KeyError(
                    f"2D sequence {sequence_name} is missing "
                    f"{camera_name}.\n"
                    f"Available: {list(seq.keys())}"
                )

            arr = _to_numpy_float32(
                seq[camera_name]
            )

            output.append(arr)

    # --------------------------------------------------------
    # List format
    # --------------------------------------------------------

    elif isinstance(seq, (list, tuple)):

        for camera_id in camera_ids:

            if camera_id >= len(seq):
                raise IndexError(
                    f"2D sequence {sequence_name} contains "
                    f"{len(seq)} cameras, requested camera "
                    f"{camera_id}"
                )

            output.append(
                _to_numpy_float32(
                    seq[camera_id]
                )
            )

    else:

        raise TypeError(
            f"Unsupported 2D format for {sequence_name}: "
            f"{type(seq)}"
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    for camera_id, arr in zip(
        camera_ids,
        output
    ):

        if arr.ndim != 3:
            raise ValueError(
                f"{sequence_name}/camera{camera_id}: "
                f"expected (T,J,2), got {arr.shape}"
            )

        if arr.shape[-1] != 2:
            raise ValueError(
                f"{sequence_name}/camera{camera_id}: "
                f"expected last dimension 2, got {arr.shape}"
            )

    return output


def _make_view_stack(
    arrays,
    sequence_name,
    expected_last_dim
):
    """
    Convert:

        [
            (T,J,C),
            (T,J,C),
            (T,J,C),
            (T,J,C)
        ]

    into:

        (T,4,J,C)
    """

    if len(arrays) == 0:
        raise ValueError(
            f"No camera arrays for {sequence_name}"
        )

    lengths = [
        arr.shape[0]
        for arr in arrays
    ]

    min_length = min(lengths)

    arrays = [
        arr[:min_length]
        for arr in arrays
    ]

    joint_counts = [
        arr.shape[1]
        for arr in arrays
    ]

    if len(set(joint_counts)) != 1:
        raise ValueError(
            f"Different joint counts in {sequence_name}: "
            f"{joint_counts}"
        )

    for arr in arrays:

        if arr.shape[-1] != expected_last_dim:
            raise ValueError(
                f"Invalid final dimension in {sequence_name}: "
                f"{arr.shape}"
            )

    return np.stack(
        arrays,
        axis=1
    ).astype(
        np.float32
    )


def _find_2d_file(root_path):
    """
    Find the 2D NPZ generated for the project.
    """

    candidates = [

        os.path.join(
            root_path,
            "data_2d_h36m_cpn_ft_h36m_dbb.npz"
        ),

        os.path.join(
            root_path,
            "data_2d_belief.npz"
        ),

        os.path.join(
            root_path,
            "data",
            "data_2d_h36m_cpn_ft_h36m_dbb.npz"
        ),

        os.path.join(
            root_path,
            "data",
            "data_2d_belief.npz"
        ),

        os.path.join(
            "/content/Compare_pose_2D",
            "data",
            "data_2d_h36m_cpn_ft_h36m_dbb.npz"
        ),

        os.path.join(
            "/content/Compare_pose_2D",
            "data",
            "data_2d_belief.npz"
        )
    ]

    for path in candidates:

        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "\nCould not find 2D keypoint NPZ.\n"
        "Tried:\n"
        + "\n".join(
            f"  {p}"
            for p in candidates
        )
    )


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

        if len(poses_2d) == 0:
            raise ValueError(
                "poses_2d is empty."
            )

        if poses_3d is not None:

            if set(poses_3d.keys()) != set(
                poses_2d.keys()
            ):
                raise ValueError(
                    "3D and 2D sequence keys do not match."
                )

        if set(images.keys()) != set(
            poses_2d.keys()
        ):
            raise ValueError(
                "Image-feature and 2D sequence keys "
                "do not match."
            )

        self.batch_size = max(
            1,
            int(batch_size)
        )

        self.pad = int(pad)

        self.causal_shift = int(
            causal_shift
        )

        self.shuffle = shuffle

        self.endless = endless

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

        self.random = np.random.RandomState(
            random_seed
        )

        self.state = None

        self.saved_index = {}

        self.pairs = []

        start_index = 0

        # ----------------------------------------------------
        # Build chunks
        # ----------------------------------------------------

        for seq_key in poses_2d.keys():

            seq_2d = poses_2d[
                seq_key
            ]

            frame_count = seq_2d.shape[0]

            if poses_3d is not None:

                seq_3d = poses_3d[
                    seq_key
                ]

                if seq_3d.shape[0] != frame_count:

                    raise ValueError(
                        f"Frame mismatch for {seq_key}: "
                        f"2D={frame_count}, "
                        f"3D={seq_3d.shape[0]}"
                    )

            seq_img = images[
                seq_key
            ]

            if seq_img.shape[0] != frame_count:

                raise ValueError(
                    f"Image feature mismatch for "
                    f"{seq_key}: "
                    f"2D={frame_count}, "
                    f"images={seq_img.shape[0]}"
                )

            if frame_count == 0:
                continue

            n_chunks = (
                frame_count
                + chunk_length
                - 1
            ) // chunk_length

            offset = (
                n_chunks * chunk_length
                - frame_count
            ) // 2

            bounds = (
                np.arange(
                    n_chunks + 1
                )
                * chunk_length
                - offset
            )

            for chunk_id in range(
                n_chunks
            ):

                start = bounds[
                    chunk_id
                ]

                end = bounds[
                    chunk_id + 1
                ]

                self.pairs.append(
                    (
                        seq_key,
                        start,
                        end,
                        False,
                        False
                    )
                )

                # Reverse augmentation
                if reverse_aug:

                    self.pairs.append(
                        (
                            seq_key,
                            start,
                            end,
                            False,
                            True
                        )
                    )

                # Mirror augmentation
                if augment:

                    self.pairs.append(
                        (
                            seq_key,
                            start,
                            end,
                            True,
                            False
                        )
                    )

                    if reverse_aug:

                        self.pairs.append(
                            (
                                seq_key,
                                start,
                                end,
                                True,
                                True
                            )
                        )

            if poses_3d is not None:

                self.saved_index[
                    seq_key
                ] = [
                    start_index,
                    start_index + frame_count
                ]

                start_index += frame_count

        self.num_batches = (
            len(self.pairs)
            + self.batch_size
            - 1
        ) // self.batch_size

        self.batch_3d = None

        self.batch_2d = None

        self.batch_images = None

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

    def set_random_state(
        self,
        random
    ):

        self.random = random

    def augment_enabled(self):

        return self.augment

    # --------------------------------------------------------
    # Pairs
    # --------------------------------------------------------

    def next_pairs(self):

        if self.state is None:

            if self.shuffle:

                indices = self.random.permutation(
                    len(self.pairs)
                )

                pairs = [
                    self.pairs[i]
                    for i in indices
                ]

            else:

                pairs = self.pairs

            return 0, pairs

        return self.state

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    def get_batch(
        self,
        seq_key,
        start_3d,
        end_3d,
        mask,
        flip,
        reverse
    ):

        seq_2d = self.poses_2d[
            seq_key
        ]

        seq_img = self.images[
            seq_key
        ]

        seq_3d = None

        if self.poses_3d is not None:

            seq_3d = self.poses_3d[
                seq_key
            ]

        # ----------------------------------------------------
        # 2D temporal range
        # ----------------------------------------------------

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

        low_2d = max(
            0,
            start_2d
        )

        high_2d = min(
            seq_2d.shape[0],
            end_2d
        )

        pad_left = (
            low_2d
            - start_2d
        )

        pad_right = (
            end_2d
            - high_2d
        )

        current_2d = seq_2d[
            low_2d:high_2d
        ]

        current_img = seq_img[
            low_2d:high_2d
        ]

        # ----------------------------------------------------
        # Edge padding
        # ----------------------------------------------------

        if (
            pad_left > 0
            or pad_right > 0
        ):

            current_2d = np.pad(
                current_2d,
                (
                    (pad_left, pad_right),
                    (0, 0),
                    (0, 0),
                    (0, 0)
                ),
                mode="edge"
            )

            current_img = np.pad(
                current_img,
                (
                    (pad_left, pad_right),
                    (0, 0),
                    (0, 0)
                ),
                mode="edge"
            )

        self.batch_2d = current_2d

        self.batch_images = current_img

        # ----------------------------------------------------
        # 3D
        # ----------------------------------------------------

        if seq_3d is not None:

            low_3d = max(
                0,
                start_3d
            )

            high_3d = min(
                seq_3d.shape[0],
                end_3d
            )

            pad_left_3d = (
                low_3d
                - start_3d
            )

            pad_right_3d = (
                end_3d
                - high_3d
            )

            current_3d = seq_3d[
                low_3d:high_3d
            ]

            if (
                pad_left_3d > 0
                or pad_right_3d > 0
            ):

                current_3d = np.pad(
                    current_3d,
                    (
                        (
                            pad_left_3d,
                            pad_right_3d
                        ),
                        (0, 0),
                        (0, 0)
                    ),
                    mode="edge"
                )

            self.batch_3d = current_3d

        else:

            self.batch_3d = None

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        # The belief dataset does not contain calibration
        # parameters compatible with the original H36M camera
        # object.
        camera_output = None

        if seq_3d is not None:

            return (
                camera_output,
                self.batch_3d.copy(),
                self.batch_2d.copy(),
                self.batch_images.copy(),
                seq_key[1],
                seq_key[0],
                low_2d,
                high_2d
            )

        return (
            camera_output,
            None,
            self.batch_2d.copy(),
            self.batch_images.copy(),
            seq_key[1],
            seq_key[0],
            low_2d,
            high_2d
        )


# ============================================================
# FUSION DATASET
# ============================================================

class Fusion(data.Dataset):

    """
    DSVTformer Fusion dataset for the converted belief dataset.

    Actual 3D structure:

        dataset[
            "video_0_seg_1"
        ][
            "default"
        ][
            "positions"
        ][
            "camera0"
        ]

    where camera0 has:

        (T, 17, 3)

    Actual 2D structure:

        positions_2d[
            "video_0_seg_1"
        ][
            "camera0"
        ]

    where camera0 has:

        (T, 17, 2)

    Final model input:

        2D:
            (T, 4, 17, 2)

        image features:
            (T, 4, 2048)

        3D:
            (T, 17, 3)
    """

    def __init__(
        self,
        opt,
        dataset,
        root_path,
        train=True
    ):

        super().__init__()

        self.data_type = getattr(
            opt,
            "dataset",
            "belief"
        )

        self.train = train

        self.keypoints_name = getattr(
            opt,
            "keypoints",
            "cpn"
        )

        self.root_path = root_path

        # ----------------------------------------------------
        # Options
        # ----------------------------------------------------

        self.train_list = self._get_subject_list(
            getattr(
                opt,
                "subjects_train",
                ""
            )
        )

        self.test_list = self._get_subject_list(
            getattr(
                opt,
                "subjects_test",
                ""
            )
        )

        actions = getattr(
            opt,
            "actions",
            "*"
        )

        self.action_filter = (
            None
            if actions == "*"
            else str(actions).split(",")
        )

        self.downsample = int(
            getattr(
                opt,
                "downsample",
                1
            )
        )

        self.subset = float(
            getattr(
                opt,
                "subset",
                1
            )
        )

        self.stride = int(
            getattr(
                opt,
                "stride",
                1
            )
        )

        self.crop_uv = getattr(
            opt,
            "crop_uv",
            False
        )

        self.test_aug = getattr(
            opt,
            "test_augmentation",
            False
        )

        self.pad = int(
            getattr(
                opt,
                "pad",
                0
            )
        )

        self.batch_size = int(
            getattr(
                opt,
                "batch_size",
                1
            )
        )

        self.data_augmentation = getattr(
            opt,
            "data_augmentation",
            False
        )

        self.reverse_augmentation = getattr(
            opt,
            "reverse_augmentation",
            False
        )

        self.out_all = getattr(
            opt,
            "out_all",
            False
        )

        # ----------------------------------------------------
        # Symmetry
        # ----------------------------------------------------

        self._setup_symmetry(
            dataset
        )

        # ----------------------------------------------------
        # Prepare
        # ----------------------------------------------------

        self.keypoints, self.img = (
            self.prepare_data(
                dataset
            )
        )

        # ----------------------------------------------------
        # Select sequences
        # ----------------------------------------------------

        if self.train:

            selected = self.train_list

        else:

            selected = self.test_list

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

        if len(self.poses_train_2d) == 0:

            raise RuntimeError(
                "\nNo sequences were selected.\n"
                f"Requested subjects: {selected}\n"
                "Available belief sequences include names "
                "such as video_0_seg_1.\n"
            )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if self.train:

            generator_batch_size = max(
                1,
                self.batch_size
                // max(1, self.stride)
            )

            self.generator = ChunkedGenerator(
                generator_batch_size,
                self.cameras_train,
                self.poses_train,
                self.poses_train_2d,
                self.images_train,
                chunk_length=max(
                    1,
                    self.stride
                ),
                pad=self.pad,
                augment=self.data_augmentation,
                reverse_aug=self.reverse_augmentation,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right,
                out_all=self.out_all
            )

            print(
                "INFO: Training on {} chunks".format(
                    len(
                        self.generator.pairs
                    )
                )
            )

        # ----------------------------------------------------
        # Testing
        # ----------------------------------------------------

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
                max(
                    1,
                    self.batch_size
                ),
                self.cameras_test,
                self.poses_test,
                self.poses_test_2d,
                self.images_test,
                chunk_length=1,
                pad=self.pad,
                augment=False,
                reverse_aug=False,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right,
                out_all=self.out_all
            )

            self.key_index = (
                self.generator.saved_index
            )

            print(
                "INFO: Testing on {} frames".format(
                    len(
                        self.generator.pairs
                    )
                )
            )

    # ========================================================
    # SUBJECT LIST
    # ========================================================

    @staticmethod
    def _get_subject_list(value):

        if value is None:
            return []

        value = str(value).strip()

        if value == "":
            return []

        return [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    # ========================================================
    # SYMMETRY
    # ========================================================

    def _setup_symmetry(
        self,
        dataset
    ):

        try:

            skeleton = dataset.skeleton()

            self.kps_left = list(
                skeleton.joints_left()
            )

            self.kps_right = list(
                skeleton.joints_right()
            )

            self.joints_left = list(
                skeleton.joints_left()
            )

            self.joints_right = list(
                skeleton.joints_right()
            )

            return

        except Exception:
            pass

        # MPI-INF-3DHP 17-joint fallback.
        #
        # If your model has its own exact joint ordering,
        # these can be changed later.
        self.kps_left = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8
        ]

        self.kps_right = [
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ]

        self.joints_left = list(
            self.kps_left
        )

        self.joints_right = list(
            self.kps_right
        )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_data(
        self,
        dataset
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Preparing belief dataset"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Find 2D NPZ
        # ----------------------------------------------------

        keypoints_path = _find_2d_file(
            self.root_path
        )

        print(
            "2D keypoint file:"
        )

        print(
            keypoints_path
        )

        keypoints_npz = np.load(
            keypoints_path,
            allow_pickle=True
        )

        print(
            "2D NPZ keys:",
            keypoints_npz.files
        )

        if "positions_2d" not in keypoints_npz.files:

            raise KeyError(
                "positions_2d not found in 2D NPZ.\n"
                f"Available keys: {keypoints_npz.files}"
            )

        keypoints = (
            keypoints_npz[
                "positions_2d"
            ].item()
        )

        if not isinstance(keypoints, dict):

            raise TypeError(
                "positions_2d must contain a dict."
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if "metadata" in keypoints_npz.files:

            try:

                metadata = (
                    keypoints_npz[
                        "metadata"
                    ].item()
                )

                if (
                    isinstance(metadata, dict)
                    and
                    "keypoints_symmetry"
                    in metadata
                ):

                    symmetry = (
                        metadata[
                            "keypoints_symmetry"
                        ]
                    )

                    if len(symmetry) >= 2:

                        self.kps_left = list(
                            symmetry[0]
                        )

                        self.kps_right = list(
                            symmetry[1]
                        )

                        self.joints_left = list(
                            symmetry[0]
                        )

                        self.joints_right = list(
                            symmetry[1]
                        )

            except Exception as e:

                print(
                    "WARNING: Could not parse "
                    f"2D metadata: {e}"
                )

        # ----------------------------------------------------
        # Dataset sequence names
        # ----------------------------------------------------

        dataset_sequences = set(
            _get_sequence_names(
                dataset
            )
        )

        print(
            "3D dataset sequences:",
            len(dataset_sequences)
        )

        print(
            "2D dataset sequences:",
            len(keypoints)
        )

        # ----------------------------------------------------
        # Intersection
        # ----------------------------------------------------

        common_sequences = (
            dataset_sequences
            &
            set(keypoints.keys())
        )

        print(
            "Matching sequences:",
            len(common_sequences)
        )

        if len(common_sequences) == 0:

            # Print examples to make debugging easier.
            print(
                "\nExample 3D sequences:"
            )

            for x in list(
                dataset_sequences
            )[:10]:

                print(
                    " ",
                    x
                )

            print(
                "\nExample 2D sequences:"
            )

            for x in list(
                keypoints.keys()
            )[:10]:

                print(
                    " ",
                    x
                )

            raise RuntimeError(
                "No matching sequence names between "
                "3D and 2D datasets."
            )

        # ----------------------------------------------------
        # Output structures
        # ----------------------------------------------------

        converted_2d = {}

        converted_images = {}

        # ----------------------------------------------------
        # Process each sequence
        # ----------------------------------------------------

        for sequence_name in sorted(
            common_sequences
        ):

            # ----------------------------------------------
            # 2D
            # ----------------------------------------------

            camera_2d = _get_2d_sequence(
                keypoints,
                sequence_name,
                BELIEF_CAMERA_IDS
            )

            poses_2d = _make_view_stack(
                camera_2d,
                sequence_name,
                expected_last_dim=2
            )

            # ----------------------------------------------
            # 3D
            #
            # Use camera0 as GT.
            # ----------------------------------------------

            gt_3d = _get_camera_positions(
                dataset,
                sequence_name,
                GT_CAMERA_ID
            )

            # ----------------------------------------------
            # Synchronize frame count
            # ----------------------------------------------

            lengths = [
                poses_2d.shape[0],
                gt_3d.shape[0]
            ]

            min_length = min(
                lengths
            )

            poses_2d = poses_2d[
                :min_length
            ]

            gt_3d = gt_3d[
                :min_length
            ]

            # ----------------------------------------------
            # Dummy image features
            # ----------------------------------------------

            if USE_DUMMY_IMAGE_FEATURES:

                image_features = np.zeros(
                    (
                        min_length,
                        len(
                            BELIEF_CAMERA_IDS
                        ),
                        DUMMY_IMAGE_FEATURE_DIM
                    ),
                    dtype=np.float32
                )

            else:

                raise RuntimeError(
                    "Real image features are not configured."
                )

            # ----------------------------------------------
            # Store
            #
            # Internal key is:
            #
            # (sequence_name, "default")
            #
            # This avoids treating video IDs as H36M
            # subjects.
            # ----------------------------------------------

            seq_key = (
                sequence_name,
                "default"
            )

            converted_2d[
                seq_key
            ] = poses_2d

            converted_images[
                seq_key
            ] = image_features

            # ----------------------------------------------
            # Store GT separately.
            # ----------------------------------------------

        # ----------------------------------------------------
        # 3D output
        # ----------------------------------------------------

        converted_3d = {}

        for seq_key in converted_2d.keys():

            sequence_name = seq_key[0]

            gt_3d = _get_camera_positions(
                dataset,
                sequence_name,
                GT_CAMERA_ID
            )

            min_length = min(
                gt_3d.shape[0],
                converted_2d[
                    seq_key
                ].shape[0]
            )

            converted_3d[
                seq_key
            ] = gt_3d[
                :min_length
            ].astype(
                np.float32
            )

            converted_2d[
                seq_key
            ] = converted_2d[
                seq_key
            ][:min_length]

            converted_images[
                seq_key
            ] = converted_images[
                seq_key
            ][:min_length]

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "BELIEF DATASET PREPARATION COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            "Selected input cameras:",
            BELIEF_CAMERA_IDS
        )

        print(
            "3D GT camera:",
            GT_CAMERA_ID
        )

        print(
            "Sequences:",
            len(converted_2d)
        )

        if len(converted_2d) > 0:

            first_key = next(
                iter(
                    converted_2d
                )
            )

            print(
                "\nFirst sequence:",
                first_key
            )

            print(
                "2D shape:",
                converted_2d[
                    first_key
                ].shape
            )

            print(
                "3D shape:",
                converted_3d[
                    first_key
                ].shape
            )

            print(
                "Image feature shape:",
                converted_images[
                    first_key
                ].shape
            )

        print(
            "=" * 70
        )

        self._belief_keypoints = (
            converted_2d
        )

        self._belief_images = (
            converted_images
        )

        self._belief_3d = (
            converted_3d
        )

        return (
            converted_2d,
            converted_images
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

        out_camera_params = None

        all_keys = list(
            self._belief_keypoints.keys()
        )

        # ----------------------------------------------------
        # Requested filters
        # ----------------------------------------------------

        requested = set(
            str(x).strip()
            for x in subjects
            if str(x).strip()
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Original main_img.py uses:
        #
        # subjects_train = 8
        # subjects_test  = 9,11
        #
        # Those are H36M subject IDs.
        #
        # Your belief dataset instead has:
        #
        # video_0_seg_1
        #
        # video_1_seg_1
        #
        # etc.
        #
        # Therefore:
        #
        # If the requested list contains actual belief
        # sequence names or video IDs, filter them.
        #
        # Otherwise we use all sequences.
        # ----------------------------------------------------

        explicit_belief_filter = False

        for key in all_keys:

            sequence_name = key[0]

            if (
                sequence_name in requested
            ):
                explicit_belief_filter = True
                break

            if sequence_name.startswith(
                "video_"
            ):

                parts = sequence_name.split("_")

                if len(parts) >= 2:

                    video_id = parts[1]

                    if video_id in requested:

                        explicit_belief_filter = True
                        break

        # ----------------------------------------------------
        # Iterate
        # ----------------------------------------------------

        for seq_key in all_keys:

            sequence_name = seq_key[0]

            # ----------------------------------------------
            # Optional filter
            # ----------------------------------------------

            if explicit_belief_filter:

                matches = (
                    sequence_name
                    in requested
                )

                if sequence_name.startswith(
                    "video_"
                ):

                    parts = sequence_name.split("_")

                    if len(parts) >= 2:

                        video_id = parts[1]

                        matches = (
                            matches
                            or
                            video_id in requested
                        )

                if not matches:
                    continue

            # ----------------------------------------------
            # Data
            # ----------------------------------------------

            out_poses_2d[
                seq_key
            ] = self._belief_keypoints[
                seq_key
            ]

            out_images[
                seq_key
            ] = self._belief_images[
                seq_key
            ]

            out_poses_3d[
                seq_key
            ] = self._belief_3d[
                seq_key
            ]

        # ----------------------------------------------------
        # Subset
        # ----------------------------------------------------

        if subset is not None:

            subset = float(subset)

            if (
                0 < subset < 1
            ):

                rng = np.random.RandomState(
                    1234
                )

                for key in list(
                    out_poses_3d.keys()
                ):

                    n = out_poses_3d[
                        key
                    ].shape[0]

                    keep = max(
                        1,
                        int(
                            n * subset
                        )
                    )

                    indices = rng.choice(
                        n,
                        keep,
                        replace=False
                    )

                    indices.sort()

                    out_poses_3d[
                        key
                    ] = out_poses_3d[
                        key
                    ][indices]

                    out_poses_2d[
                        key
                    ] = out_poses_2d[
                        key
                    ][indices]

                    out_images[
                        key
                    ] = out_images[
                        key
                    ][indices]

        # ----------------------------------------------------
        # Downsample
        # ----------------------------------------------------

        if self.downsample > 1:

            for key in list(
                out_poses_3d.keys()
            ):

                out_poses_3d[
                    key
                ] = out_poses_3d[
                    key
                ][
                    ::self.downsample
                ]

                out_poses_2d[
                    key
                ] = out_poses_2d[
                    key
                ][
                    ::self.downsample
                ]

                out_images[
                    key
                ] = out_images[
                    key
                ][
                    ::self.downsample
                ]

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            "FETCH"
        )

        print(
            "Requested:",
            list(subjects)
        )

        print(
            "Selected sequences:",
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
                "Images:",
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
            seq_key,
            start_3d,
            end_3d,
            flip,
            reverse
        ) = self.generator.pairs[
            index
        ]

        (
            cam,
            gt_3d,
            input_2d,
            images,
            action,
            subject,
            low_2d,
            high_2d
        ) = self.generator.get_batch(
            seq_key,
            start_3d,
            end_3d,
            mask=False,
            flip=flip,
            reverse=reverse
        )

        # ----------------------------------------------------
        # Mirror augmentation
        # ----------------------------------------------------

        if flip:

            input_2d = input_2d.copy()

            # x-coordinate reflection
            input_2d[
                ...,
                0
            ] *= -1

            if (
                self.kps_left is not None
                and
                self.kps_right is not None
            ):

                input_2d[
                    :,
                    :,
                    self.kps_left
                    +
                    self.kps_right,
                    :
                ] = input_2d[
                    :,
                    :,
                    self.kps_right
                    +
                    self.kps_left,
                    :
                ]

            if gt_3d is not None:

                gt_3d = gt_3d.copy()

                gt_3d[
                    ...,
                    0
                ] *= -1

                if (
                    self.joints_left is not None
                    and
                    self.joints_right is not None
                ):

                    gt_3d[
                        :,
                        self.joints_left
                        +
                        self.joints_right,
                        :
                    ] = gt_3d[
                        :,
                        self.joints_right
                        +
                        self.joints_left,
                        :
                    ]

        # ----------------------------------------------------
        # Reverse temporal order
        # ----------------------------------------------------

        if reverse:

            input_2d = input_2d[
                ::-1
            ].copy()

            images = images[
                ::-1
            ].copy()

            if gt_3d is not None:

                gt_3d = gt_3d[
                    ::-1
                ].copy()

        # ----------------------------------------------------
        # Test augmentation
        # ----------------------------------------------------

        if (
            not self.train
            and self.test_aug
        ):

            input_2d_aug = input_2d.copy()

            input_2d_aug[
                ...,
                0
            ] *= -1

            images_aug = images.copy()

            if (
                self.kps_left is not None
                and
                self.kps_right is not None
            ):

                input_2d_aug[
                    :,
                    :,
                    self.kps_left
                    +
                    self.kps_right,
                    :
                ] = input_2d_aug[
                    :,
                    :,
                    self.kps_right
                    +
                    self.kps_left,
                    :
                ]

            input_2d = np.stack(
                [
                    input_2d,
                    input_2d_aug
                ],
                axis=0
            )

            images = np.stack(
                [
                    images,
                    images_aug
                ],
                axis=0
            )

        # ----------------------------------------------------
        # Bounding box
        #
        # Belief data does not contain a detector bounding box.
        # Use normalized full-frame box.
        # ----------------------------------------------------

        bb_box = np.array(
            [
                0.0,
                0.0,
                1.0,
                1.0
            ],
            dtype=np.float32
        )

        scale = np.float64(
            1.0
        )

        # ----------------------------------------------------
        # Return interface expected by main_img.py
        # ----------------------------------------------------

        return (
            cam,
            gt_3d,
            input_2d,
            images,
            action,
            subject,
            scale,
            bb_box,
            start_3d,
            high_2d
        )
