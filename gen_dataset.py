import os
import glob
import json
import re
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Your actual annotation directory
JSON_DIR = "/home/duc090306/belief_data/annotations"

# Output directory
OUT_DIR = "/home/duc090306/belief_data"

# Cameras contained in your JSON files.
#
# Your actual JSON contains:
# camera0, camera1, ..., camera8
#
# Use all 9 by default.
CAMERAS = [
    "camera0",
    "camera1",
    "camera2",
    "camera3",
    "camera4",
    "camera5",
    "camera6",
    "camera7",
    "camera8",
]

# Output filenames
OUT_3D = "data_3d_h36m.npz"
OUT_2D = "data_2d_h36m_cpn_ft_h36m_dbb.npz"

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# Your JSON coordinates are in METERS.
#
# Human3.6M / DSVTformer 3D ground truth is normally handled
# in millimeters.
#
# Therefore:
#
#     meters -> millimeters
#
# 1.0 meter = 1000 mm
#
CONVERT_METERS_TO_MM = True

# Root-center the skeleton around the pelvis.
#
# Human3.6M-style 17-joint data normally uses pelvis as root.
ROOT_CENTER = True

# ------------------------------------------------------------
# 2D representation
# ------------------------------------------------------------
#
# Your JSON DOES NOT contain image pixel coordinates.
# It contains 3D coordinates in camera space.
#
# Therefore we cannot produce true pixel coordinates without
# camera intrinsics.
#
# We can nevertheless produce normalized perspective coordinates:
#
#       x_2d = X / Z
#       y_2d = Y / Z
#
# These are NOT pixel coordinates.
#
# Set this to True if you want normalized camera-plane 2D.
#
CREATE_NORMALIZED_2D = True

# Small value to avoid division by zero
EPS = 1e-8


# ============================================================
# HUMAN3.6M 17-JOINT ORDER
# ============================================================
#
# Standard Human3.6M 17-joint ordering:
#
#  0 pelvis
#  1 right_hip
#  2 right_knee
#  3 right_ankle
#  4 left_hip
#  5 left_knee
#  6 left_ankle
#  7 spine
#  8 thorax
#  9 neck
# 10 head
# 11 left_shoulder
# 12 left_elbow
# 13 left_wrist
# 14 right_shoulder
# 15 right_elbow
# 16 right_wrist
#
# Human3.6M uses this 17-joint convention. 
# ============================================================

H36M_JOINT_NAMES = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


# ============================================================
# MPI-INF-3DHP / VNect -> Human3.6M mapping
# ============================================================
#
# Your JSON contains the following relevant joints:
#
# pelvis
# spine
# spine2
# spine3
# spine4
# neck
# head
# head_top
# left_clavicle
# left_shoulder
# left_elbow
# left_wrist
# left_hand
# right_clavicle
# right_shoulder
# right_elbow
# right_wrist
# right_hand
# left_hip
# left_knee
# left_ankle
# left_foot
# left_toe
# right_hip
# right_knee
# right_ankle
# right_foot
# right_toe
#
# We select the joints required by H36M.
#
# "spine" is used for H36M spine.
#
# "spine4" is used as the H36M thorax.
#
# This keeps the mapping explicit so it is easy to change later.
# ============================================================

VNect_TO_H36M = {
    "pelvis": "pelvis",

    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",

    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",

    "spine": "spine",
    "spine4": "thorax",

    "neck": "neck",
    "head": "head",

    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",

    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
}


# ============================================================
# LEFT/RIGHT SYMMETRY
# ============================================================
#
# This is the metadata used by your previous DSVTformer NPZ.
#
# H36M joint indices:
#
# left:
#   4,5,6 = left leg
#   11,12,13 = left arm
#
# right:
#   1,2,3 = right leg
#   14,15,16 = right arm
#
# ============================================================

KEYPOINTS_SYMMETRY = (
    [4, 5, 6, 11, 12, 13],
    [1, 2, 3, 14, 15, 16],
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_video_segment(filename):
    """
    Extract:

        video_6_seg_14.json

    -> video_id = 6
       segment_id = 14
    """

    basename = os.path.basename(filename)

    match = re.match(
        r"video_(\d+)_seg_(\d+)\.json",
        basename
    )

    if match is None:
        return None, None

    video_id = int(match.group(1))
    segment_id = int(match.group(2))

    return video_id, segment_id


def convert_frame_to_h36m(camera_data):
    """
    Convert one VNect 28-joint camera skeleton into
    Human3.6M 17-joint format.

    Input:
        camera_data = {
            "pelvis": [x,y,z],
            ...
        }

    Output:
        ndarray shape (17,3)
    """

    joints = []

    missing = []

    for h36m_joint in H36M_JOINT_NAMES:

        # Find the VNect joint corresponding to this H36M joint
        vnect_joint = None

        for source_joint, target_joint in VNect_TO_H36M.items():
            if target_joint == h36m_joint:
                vnect_joint = source_joint
                break

        if vnect_joint is None:
            missing.append(h36m_joint)
            joints.append([0.0, 0.0, 0.0])
            continue

        if vnect_joint not in camera_data:
            missing.append(vnect_joint)
            joints.append([0.0, 0.0, 0.0])
            continue

        xyz = camera_data[vnect_joint]

        if len(xyz) != 3:
            raise ValueError(
                f"Joint {vnect_joint} does not contain 3 coordinates."
            )

        joints.append([
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
        ])

    if missing:
        print(
            "WARNING: Missing joints:",
            missing
        )

    return np.asarray(joints, dtype=np.float32)


def root_center_pose(pose):
    """
    Make pelvis the origin.

    pose:
        (17,3)
    """

    pelvis = pose[0].copy()

    return pose - pelvis


def make_2d_from_camera_3d(pose_3d):
    """
    Convert camera-space 3D coordinates into normalized
    perspective coordinates.

        x = X / Z
        y = Y / Z

    Input:
        (17,3)

    Output:
        (17,2)

    IMPORTANT:
        These are normalized camera coordinates,
        NOT image pixels.
    """

    xyz = pose_3d

    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    Z_safe = np.where(
        np.abs(Z) < EPS,
        EPS,
        Z
    )

    x = X / Z_safe
    y = Y / Z_safe

    return np.stack([x, y], axis=-1).astype(np.float32)


def save_npz(path, **kwargs):
    """
    Save compressed NPZ.
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    np.savez_compressed(
        path,
        **kwargs
    )

    print(f"Saved: {path}")


# ============================================================
# PROCESS ONE JSON FILE
# ============================================================

def process_json_file(json_file):
    """
    Process one:

        video_X_seg_Y.json

    file.

    Returns:

        positions_3d
        positions_2d
        metadata
    """

    video_id, segment_id = extract_video_segment(
        json_file
    )

    if video_id is None:
        print(
            f"Skipping file with unexpected name: {json_file}"
        )
        return None

    print()
    print("=" * 70)
    print(
        f"Processing video={video_id}, "
        f"segment={segment_id}"
    )
    print(
        os.path.basename(json_file)
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as f:

        frames = json.load(f)

    # Actual file structure is a LIST of frames.
    if not isinstance(frames, list):
        raise ValueError(
            f"{json_file} does not contain a list of frames."
        )

    print(
        f"Number of frames in JSON: {len(frames)}"
    )

    if len(frames) == 0:
        print("WARNING: Empty JSON file.")
        return None

    # --------------------------------------------------------
    # Check first frame
    # --------------------------------------------------------

    first_frame = frames[0]

    print(
        "First frame:",
        first_frame.get("frame_name")
    )

    print(
        "Timestamp:",
        first_frame.get("timestamp_sec")
    )

    print(
        "Coordinate system:",
        first_frame.get("coordinate_system")
    )

    print(
        "Skeleton:",
        first_frame.get("skeleton")
    )

    # --------------------------------------------------------
    # Verify camera availability
    # --------------------------------------------------------

    available_cameras = [
        key
        for key in first_frame.keys()
        if key.startswith("camera")
    ]

    print(
        "Available cameras:",
        available_cameras
    )

    selected_cameras = [
        cam
        for cam in CAMERAS
        if cam in available_cameras
    ]

    if len(selected_cameras) == 0:
        raise ValueError(
            "None of the requested cameras were found."
        )

    print(
        "Using cameras:",
        selected_cameras
    )

    # --------------------------------------------------------
    # Sort frames by frame_id
    # --------------------------------------------------------

    frames = sorted(
        frames,
        key=lambda frame: int(
            frame.get("frame_id", 0)
        )
    )

    # --------------------------------------------------------
    # Create containers
    # --------------------------------------------------------

    positions_3d = {}

    positions_2d = {}

    frame_ids = []

    timestamps = []

    # --------------------------------------------------------
    # Process each camera
    # --------------------------------------------------------

    for camera_name in selected_cameras:

        print(
            f"\nProcessing {camera_name}..."
        )

        sequence_3d = []
        sequence_2d = []

        for frame_index, frame in enumerate(frames):

            # ----------------------------------------------
            # Metadata
            # ----------------------------------------------

            frame_id = frame.get(
                "frame_id",
                frame_index
            )

            timestamp = frame.get(
                "timestamp_sec",
                None
            )

            if camera_name not in frame:

                print(
                    f"WARNING: "
                    f"{camera_name} missing "
                    f"in frame {frame_id}"
                )

                continue

            camera_data = frame[camera_name]

            # ----------------------------------------------
            # Convert 28 -> 17 joints
            # ----------------------------------------------

            pose_3d = convert_frame_to_h36m(
                camera_data
            )

            # ----------------------------------------------
            # Convert meters -> millimeters
            # ----------------------------------------------

            if CONVERT_METERS_TO_MM:
                pose_3d = pose_3d * 1000.0

            # ----------------------------------------------
            # Root center
            # ----------------------------------------------

            if ROOT_CENTER:
                pose_3d = root_center_pose(
                    pose_3d
                )

            # ----------------------------------------------
            # Save 3D
            # ----------------------------------------------

            sequence_3d.append(
                pose_3d
            )

            # ----------------------------------------------
            # Generate normalized 2D
            # ----------------------------------------------

            if CREATE_NORMALIZED_2D:

                # IMPORTANT:
                #
                # Projection must happen before
                # root-centering.
                #
                # Therefore we need the original
                # camera-space coordinates.
                #
                original_pose = convert_frame_to_h36m(
                    camera_data
                )

                pose_2d = make_2d_from_camera_3d(
                    original_pose
                )

                sequence_2d.append(
                    pose_2d
                )

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        sequence_3d = np.asarray(
            sequence_3d,
            dtype=np.float32
        )

        positions_3d[camera_name] = (
            sequence_3d
        )

        if CREATE_NORMALIZED_2D:

            sequence_2d = np.asarray(
                sequence_2d,
                dtype=np.float32
            )

            positions_2d[camera_name] = (
                sequence_2d
            )

        print(
            f"{camera_name}: "
            f"3D shape = {sequence_3d.shape}"
        )

        if CREATE_NORMALIZED_2D:

            print(
                f"{camera_name}: "
                f"2D shape = {sequence_2d.shape}"
            )

    # --------------------------------------------------------
    # Collect frame metadata
    # --------------------------------------------------------

    for frame in frames:

        frame_ids.append(
            frame.get("frame_id")
        )

        timestamps.append(
            frame.get("timestamp_sec")
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "video_id": video_id,
        "segment_id": segment_id,

        "frame_ids": np.asarray(
            frame_ids,
            dtype=np.int64
        ),

        "timestamps": np.asarray(
            timestamps,
            dtype=np.float32
        ),

        "cameras": selected_cameras,

        "joint_names": H36M_JOINT_NAMES,

        "source_skeleton":
            "MPI-INF-3DHP_28joints_VNect",

        "coordinate_system":
            "absolute_camera_space_m",

        "source_units":
            "meters",

        "output_3d_units":
            "millimeters"
            if CONVERT_METERS_TO_MM
            else "meters",

        "root_centered":
            ROOT_CENTER,

        "two_d_representation":
            "normalized_camera_coordinates_X_over_Z_Y_over_Z",
    }

    return (
        positions_3d,
        positions_2d,
        metadata
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BELIEF DATA -> DSVTformer PREPROCESSOR")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find actual files
    # --------------------------------------------------------

    pattern = os.path.join(
        JSON_DIR,
        "video_*_seg_*.json"
    )

    json_files = glob.glob(pattern)

    json_files = sorted(
        json_files
    )

    print(
        f"JSON directory:\n{JSON_DIR}"
    )

    print(
        f"Output directory:\n{OUT_DIR}"
    )

    print(
        f"\nFound {len(json_files)} JSON files."
    )

    if len(json_files) == 0:

        raise FileNotFoundError(
            "\nNo files matching:\n"
            f"    {pattern}\n\n"
            "Check JSON_DIR."
        )

    # --------------------------------------------------------
    # Global containers
    # --------------------------------------------------------

    all_positions_3d = {}

    all_positions_2d = {}

    all_metadata = {}

    total_sequences = 0

    # --------------------------------------------------------
    # Process every video segment
    # --------------------------------------------------------

    for json_file in json_files:

        result = process_json_file(
            json_file
        )

        if result is None:
            continue

        (
            positions_3d,
            positions_2d,
            metadata
        ) = result

        video_id = metadata[
            "video_id"
        ]

        segment_id = metadata[
            "segment_id"
        ]

        # ----------------------------------------------------
        # Use:
        #
        # video_6_seg_14
        #
        # as the sequence identifier.
        # ----------------------------------------------------

        sequence_name = (
            f"video_{video_id}_seg_{segment_id}"
        )

        all_positions_3d[
            sequence_name
        ] = positions_3d

        if CREATE_NORMALIZED_2D:

            all_positions_2d[
                sequence_name
            ] = positions_2d

        all_metadata[
            sequence_name
        ] = metadata

        total_sequences += 1

    # ========================================================
    # SAVE 3D
    # ========================================================

    out_3d_path = os.path.join(
        OUT_DIR,
        OUT_3D
    )

    save_npz(
        out_3d_path,
        positions_3d=all_positions_3d,
        metadata=all_metadata
    )

    # ========================================================
    # SAVE 2D
    # ========================================================

    if CREATE_NORMALIZED_2D:

        out_2d_path = os.path.join(
            OUT_DIR,
            OUT_2D
        )

        dsvtformer_metadata = {
            "keypoints_symmetry":
                KEYPOINTS_SYMMETRY,

            "joint_names":
                H36M_JOINT_NAMES,

            "source":
                "video_*_seg_*.json",

            "coordinate_type":
                "normalized_camera_coordinates",

            "coordinate_formula":
                "[X/Z, Y/Z]",
        }

        save_npz(
            out_2d_path,
            positions_2d=all_positions_2d,
            metadata=dsvtformer_metadata
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Sequences processed: {total_sequences}"
    )

    print(
        f"3D output:\n{out_3d_path}"
    )

    if CREATE_NORMALIZED_2D:

        print(
            f"2D output:\n{out_2d_path}"
        )

    print()
    print(
        "H36M joints:"
    )

    for i, name in enumerate(
        H36M_JOINT_NAMES
    ):
        print(
            f"  {i:2d}: {name}"
        )

    print()
    print(
        "Cameras:"
    )

    print(
        CAMERAS
    )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
