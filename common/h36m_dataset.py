class Human36mDataset(MocapDataset):

    def __init__(self, path, opt, remove_static_joints=False):
        """
        Loader for the converted belief_data dataset.

        Expected NPZ structure:

        positions_3d
        └── video_X_seg_Y
            ├── camera0 -> (T, 17, 3)
            ├── camera1 -> (T, 17, 3)
            ├── ...
            └── camera8 -> (T, 17, 3)

        This is NOT the original Human3.6M structure.
        Therefore each video segment is treated as an action/sequence.
        """

        super().__init__(
            fps=50,
            skeleton=h36m_skeleton
        )

        # --------------------------------------------------
        # IMPORTANT
        # --------------------------------------------------
        # Your data contains video IDs, NOT H36M subjects.
        #
        # We use:
        #
        # video_0_seg_1
        #
        # as a sequence name.
        #
        # The original H36M train/test subject lists cannot
        # be used directly.
        # --------------------------------------------------

        self.train_list = []
        self.test_list = []

        # --------------------------------------------------
        # Load the converted dataset
        # --------------------------------------------------

        print()
        print("=" * 70)
        print("Loading belief_data 3D dataset")
        print("=" * 70)

        npz = np.load(
            path,
            allow_pickle=True
        )

        if 'positions_3d' not in npz:
            raise KeyError(
                "NPZ does not contain 'positions_3d'"
            )

        raw_data = npz['positions_3d'].item()

        print(
            f"Found {len(raw_data)} sequences"
        )

        # --------------------------------------------------
        # Inspect structure
        # --------------------------------------------------

        first_key = next(iter(raw_data))

        print(
            f"First sequence: {first_key}"
        )

        print(
            f"First sequence type: "
            f"{type(raw_data[first_key])}"
        )

        # --------------------------------------------------
        # Build a camera representation
        # --------------------------------------------------
        #
        # Your dataset has 9 cameras:
        #
        # camera0 ... camera8
        #
        # These are NOT Human3.6M cameras.
        #
        # We therefore create lightweight camera metadata
        # instead of using h36m_cameras_extrinsic_params.
        # --------------------------------------------------

        self._cameras = {}

        # --------------------------------------------------
        # Convert:
        #
        # video_0_seg_1
        #
        # into:
        #
        # self._data[sequence]
        #      ['positions']
        #      ['cameras']
        #
        # --------------------------------------------------

        self._data = {}

        for sequence_name, sequence_data in raw_data.items():

            if not isinstance(sequence_data, dict):
                print(
                    f"[WARNING] Skipping {sequence_name}: "
                    f"expected dict, got "
                    f"{type(sequence_data)}"
                )
                continue

            self._data[sequence_name] = {}

            # --------------------------------------------------
            # Determine available cameras
            # --------------------------------------------------

            camera_names = sorted(
                sequence_data.keys(),
                key=lambda x: int(
                    x.replace("camera", "")
                )
                if x.startswith("camera")
                else 999
            )

            if len(camera_names) == 0:
                print(
                    f"[WARNING] No cameras found "
                    f"for {sequence_name}"
                )
                continue

            # --------------------------------------------------
            # Build camera metadata
            # --------------------------------------------------

            cameras = []

            for camera_name in camera_names:

                positions = sequence_data[camera_name]

                if not isinstance(
                    positions,
                    np.ndarray
                ):
                    positions = np.asarray(
                        positions,
                        dtype=np.float32
                    )

                positions = positions.astype(
                    np.float32
                )

                if positions.ndim != 3:
                    print(
                        f"[WARNING] {sequence_name}/"
                        f"{camera_name} has shape "
                        f"{positions.shape}; "
                        f"expected (T,17,3)"
                    )
                    continue

                # ----------------------------------------------
                # Camera information
                #
                # These are placeholders because your JSON
                # contains 3D camera-space coordinates but does
                # not contain H36M calibration parameters.
                # ----------------------------------------------

                cam = {
                    'id': camera_name,
                    'res_w': 288,
                    'res_h': 384,

                    # Normalized camera center
                    'center': np.array(
                        [0.0, 0.0],
                        dtype=np.float32
                    ),

                    # Placeholder focal length
                    'focal_length': np.array(
                        [1.0, 1.0],
                        dtype=np.float32
                    ),

                    'radial_distortion': np.zeros(
                        3,
                        dtype=np.float32
                    ),

                    'tangential_distortion': np.zeros(
                        2,
                        dtype=np.float32
                    ),

                    'translation': np.zeros(
                        3,
                        dtype=np.float32
                    ),

                    'intrinsic': np.array(
                        [
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        dtype=np.float32
                    )
                }

                cameras.append(cam)

            # --------------------------------------------------
            # Store camera list
            # --------------------------------------------------

            self._cameras[
                sequence_name
            ] = cameras

            # --------------------------------------------------
            # The rest of the project expects:
            #
            # subject
            #    action
            #       positions
            #
            # We use the sequence itself as subject and
            # "default" as its action.
            # --------------------------------------------------

            self._data[
                sequence_name
            ]['default'] = {
                'positions': sequence_data,
                'cameras': cameras
            }

        # --------------------------------------------------
        # Create train/test lists
        # --------------------------------------------------
        #
        # Since this is belief_data and not H36M, use all
        # sequences for testing unless main_img.py explicitly
        # supplies another split.
        # --------------------------------------------------

        sequences = list(
            self._data.keys()
        )

        self.train_list = sequences
        self.test_list = sequences

        # --------------------------------------------------
        # Print summary
        # --------------------------------------------------

        print()
        print(
            f"Loaded sequences: "
            f"{len(self._data)}"
        )

        if len(sequences) > 0:

            print(
                "Example sequences:"
            )

            for seq in sequences[:5]:

                print(
                    f"  {seq}"
                )

                cameras = self._data[
                    seq
                ]['default']['cameras']

                print(
                    f"    cameras: "
                    f"{len(cameras)}"
                )

        print(
            "=" * 70
        )
        print()

    def supports_semi_supervised(self):
        return True
