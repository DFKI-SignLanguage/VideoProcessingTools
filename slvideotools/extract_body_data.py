# Extracts the body (pose) landmarks from the frames of a video using MediaPipe.
# See: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

import math
from pathlib import Path

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from PIL.Image import Image
import PIL.Image
from PIL import ImageDraw

from .datagen import create_frame_producer, create_frame_consumer
from .datagen import FrameProducer, FrameConsumer

from typing import List, Tuple, Optional, Union


# Models must be downloaded into the `models` directory. See models/README.txt
MODELS_DIR_PATH = Path(__file__).parent.parent / 'models'
POSE_LANDMARKER_MODEL_PATH = MODELS_DIR_PATH / 'pose_landmarker_heavy.task'

MEDIAPIPE_POSE_LANDMARKS_COUNT = 33

# Landmarks with a visibility below this value are drawn as hollow circles in the composite frames.
COMPOSITE_VISIBILITY_THRESHOLD = 0.5

# Edges of the 33-points MediaPipe (BlazePose) body topology.
# Defined here because `mp_vision.PoseLandmarksConnections` is not exported by all mediapipe versions.
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # arms and hands
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # legs and feet
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Indices of the landmarks on the signer's anatomical left and right side (all odd/even indices, except the nose).
LEFT_SIDE_LANDMARKS = {i for i in range(1, MEDIAPIPE_POSE_LANDMARKS_COUNT) if i in (1, 2, 3, 7, 9) or (i >= 11 and i % 2 == 1)}
RIGHT_SIDE_LANDMARKS = {i for i in range(1, MEDIAPIPE_POSE_LANDMARKS_COUNT) if i not in LEFT_SIDE_LANDMARKS}

LEFT_COLOR = (220, 20, 20)    # red
RIGHT_COLOR = (20, 60, 220)   # blue
CENTER_COLOR = (20, 200, 20)  # green
SKELETON_COLOR = (240, 240, 240)


def _none_to_nan(v: Optional[float]) -> float:
    return float('nan') if v is None else v


def _draw_body(pil_draw: ImageDraw.ImageDraw, landmarks: np.ndarray, visibility: np.ndarray,
               width: int, height: int) -> None:
    """Draws the skeleton and the landmarks of one frame. Landmarks are in normalized [0,1] coordinates."""

    radius = max(2, int(width / 300))
    line_width = max(1, int(width / 600))

    xs = landmarks[:, 0] * width
    ys = landmarks[:, 1] * height

    for a, b in POSE_CONNECTIONS:
        pil_draw.line(xy=[xs[a], ys[a], xs[b], ys[b]], fill=SKELETON_COLOR, width=line_width)

    for i in range(MEDIAPIPE_POSE_LANDMARKS_COUNT):
        if i in LEFT_SIDE_LANDMARKS:
            color = LEFT_COLOR
        elif i in RIGHT_SIDE_LANDMARKS:
            color = RIGHT_COLOR
        else:
            color = CENTER_COLOR

        vis = visibility[i, 0]
        visible = not math.isnan(vis) and vis >= COMPOSITE_VISIBILITY_THRESHOLD
        pil_draw.ellipse(xy=[xs[i] - radius, ys[i] - radius, xs[i] + radius, ys[i] + radius],
                         fill=color if visible else None, outline=color, width=line_width)


def extract_body_data(frames_in: FrameProducer,
                      composite_frames_out: Optional[FrameConsumer] = None,
                      fps: Optional[float] = None,
                      model_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the MediaPipe Pose Landmarker data from the frames of a video.

    :param frames_in: A FrameProducer of some video material containing (the upper part of) a body.
    :param composite_frames_out: If provided, the landmarks are overlaid to the original frames and saved here.
    :param fps: The frame rate of the input. If provided, the landmarker runs in VIDEO mode
      (it tracks the body across frames, with timestamps i * 1000 / fps milliseconds).
      If None, the landmarker runs in IMAGE mode (every frame is processed independently).
    :param model_path: Path to the `pose_landmarker_*.task` model. Default: `models/pose_landmarker_heavy.task`.
    :return: a 2-tuple of numpy ndarrays of type float32:
      1) the body landmarks in shape [N][33][3], with <x,y,z> coordinates.
         As for MediaPipe, x and y are normalized in the range [0,1] by the frame width and height, respectively.
         Landmarks out of the frame are estimated by MediaPipe and may lay outside the [0,1] range.
      2) the landmark scores in shape [N][33][2], with <visibility,presence> in range [0,1].
      Where N is the number of frames. Inside a frame, if no bodies are detected, all values are NaN.
      If more bodies are detected, only the first in the MediaPipe list is used.
    """

    if model_path is None:
        model_path = POSE_LANDMARKER_MODEL_PATH

    running_mode = mp_vision.RunningMode.IMAGE if fps is None else mp_vision.RunningMode.VIDEO
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_poses=1,
    )

    out_landmarks_list: List[np.ndarray] = []
    out_visibility_list: List[np.ndarray] = []

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:

        for i, rgb_image in enumerate(frames_in.frames()):

            #
            # RUN MEDIAPIPE POSE LANDMARKER
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))
            if fps is None:
                results = landmarker.detect(mp_image)
            else:
                # Timestamps must be strictly increasing: true for any fps < 1000.
                results = landmarker.detect_for_video(mp_image, round(i * 1000 / fps))

            frame_landmarks = np.full((MEDIAPIPE_POSE_LANDMARKS_COUNT, 3), np.nan, dtype=np.float32)
            frame_visibility = np.full((MEDIAPIPE_POSE_LANDMARKS_COUNT, 2), np.nan, dtype=np.float32)
            if results.pose_landmarks:
                # Assume there is only one body
                landmarks = results.pose_landmarks[0]
                assert len(landmarks) == MEDIAPIPE_POSE_LANDMARKS_COUNT
                frame_landmarks[:] = [(lm.x, lm.y, lm.z) for lm in landmarks]
                frame_visibility[:] = [(_none_to_nan(lm.visibility), _none_to_nan(lm.presence)) for lm in landmarks]

            out_landmarks_list.append(frame_landmarks)
            out_visibility_list.append(frame_visibility)

            #
            # Manage composite video output
            if composite_frames_out is not None:
                pil_image: Image = PIL.Image.fromarray(obj=rgb_image)
                if results.pose_landmarks:
                    _draw_body(pil_draw=ImageDraw.Draw(pil_image), landmarks=frame_landmarks,
                               visibility=frame_visibility, width=rgb_image.shape[1], height=rgb_image.shape[0])
                composite_frames_out.consume(np.asarray(pil_image))

    if out_landmarks_list:
        out_landmarks = np.stack(out_landmarks_list)
        out_visibility = np.stack(out_visibility_list)
    else:
        out_landmarks = np.empty((0, MEDIAPIPE_POSE_LANDMARKS_COUNT, 3), dtype=np.float32)
        out_visibility = np.empty((0, MEDIAPIPE_POSE_LANDMARKS_COUNT, 2), dtype=np.float32)

    return out_landmarks, out_visibility


#
# MAIN
#
if __name__ == '__main__':
    import argparse
    import os

    from .common import video_info_ext

    parser = argparse.ArgumentParser(
        description="Uses mediapipe to extract the body landmarks data from the frames of a video.")

    parser.add_argument('--inframes', '--invideo',
                        help='Path to a video or image directory providing the frames with the body of a person.'
                             ' For a video, the frame rate is read from the file and the landmarks are tracked across'
                             ' frames (MediaPipe VIDEO mode).'
                             ' For an image directory, every frame is processed independently (MediaPipe IMAGE mode).',
                        required=True, type=str)

    parser.add_argument('--outlandmarks',
                        help='Path to the output numpy array of size [N][33][3],'
                             ' where N is the number of video frames,'
                             ' 33 are the number of landmarks of the [MediaPipe](https://mediapipe.dev) Pose Landmarker,'
                             ' and 3 is to store <x,y,z> 3D coords.'
                             ' As for MediaPipe, x and y are normalized in the range [0,1] by the frame width and height.'
                             ' Inside a frame, if no bodies are detected, all values are NaN!'
                             ' If more bodies are detected, only the first in the mediapipe list is used.',
                        required=True, type=str)

    parser.add_argument('--outvisibility',
                        help='Path to the output numpy array of size [N][33][2],'
                             ' storing the MediaPipe <visibility,presence> scores of each landmark, in range [0,1].'
                             ' Inside a frame, if no bodies are detected, all values are NaN!',
                        required=False, type=str)

    parser.add_argument('--outcompositeframes', '--outcompositevideo',
                        help='Path to a videofile or directory for image files. Will have the same resolution and content of the input frames,'
                             ' plus the overlay of the body landmarks and skeleton.'
                             ' Red landmarks are on the left side of the body, blue on the right side, green is the nose.'
                             ' Landmarks with visibility below {} are drawn as hollow circles.'.format(COMPOSITE_VISIBILITY_THRESHOLD),
                        required=False, type=str)

    #
    # Extract arguments
    args = parser.parse_args()

    source_path = args.inframes
    landmarkspath = args.outlandmarks
    visibilitypath = args.outvisibility
    compositeoutpath = args.outcompositeframes

    in_fps = video_info_ext(source_path)["fps"] if os.path.isfile(source_path) else None

    #
    print("Extracting body landmarks from '{}' and save into '{}'...".format(source_path, landmarkspath))
    with create_frame_producer(source_path) as frames_prod:

        # Instantiate the frame consumer for the overlay video, if requested
        frames_cons = create_frame_consumer(compositeoutpath) if compositeoutpath is not None else None

        try:
            landmarksdata, visibilitydata = extract_body_data(frames_in=frames_prod,
                                                              composite_frames_out=frames_cons,
                                                              fps=in_fps)
        finally:
            # Remember to close the frame consumer, if specified.
            if frames_cons is not None:
                frames_cons.close()

    # Save numpy arrays to a file
    for filepath, data in [(landmarkspath, landmarksdata), (visibilitypath, visibilitydata)]:
        if filepath is not None:
            print("Saving numpy array to '{}'".format(filepath))
            np.save(file=filepath, arr=data, allow_pickle=False)

    print("Done.")
