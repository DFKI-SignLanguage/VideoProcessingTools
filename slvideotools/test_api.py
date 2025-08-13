import pytest

import pkg_resources
import os
import json

from .trim_video import trim_video
from .extract_face_bounds import extract_face_bounds
from .draw_bbox import draw_bbox
from .crop_video import crop_video
from .extract_face_data import extract_face_data
from .compute_motion_energy import compute_motion_energy

from .common import video_info
from .common import bbox_to_dict

from .datagen import create_frame_consumer
from .datagen import create_frame_producer


TEST_VIDEO_PATH = pkg_resources.resource_filename("slvideotools.data", "testvideo.mp4")


def test_trimming(tmp_path):
    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    #
    # Trim the video
    trimmed_video_path = tmp_path / "trimmed_video.mp4"
    # Take approximately 80% of the central part of the video
    sframe = int(n_frames * 0.4)
    eframe = sframe + int(n_frames * 0.2)
    trim_video(input_path=TEST_VIDEO_PATH,
               output_path=str(trimmed_video_path),
               start_frame=sframe,
               end_frame=eframe)

    trimmed_w, trimmed_h, trimmed_n_frames = video_info(trimmed_video_path)
    assert trimmed_w == video_w
    assert trimmed_h == video_h
    assert trimmed_n_frames == eframe - sframe + 1


#
# Generates all face detection methods
@pytest.fixture(params=["mediapipe", "mtcnn"])
def face_extraction_method(request) -> str:
    return request.param


def test_face_detection_pipeline(tmp_path, face_extraction_method):

    print("Face detection test output: " + str(tmp_path))

    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    #
    # Extract face bounds
    with create_frame_producer(TEST_VIDEO_PATH) as frame_prod:

        face_bounds = extract_face_bounds(frames_in=frame_prod, head_focus=True, method=face_extraction_method)

        with open(os.path.join(tmp_path, "bounds-{}.json".format(face_extraction_method)), "w") as boundsfile:
            json.dump(obj=bbox_to_dict(face_bounds), fp=boundsfile, indent=4)

    bounds_x, bounds_y, bounds_w, bounds_h = face_bounds

    assert 0 <= bounds_x < video_w  # x
    assert 0 <= bounds_y < video_h  # y
    assert bounds_x + bounds_w <= video_w  # width
    assert bounds_y + bounds_h <= video_h  # height

    #
    # Test the creation of video with bbox
    bbox_video_path = tmp_path / "bbox_video-{}.mp4".format(face_extraction_method)

    with create_frame_producer(TEST_VIDEO_PATH) as frame_prod,\
        create_frame_consumer(str(bbox_video_path)) as frame_cons:

        draw_bbox(frames_in=frame_prod, bbox=face_bounds, frames_out=frame_cons)


def test_cropping(tmp_path):

    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    # Set an initial bounding box
    x = int(video_w * 0.25)
    y = int(video_h * 0.23)
    w = x + int(video_w * 0.5)
    h = y + int(video_h * 0.45)

    # Try several cropping sizes (to test stability on even sizes)
    for i in range(4):

        bounds = x, y, w, h

        #
        # Crop the video
        cropped_video_path = tmp_path / "cropped_video-{}.mov".format(i)

        with create_frame_producer(TEST_VIDEO_PATH) as prod,\
            create_frame_consumer(str(cropped_video_path)) as cons:

            crop_video(frames_producer=prod,
                       bounds_tuple=bounds,
                       frames_consumer=cons)

        cropped_w, cropped_h, cropped_n_frames = video_info(cropped_video_path)
        assert w-1 <= cropped_w <= w
        assert h-1 <= cropped_h <= h
        assert cropped_n_frames == n_frames

        # reduce the cropping size
        w -= 1
        h -= 1


def test_face_data_extraction(tmp_path):

    import numpy as np

    print("Writing face extraction data to ", tmp_path)

    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    with create_frame_producer(dir_or_video=TEST_VIDEO_PATH) as frame_prod,\
            create_frame_consumer(dir_or_video=os.path.join(tmp_path, "landmarks_composite.mp4")) as frame_cons:

        landmarks_data, nosetip_data, facerotation_data, facescale_data = extract_face_data(
            frames_in=frame_prod,
            composite_frames_out=frame_cons,
            normalize_landmarks=True)

    assert type(landmarks_data) == np.ndarray
    assert type(nosetip_data) == np.ndarray
    assert type(facerotation_data) == np.ndarray
    assert type(facescale_data) == np.ndarray

    assert len(landmarks_data.shape) == 3
    assert landmarks_data.shape[0] == n_frames
    assert landmarks_data.shape[1] == 468
    assert landmarks_data.shape[2] == 3
    assert landmarks_data.dtype == np.float32

    assert len(nosetip_data.shape) == 2
    assert nosetip_data.shape[0] == n_frames
    assert nosetip_data.shape[1] == 3
    assert nosetip_data.dtype == np.float32

    assert len(facerotation_data.shape) == 3
    assert facerotation_data.shape[0] == n_frames
    assert facerotation_data.shape[1] == 3
    assert facerotation_data.shape[2] == 3
    assert facerotation_data.dtype == np.float32

    assert len(facescale_data.shape) == 1
    assert facescale_data.shape[0] == n_frames
    assert facescale_data.dtype == np.float32


def test_motion_energy_computation(tmp_path):

    import numpy as np

    print("Writing face extraction data to ", tmp_path)

    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    # Limit to a few keyframes to speed-up the test
    FRAME_START = 10
    FRAME_END = 100

    with create_frame_producer(dir_or_video=TEST_VIDEO_PATH) as frame_prod:

        out_video_path = os.path.join(tmp_path, "motion_energy.mp4")

        motion_curve = compute_motion_energy(
            frames_in=frame_prod,
            frame_start=FRAME_START,
            frame_end=FRAME_END,
            out_video_path=out_video_path,
            normalize=True
        )

        assert len(motion_curve) > 0
        assert type(motion_curve) == np.ndarray
        assert type(motion_curve[0]) == np.float64

        assert np.max(motion_curve) == 1.0

        assert len(motion_curve) == FRAME_END - FRAME_START - 1

        out_video_w, out_video_h, out_video_n_frames = video_info(out_video_path)

        assert out_video_w == video_w
        assert out_video_h == video_h
        assert out_video_n_frames == len(motion_curve)
