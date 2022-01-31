import pkg_resources
import os
import json

from .trim_video import trim_video
from .extract_face_bounds import extract_face_bounds
from .crop_video import crop_video
from .extract_face_data import extract_face_data

from .common import video_info

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


def test_cropping_pipeline(tmp_path):

    print("Cropping test output: " + str(tmp_path))

    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    #
    # Path to video with composite rectangle
    bbox_video_path = tmp_path / "bbox_video.mp4"

    #
    # Extract face bounds
    with create_frame_producer(TEST_VIDEO_PATH) as frame_prod:
        bounds = extract_face_bounds(frames_in=frame_prod)
        with open(os.path.join(tmp_path, "bounds.json"), "w") as boundsfile:
            json.dump(obj=bounds, fp=boundsfile, indent=4)
        bounds_x, bounds_y, bounds_w, bounds_h = bounds

    assert 0 <= bounds_x < video_w  # x
    assert 0 <= bounds_y < video_h  # y
    assert bounds_x + bounds_w <= video_w  # width
    assert bounds_y + bounds_h <= video_h  # height

    #
    # TODO -- test creation of video with bbox
    # output_video_path = str(bbox_video_path)

    #
    # Crop the video
    cropped_video_path = tmp_path / "cropped_video.mp4"

    with create_frame_producer(TEST_VIDEO_PATH) as prod,\
        create_frame_consumer(str(cropped_video_path)) as cons:

        crop_video(frames_producer=prod,
                   bounds_tuple=bounds,
                   frames_consumer=cons)

    cropped_w, cropped_h, cropped_n_frames = video_info(cropped_video_path)
    assert cropped_w - 2 < bounds_w < cropped_w + 2
    assert cropped_h - 2 < bounds_h < cropped_h + 2
    assert cropped_n_frames == n_frames


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
