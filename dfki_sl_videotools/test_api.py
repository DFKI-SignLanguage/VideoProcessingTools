import pkg_resources
import os
import json

from .trim_video import trim_video
from .extract_face_bounds import extract_face_bounds
from .crop_video import crop_video
from .extract_face_data import extract_face_data

from .common import video_info


TEST_VIDEO_PATH = pkg_resources.resource_filename("dfki_sl_videotools.data", "testvideo2.mp4")


def test_trimming(tmp_path):
    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    #
    # Trim the video
    trimmed_video_path = tmp_path / "trimmed_video.mp4"
    # Take approximately 80% of the central part of the video
    sframe = int(n_frames * 0.1)
    eframe = sframe + int(n_frames * 0.8)
    trim_video(input_path=TEST_VIDEO_PATH,
               output_path=str(trimmed_video_path),
               start_frame=sframe,
               end_frame=eframe)

    trimmed_w, trimmed_h, trimmed_n_frames = video_info(trimmed_video_path)
    assert trimmed_w == video_w
    assert trimmed_h == video_h
    assert trimmed_n_frames == eframe - sframe + 1


def test_cropping_pipeline(tmp_path):

    #
    # Fetch video info
    video_w, video_h, n_frames = video_info(TEST_VIDEO_PATH)

    #
    # Extract face bounds
    bounds = extract_face_bounds(input_video_path=str(TEST_VIDEO_PATH))
    with open(os.path.join(tmp_path, "bounds.json"), "w") as boundsfile:
        json.dump(obj=bounds, fp=boundsfile, indent=4)
    bounds_x, bounds_y, bounds_w, bounds_h = bounds

    assert 0 <= bounds_x < video_w  # x
    assert 0 <= bounds_y < video_h  # y
    assert bounds_x + bounds_w <= video_w  # width
    assert bounds_y + bounds_h <= video_h  # height

    #
    # Crop the video
    cropped_video_path = tmp_path / "cropped_video.mp4"

    crop_video(input_video_path=TEST_VIDEO_PATH,
               bounds_tuple=bounds,
               output_video_path=str(cropped_video_path))

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

    face_data = extract_face_data(videofilename=TEST_VIDEO_PATH,
                                  outcompositevideo=os.path.join(tmp_path, "mediapipe_face_landmarks.mp4"),
                                  normalize_landmarks=True)

    assert type(face_data) == np.ndarray

    data_n_frames, data_n_landmarks, data_n_coords = face_data.shape

    assert data_n_frames == n_frames
    assert data_n_landmarks == 468
    assert data_n_coords == 3
