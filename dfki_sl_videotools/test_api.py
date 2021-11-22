import ffmpeg

import pkg_resources

from .trim_video import trim_video
from .extract_face_bounds import extract_face_bounds
from .crop_video import crop_video
from .extract_face_data import extract_face_data

import os

TEST_VIDEO_PATH = pkg_resources.resource_filename("dfki_sl_videotools.data", "testvideo.mp4")


def test_face_pipeline(tmp_path):

    #
    # Fetch video info
    info = ffmpeg.probe(TEST_VIDEO_PATH)
    # retrieve the first stream of type 'video'
    info_video = [stream for stream in info['streams'] if stream['codec_type'] == 'video'][0]

    video_w = info_video['width']
    video_h = info_video['height']
    n_frames = int(info_video['nb_frames'])

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

    info_trimmed_video = ffmpeg.probe(trimmed_video_path)
    info_trimmed_video = [stream for stream in info_trimmed_video['streams'] if stream['codec_type'] == 'video'][0]
    trimmed_n_frames = int(info_trimmed_video['nb_frames'])
    assert trimmed_n_frames == eframe - sframe + 1

    #
    # Extract face bounds
    bounds = extract_face_bounds(input_video_path=str(trimmed_video_path))
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

    info_cropped_video = ffmpeg.probe(cropped_video_path)
    for stream in info_cropped_video['streams']:
        if stream['codec_type'] == 'video':
            # The resolution of the cropped video might not be exactly the one of the crop request:
            # it depends of the codec restrictions.
            cropped_video_w = stream['width']
            cropped_video_h = stream['height']
            assert cropped_video_w - 2 < bounds_w < cropped_video_w + 2
            assert cropped_video_h - 2 < bounds_h < cropped_video_h + 2

    pass


def test_face_data_extraction(tmp_path):

    import numpy as np

    print("Writing face extraction data to ", tmp_path)

    #
    # Fetch video info
    info = ffmpeg.probe(TEST_VIDEO_PATH)
    # retrieve the first stream of type 'video'
    info_video = [stream for stream in info['streams'] if stream['codec_type'] == 'video'][0]

    video_w = info_video['width']
    video_h = info_video['height']
    n_frames = int(info_video['nb_frames'])

    face_data = extract_face_data(videofilename=TEST_VIDEO_PATH, outcompositevideo=os.path.join(tmp_path, "mediapipe_face_landmarks.mp4"))

    assert type(face_data) == np.ndarray

    data_n_frames, data_n_landmarks, data_n_coords = face_data.shape

    assert data_n_frames == n_frames
    assert data_n_landmarks == 468
    assert data_n_coords == 3
