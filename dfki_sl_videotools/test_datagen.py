from .datagen import ImageFileProducer
from .datagen import VideoFrameProducer

from .common import video_info

import pkg_resources

import numpy as np

TEST_FRAMES_PATH = pkg_resources.resource_filename("dfki_sl_videotools.data", "frames")
TEST_VIDEO_PATH = pkg_resources.resource_filename("dfki_sl_videotools.data", "testvideo.mp4")


def test_image_files_production():

    with ImageFileProducer(directory_path=TEST_FRAMES_PATH) as prod:
        for frame in prod.frames():
            assert type(frame) == np.ndarray
            assert frame.shape[2] == 3


def test_video_frames_production():

    w, h, n_frames = video_info(TEST_VIDEO_PATH)

    with VideoFrameProducer(videofile_path=TEST_VIDEO_PATH) as prod:
        frame_count = 0
        for frame in prod.frames():
            assert type(frame) == np.ndarray
            assert frame.shape[0] == w
            assert frame.shape[1] == h
            assert frame.shape[2] == 3

            frame_count += 1

    assert frame_count == n_frames
