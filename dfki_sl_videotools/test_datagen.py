from .datagen import ImageDirFrameProducer
from .datagen import VideoFrameProducer
from .datagen import ImageDirFrameConsumer
from .datagen import VideoFrameConsumer

from .common import video_info

import pkg_resources

import numpy as np

TEST_FRAMES_PATH = pkg_resources.resource_filename("dfki_sl_videotools.data", "frames")
TEST_VIDEO_PATH = pkg_resources.resource_filename("dfki_sl_videotools.data", "testvideo.mp4")


def test_image_files_production(tmp_path):
    """Test the producer (and the consumer) reading (and writing) frames from (to) a directory."""

    print("Generating frames in path " + str(tmp_path))

    with ImageDirFrameProducer(directory_path=TEST_FRAMES_PATH) as prod,\
         ImageDirFrameConsumer(dest_dir=str(tmp_path)) as cons:

        # For each frame on the producer
        for frame in prod.frames():
            assert type(frame) == np.ndarray
            assert frame.shape[2] == 3
            assert frame.dtype == np.uint8

            # Feed the frame to the consumer
            cons.consume(frame=frame)


def test_video_frames_production(tmp_path):
    """Test the producer (and the consumer) reading (and writing) frames from (to) a video."""

    video_path = str(tmp_path / "produced_video.mp4")

    print("Generating video in path " + str(video_path))

    w, h, n_frames = video_info(TEST_VIDEO_PATH)

    with VideoFrameProducer(videofile_path=TEST_VIDEO_PATH) as prod,\
         VideoFrameConsumer(video_path=video_path) as cons:

        frame_count = 0
        for frame in prod.frames():
            assert type(frame) == np.ndarray
            assert frame.shape[0] == w
            assert frame.shape[1] == h
            assert frame.shape[2] == 3
            assert frame.dtype == np.uint8

            cons.consume(frame)

            frame_count += 1

    assert frame_count == n_frames

    w2, h2, n_frames2 = video_info(video_path)

    assert w == w2
    assert h == h2
    assert n_frames == n_frames2
