from .datagen import ImageDirFrameProducer
from .datagen import VideoFrameProducer
from .datagen import ImageDirFrameConsumer
from .datagen import VideoFrameConsumer

from .datagen import FrameProducer
from .datagen import FrameConsumer
from .datagen import create_frame_producer
from .datagen import create_frame_consumer

from .common import video_info
from .common import video_info_ext

import json

from importlib.resources import files

import numpy as np

TEST_FRAMES_PATH = str(files("slvideotools.data") / "frames")
TEST_VIDEO_PATH = str(files("slvideotools.data") / "testvideo.mp4")


def test_image_files_production(tmp_path):
    """Test the producer (and the consumer) reading (and writing) frames from (to) a directory."""

    print("Generating frames in path " + str(tmp_path))

    with ImageDirFrameProducer(source_dir=TEST_FRAMES_PATH) as prod,\
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

    with VideoFrameProducer(video_in=TEST_VIDEO_PATH) as prod,\
         VideoFrameConsumer(video_out=video_path) as cons:

        frame_count = 0
        for frame in prod.frames():
            assert type(frame) == np.ndarray
            assert frame.shape[0] == h
            assert frame.shape[1] == w
            assert frame.shape[2] == 3
            assert frame.dtype == np.uint8

            cons.consume(frame)

            frame_count += 1

    assert frame_count == n_frames

    w2, h2, n_frames2 = video_info(video_path)

    assert w == w2
    assert h == h2
    assert n_frames == n_frames2


def test_video_consumer_ffmpeg_params(tmp_path):
    """Test that the ffmpeg_params of the VideoFrameConsumer are used, e.g., to set the output frame rate."""

    video_path = str(tmp_path / "video_50fps.mp4")
    target_fps = 50

    with VideoFrameProducer(video_in=TEST_VIDEO_PATH) as prod,\
         VideoFrameConsumer(video_out=video_path, ffmpeg_params={"framerate": target_fps}) as cons:
        for frame in prod.frames():
            cons.consume(frame)

    in_info = video_info_ext(TEST_VIDEO_PATH)
    out_info = video_info_ext(video_path)

    assert out_info["fps"] == target_fps
    assert out_info["width"] == in_info["width"]
    assert out_info["height"] == in_info["height"]
    assert out_info["n_frames"] == in_info["n_frames"]


def test_video_info_ext():
    """Test that the extended video info is consistent with video_info() and adds a valid frame rate."""

    w, h, n_frames = video_info(TEST_VIDEO_PATH)
    info = video_info_ext(TEST_VIDEO_PATH)

    assert set(info.keys()) == {"width", "height", "n_frames", "fps"}
    assert info["width"] == w
    assert info["height"] == h
    assert info["n_frames"] == n_frames
    assert isinstance(info["fps"], float)
    assert info["fps"] > 0

    # Must be serializable as JSON
    assert json.loads(json.dumps(info)) == info


def test_factory_methods():

    dir_prod = create_frame_producer(dir_or_video=TEST_FRAMES_PATH)
    assert isinstance(dir_prod, FrameProducer)
    dir_prod.close()

    video_prod = create_frame_producer(dir_or_video=TEST_VIDEO_PATH)
    assert isinstance(video_prod, FrameProducer)
    video_prod.close()

    dir_cons = create_frame_consumer(dir_or_video=TEST_FRAMES_PATH)
    assert isinstance(dir_cons, FrameConsumer)
    dir_cons.close()

    video_cons = create_frame_consumer(dir_or_video=TEST_VIDEO_PATH)
    assert isinstance(video_cons, FrameConsumer)
    video_cons.close()
