import pytest

import ffmpeg

import pkg_resources

TEST_VIDEO_PATH = pkg_resources.resource_filename("dfki_sl_videotools.data", "testvideo.mov")

from .extract_face_bounds import extract_face_bounds
from .crop_video import crop_video


def test_face_pipeline(tmp_path):

    info = ffmpeg.probe(TEST_VIDEO_PATH)

    info_video = None
    for stream in info['streams']:
        if stream['codec_type'] == 'video':
            info_video = stream
            break

    assert info_video is not None

    video_w = info_video['width']
    video_h = info_video['height']

    bounds = extract_face_bounds(input_video_path=TEST_VIDEO_PATH)
    bounds_x, bounds_y, bounds_w, bounds_h = bounds

    assert 0 <= bounds_x < video_w  # x
    assert 0 <= bounds_y < video_h  # y
    assert bounds_x + bounds_w <= video_w  # width
    assert bounds_y + bounds_h <= video_h  # height

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
