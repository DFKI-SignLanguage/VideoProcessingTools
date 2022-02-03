import numpy as np
import json

import ffmpeg

from typing import Tuple


def clamp(x, lo, hi):
    """clamp x to the [lo,hi] range"""
    return lo if x < lo else hi if x > hi else x


def video_info(video_path: str) -> Tuple[int, int, int]:
    """
    Uses the ffmpeg.probe function to retrieve information about a video file.

    :param video_path: Path to a valid video file
    :return: A 3-tuple with integers for (width, height, number_of_frames)
    """

    #
    # Fetch video info
    info = ffmpeg.probe(video_path)
    # Get the list of all video streams
    video_streams = [stream for stream in info['streams'] if stream['codec_type'] == 'video']
    if len(video_streams) == 0:
        raise BaseException("No video streams found in file '{}'".format(video_path))

    # retrieve the first stream of type 'video'
    info_video = video_streams[0]

    video_w = info_video['width']
    video_h = info_video['height']
    n_frames = int(info_video['nb_frames'])

    return video_w, video_h, n_frames


# https://en.wikipedia.org/wiki/Point_reflection
def reflect(c: Tuple[float, float], P: Tuple[float, float]) -> Tuple[float, float]:
    """ 
        Make a reflection of a point P according to the center c

        Args :
            c : coordinate of the center
            P : coordinate of the reflecting point
        
        Returns:
            P_prime : coordinates of the mirror of P through c

    """
    P_x_prime = 2*c[0] - P[0]
    P_y_prime = 2*c[1] - P[1]

    return P_x_prime, P_y_prime


def bbox_to_dict(x: Tuple[int, int, int, int]) -> dict:
    """
        Format the numpy array bbox to json

        Args:
            x : numpy array bbox

        :returns a dictionary with keys "x", "y", "width", "height"
    """
    bbox = dict()
    bbox["x"] = int(x[0])
    bbox["y"] = int(x[1])
    bbox["width"] = int(x[2])
    bbox["height"] = int(x[3])

    return bbox


def bbox_from_dict(bounds_dict: dict) -> Tuple[int, int, int, int]:

    return bounds_dict["x"], bounds_dict["y"], bounds_dict["width"], bounds_dict["height"]
