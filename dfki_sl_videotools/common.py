import numpy as np
import json

import ffmpeg

from typing import Tuple


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


# https://stackoverflow.com/questions/41976245/crop-rectangle-in-opencv-python
def crop_bbox(image, x, y, w, h):
    """ 
        Crop an image from a bbox
    
    """
    return image[y:y+h, x:x+w]


# https://en.wikipedia.org/wiki/Point_reflection
def reflect(c, P):
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

    return np.array([P_x_prime, P_y_prime]).astype(int)


def get_bbox_pts(nose, rshoulder):
    """ 
        Get the upper left and the lower right of the ROI

        Args:
            nose : nose coordinates
            rshoulder : a shoulders coordinates, can be either left or right
        Returns: 
            bbox : a numpy array containing the upper left corner and the lower right corner coordinates

    """

    pt1 = reflect(nose, rshoulder)
    pt2 = rshoulder 

    return np.array([pt1, pt2])


"""def expand_bbox(x,y,w,h):
    pt1 = np.array([x,y])
    pt2 = np.array([x,y+h])
    pt3 = np.array([x+w,y])
    pt4 = np.array([x+w,y+h])
    return np.array([pt1,pt2,pt3,pt4])
"""


# https://stackoverflow.com/questions/23110383/how-to-dynamically-build-a-json-object-with-python
def format_json_bbox(x) -> str:
    """
        Format the numpy array bbox to json

        Args:
            x : numpy array bbox

        Returns:
            json(bbox) : bounding box
    """
    bbox = dict()
    bbox["x"] = int(x[0])
    bbox["y"] = int(x[1])
    bbox["width"] = int(x[2])
    bbox["height"] = int(x[3])

    return json.dumps(bbox)


def get_bbox_from_json(input_json_path):
    """ 
        Read bounding box coords from json file

        Args:
            input_json_path : path to the json file
        
        Returns:
        x,y,w,h = bbox["x"],bbox["y"],bbox["width"],bbox["height"]
    """
    file_in = open(input_json_path, "r")
    line = file_in.readlines()[0]
    bbox = json.loads(line)
    file_in.close
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

    return x, y, w, h
