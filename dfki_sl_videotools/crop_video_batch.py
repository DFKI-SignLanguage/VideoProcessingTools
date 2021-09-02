import json

import ffmpeg
import sys
from .common import *

from typing import Tuple
import os

def make_output_directoy(dir, name):
    director_ = os.path.splitext(name)[0]
    path_n = os.path.join(dir, director_)
    if not os.path.isfile(path_n):
        os.mkdir(path_n)
    else:
        print("This directory already exists")

    return path_n
def crop_video(input_video_path: str, bounds_tuple: Tuple[int, int, int, int], output_video_path: str) -> None:
    """
        Crop a video from coordinates defined in a json file

        Args:
            input_video_path: path to the input video
            input_json_path: path to the json file
            output_video_path: path to the output video

        Returns:
             None
    """
    stream = ffmpeg.input(input_video_path)
    x, y, w, h = bounds_tuple
    # See here to force exact cropping size: https://stackoverflow.com/questions/61304686/ffmpeg-cropping-size-is-always-wrong
    stream = ffmpeg.crop(stream, x, y, w, h, exact=0)
    stream = ffmpeg.output(stream, output_video_path)

    ffmpeg.run(stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Interactively edit a rectangular area in a video. '
                                                 'Useful to manually set cropping bounds.')
    parser.add_argument('--inv_dir',
                        help='Path to the input videofile',
                        required=True)
    parser.add_argument('--in_json',
                        help='Path to a JSON file containing the bounds information for cropping.'
                             ' Format is: { "x": int, "y": int, "width": int, "height": int}',
                        required=True)
    parser.add_argument('--ov_dir',
                        help='Path for the output videofile, showing the cropped area',
                        required=True)


    args = parser.parse_args()

    with open(args.in_json, "r") as json_file:
        bounds_dict = json.load(json_file)
        # NOTE: inbound json format {"name.m4v": [{"crop_area": {"x":0, "y": 0, "width": 377, "height": 570},
        # "metadata": { "nb_frames": "110431", "is_crop_selected": true, "v_width": 718, "v_height": 576}}], ...}
        #video_name = os.path.basename(args.invideo)
        num_videos_indx = 0
        for video in bounds_dict:
            num_videos_indx+=1
            print("VIDEO NUMBER {0}:".format(num_videos_indx))

            try:
                #print("bounds_dict: {}".format(bounds_dict))
                print("Video name: {0}".format(video))
                data_ = bounds_dict["{0}".format(video)][0]
                crop_dim= data_['crop_area']
                print("crop_dim {}".format(crop_dim))
                bounds = crop_dim["x"], crop_dim["y"], crop_dim["width"], crop_dim["height"]

                #create a directory for output
                in_video = os.path.join(args.inv_dir,video)

                #path_n = make_output_directoy(args.ov_dir, video)
                video_out = os.path.join(args.ov_dir, "Left.m4v")

                #crop_video(in_video, bounds, video_out)

            except json.JSONDecodeError as je:
                print("JSON Decoder Error:{0}".format(je))
                sys.exit(1)
            except ffmpeg.Error as e:
                print("FFMPEG Error: {0}".format(e.stderr), file=sys.stderr)
                sys.exit(1)

    print("Done.")
