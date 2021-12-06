import json

import ffmpeg
import sys
import logging
# from common import *

from typing import Tuple
import os
import shutil


def make_output_directoy(dir, name):
    director_name = os.path.splitext(name)[0]
    path_n = os.path.join(dir, director_name)
    if os.path.isdir(path_n):
        logging.warning("Directory already exist {}".format(path_n))
        return path_n
        #shutil.rmtree(path_n)
        #logging.info("Creating dir {}".format(path_n))
        #os.mkdir(path_n)
    else:
        os.mkdir(path_n)
        logging.info("Creating dir {}".format(path_n))

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
    if not os.path.isfile(output_video_path):
        stream = ffmpeg.input(input_video_path)
        x, y, w, h = bounds_tuple
        # See here to force exact cropping size: https://stackoverflow.com/questions/61304686/ffmpeg-cropping-size-is-always-wrong
        stream = ffmpeg.crop(stream, x, y, w, h, exact=0)
        stream = ffmpeg.output(stream, output_video_path)

        ffmpeg.run(stream)
    else:
        logging.warning("This video already exist")
        print("This video already exist: {}".format(output_video_path))


if __name__ == '__main__':
    import argparse
    import datetime

    logging.basicConfig(filename='log_crop_video_batch{}.log'.format(datetime.datetime.now().strftime('%M_%S_%d_%m_%Y')),
                        level=logging.DEBUG)
    logging.info("=============== Start at {} ==================".format(datetime.datetime.now()))

    parser = argparse.ArgumentParser(description='Interactively edit a rectangular area in a video. '
                                                 'Useful to manually set cropping bounds.')
    parser.add_argument('--inv_dir',
                        help='Path to the input video file',
                        required=True)
    parser.add_argument('--in_json',
                        help='Path to a JSON file containing the bounds information for cropping.'
                             ' Format is: { "x": int, "y": int, "width": int, "height": int}',
                        required=True)
    parser.add_argument('--ov_dir',
                        help='Path for the output video file, showing the cropped area',
                        required=True)
    parser.add_argument('--ov_name',
                        help='name with extension, ex patient.mp4 or patient.m4v',
                        required=True)

    args = parser.parse_args()

    with open(args.in_json, "r") as json_file:
        bounds_dict = json.load(json_file)
        # NOTE: inbound json format {"name.m4v": [{"crop_area": {"x":0, "y": 0, "width": 377, "height": 570},
        # "metadata": { "nb_frames": "110431", "is_crop_selected": true, "v_width": 718, "v_height": 576}}], ...}
        # video_name = os.path.basename(args.invideo)
        num_videos_indx = 0
        for video_name in bounds_dict:
            num_videos_indx += 1

            in_video = os.path.join(args.inv_dir, video_name)
            logging.info("VIDEO NUMBER {0}:".format(num_videos_indx))

            if not os.path.isfile(in_video):
                logging.debug("{} is not in the input dir".format(in_video))
                continue

            try:
                # print("bounds_dict: {}".format(bounds_dict))
                logging.info("Video name: {0}".format(video_name))

                data_ = bounds_dict["{0}".format(video_name)][0]
                crop_dim = data_['crop_area']
                metadata = data_['metadata']

                logging.info("crop_dim {}".format(crop_dim))
                logging.info("metadata {}".format(metadata))

                bounds = crop_dim["x"], crop_dim["y"], crop_dim["width"], crop_dim["height"]
                # create a directory for output
                path_n = make_output_directoy(args.ov_dir, video_name)
                video_out = os.path.join(path_n, args.ov_name)
                crop_video(in_video, bounds, video_out)


            except json.JSONDecodeError as je:
                logging.error("JSON Decoder Error:{0}".format(je))
                sys.exit(1)
            except ffmpeg.Error as e:
                logging.error("FFMPEG Error: {0}".format(e.stderr), file=sys.stderr)
                sys.exit(1)

    logging.info("=============== End at {} ==================".format(datetime.datetime.now()))
