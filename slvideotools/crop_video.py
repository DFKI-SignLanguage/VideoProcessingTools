from .common import *
from .datagen import create_frame_consumer, create_frame_producer
from .datagen import FrameConsumer, FrameProducer

from typing import Tuple


def crop_video(frames_producer: FrameProducer, bounds_tuple: Tuple[int, int, int, int], frames_consumer: FrameConsumer) -> None:
    """ 
        Crop a video from coordinates defined in a json file

        Args:
            frames_producer: frames source
            bounds_tuple: bounds to crop. In order: x, y, w, h
            frames_consumer: frames destination
        
        Returns:
             None
    """

    x, y, w, h = bounds_tuple

    for frame in frames_producer.frames():

        cropped_frame = frame[y:y+h, x:x+w, :]

        frames_consumer.consume(cropped_frame)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crop a video at a specified rectangular area.')
    parser.add_argument('--inframes',
                        help='Path to the input videofile o directory',
                        required=True)
    parser.add_argument('--inbounds',
                        help='Path to a JSON file containing the bounds information for cropping.'
                             ' Format is: { "x": int, "y": int, "width": int, "height": int}',
                        required=True)
    parser.add_argument('--outframes',
                        help='Path for the output videofile or directory',
                        required=True)

    args = parser.parse_args()

    input_frames_path = args.inframes
    output_frames_path = args.outframes

    with open(args.inbounds, "r") as json_file,\
            create_frame_producer(input_frames_path) as frames_prod, \
            create_frame_consumer(output_frames_path) as frames_cons:

        bounds_dict = json.load(json_file)
        bounds = bounds_dict["x"], bounds_dict["y"], bounds_dict["width"], bounds_dict["height"]

        crop_video(frames_producer=frames_prod, bounds_tuple=bounds, frames_consumer=frames_cons)

    print("Done.")
