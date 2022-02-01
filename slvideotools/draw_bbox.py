
from .datagen import FrameProducer, FrameConsumer
from .datagen import create_frame_producer, create_frame_consumer

from .common import bbox_from_dict

from typing import Tuple
import json

import numpy as np

from PIL.Image import Image
import PIL.Image
from PIL import ImageDraw


def draw_bbox(frames_in: FrameProducer, bbox: Tuple[int, int, int, int], frames_out: FrameConsumer):

    for frame in frames_in.frames():

        frame_pil: Image = PIL.Image.fromarray(obj=frame, mode='RGB')

        draw = ImageDraw.Draw(frame_pil)
        draw.rectangle(xy=[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline=(220, 10, 10))

        frames_out.consume(np.asarray(frame_pil))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Overlay a bounding box rectangle on video frames')
    parser.add_argument('--inframes',
                        help='Path to a video file or directory of frames.',
                        required=True)
    parser.add_argument('--inbounds',
                        help='Path to a JSON input file containing the bbox coordinates: [x, y, width, height]',
                        required=True)
    parser.add_argument('--outframes',
                        help='Path to a video name or existing folder to store the output frames.',
                        required=True)

    args = parser.parse_args()

    with create_frame_producer(args.inframes) as frames_prod,\
        create_frame_consumer(args.outframes) as frame_cons,\
        open(args.inbounds, 'r') as bounds_file:

        bounds_dict = json.load(bounds_file)
        bounds = bbox_from_dict(bounds_dict)

        draw_bbox(frames_in=frames_prod, frames_out=frame_cons, bbox=bounds)
