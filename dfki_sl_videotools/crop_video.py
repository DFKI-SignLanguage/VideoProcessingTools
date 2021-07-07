import ffmpeg

from .common import *


def crop_video(input_video_path: str, input_json_path: str, output_video_path: str) -> None:
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
    x, y, w, h = get_bbox_from_json(input_json_path)
    stream = ffmpeg.crop(stream, x, y, w, h)
    stream = ffmpeg.output(stream, output_video_path)

    ffmpeg.run(stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crop a video at a specified rectangular area.')
    parser.add_argument('--invideo',
                        help='Path to the input videofile',
                        required=True)
    parser.add_argument('--inbounds',
                        help='Path to a JSON file containing the bounds information for cropping.'
                             ' Format is: { "x": int, "y": int, "width": int, "height": int}',
                        required=True)
    parser.add_argument('--outvideo',
                        help='Path for the output videofile, showing the cropped area',
                        required=True)

    args = parser.parse_args()

    crop_video(args.invideo, args.inbounds, args.outvideo)

    print("Done.")
