import ffmpeg


def trim_video(input_path: str, output_path: str, start_frame: int, end_frame: int) -> None:
    """Triming function for mp4 video source.

    Args:
        input_path: path to the input video.
        output_path: path to the output video.
        start_frame : starting frame number.
        end_frame : ending frame number.

    Returns:
        None

    """
    stream = ffmpeg.input(input_path)
    stream = ffmpeg.trim(stream, start_frame=start_frame-1, end_frame=end_frame)
    # to fix the still frame https://github.com/kkroening/ffmpeg-python/issues/155
    stream = ffmpeg.filter(stream, 'setpts', expr='PTS-STARTPTS')

    stream = ffmpeg.output(stream, output_path)
    ffmpeg.run(stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Trims a video file.')
    parser.add_argument('--invideo', help='Input video filepath', required=True, type=str)
    parser.add_argument('--outvideo', help='Output video filepath', required=True, type=str)
    parser.add_argument('--startframe', help='First frame to retain (counting from 1)', required=True, type=int)
    parser.add_argument('--endframe', help='Last frame to retain (counting from 1)', required=True, type=int)

    args = parser.parse_args()

    print("Trimming MP4 file '{}' frames range {}-{} into '{}'".format(args.invideo,
                                                                       args.startframe,
                                                                       args.endframe,
                                                                       args.outvideo))

    trim_video(input_path=args.invideo,
               output_path=args.outvideo,
               start_frame=args.startframe,
               end_frame=args.endframe)
    print("Done.")
