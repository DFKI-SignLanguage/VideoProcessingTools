"""Automatically extract dense optical flow for given input video.

https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
"""

from pathlib import Path

import numpy as np
import cv2

from slvideotools.datagen import create_frame_consumer, create_frame_producer, FrameProducer

from typing import Optional


def cartesian_to_polar_np(x, y):
    """Converts pairs of x/y coordinates from cartesian to polar space."""
    radius = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return radius, angle


def compute_motion_energy(
    frames_in: FrameProducer,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    out_video_path: Optional[str] = None,
    normalize: bool = False
) -> np.ndarray:

    out_energy_data = []  # Will accumulate the dense optical flow between frames
    out_motion_video_frames = []  # (If enabled) Accumulates the frames for output RGB video

    # Cache memory of the previous frame
    frame_prev = None

    frame_counter = 0  # counts the number of input processed frames

    # For each frame in the source
    for frame_idx, frame in enumerate(frames_in.frames()):

        assert type(frame) == np.ndarray
        width, height, depth = frame.shape
        assert depth == 3

        # print(f"{frame_idx} ", end="", flush=True)

        # Skip the initial frames
        if frame_start is not None:
            if frame_idx < frame_start:
                continue

        # Or stop analysing frames
        if frame_end is not None:
            if frame_idx >= frame_end:
                break

        frame_counter += 1

        # If it is the first valid frame, memorize and skip to the next frame
        if frame_prev is None:
            frame_prev = frame
            continue

        # Compute dense optical flow between two frames
        frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame_prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute the magnitude and the direction of the flow
        # magnitude, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude, ang = cartesian_to_polar_np(flow[..., 0], flow[..., 1])

        # Converts the optical direction information to HSV color space.
        # Just for visual feedback associating color to direction.
        if out_video_path is not None:
            hsv = np.zeros_like(frame)
            hsv[..., 0] = ang * 180 / np.pi / 2  # map the flow direction to Hue
            hsv[..., 1] = 255  # Initialize the Saturation to max.
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # map the flow magnitude to Value
            rgb_video_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            out_motion_video_frames.append(rgb_video_frame)

        # Sum up the magnitude
        motion_energy = magnitude.sum()
        out_energy_data.append(float(motion_energy))

        frame_prev = frame

    print(f"Processed {frame_counter} frames")

    #
    # Convert to np.ndarray data structure
    out_energy_data = np.array(out_energy_data)

    #
    # Normalize the output curve, so that the max value is 1.0
    if normalize:
        out_energy_data = out_energy_data / np.max(out_energy_data)

    #
    # Consistency check
    if len(out_energy_data) > 0:
        assert type(out_energy_data[0]) == np.float64

    if frame_start is not None and frame_end is not None:
        assert frame_counter == frame_end - frame_start

    assert len(out_energy_data) == frame_counter - 1

    #
    # (Optionally) Write a video with the visualization of the difference between frames
    if out_video_path is not None:
        with create_frame_consumer(str(out_video_path)) as stream:
            for frame in out_motion_video_frames:
                stream.consume(frame)

    # Return data curve
    return out_energy_data


#
# MAIN
#
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculates the 'motion energy' of the input frame sequence."
                    " The motion energy is a mono-dimensional curve."
                    " Each sample is calculated by first computing the optical flow between consecutive frames"
                    " and then summing up the magnitude of each flow vector."
                    " The resulting motion curve contains one frame less than the input sequences."
                    " The output curve is a mono-dimensional numpy array."
                    " Optionally, a debug video can be generated, with color associated to the motion direction and"
                    " motion magnitude mapped to color intensity."
    )

    parser.add_argument(
        "--invideo", "-i", type=str, help="Input video.", required=True
    )
    parser.add_argument(
        "--outmotioncurve", "-o",
        type=str,
        help="Path to the output motion energy data.",
        required=True,
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If specified, normalizes the energy curve in the range [0,1] before saving it.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        help="Start frame included in the motion energy computation (default 0: first frame).",
        required=False,
        default=None
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="End frame (excluded) in the motion energy computation (default: last frame of the video).",
        required=False,
        default=None
    )

    parser.add_argument("--output-video", "-ov", type=str, help="Video output of the file.", required=False)

    args = parser.parse_args()

    in_video_path = Path(args.invideo)

    if not in_video_path.exists():
        raise Exception(f"Input video file '{in_video_path}' doesn't  exist")

    out_curve_path = Path(args.outmotioncurve)
    out_video_path = Path(args.output_video) if args.output_video is not None else None

    with create_frame_producer(dir_or_video=str(in_video_path)) as frame_prod:

        motion_curve = compute_motion_energy(
            frames_in=frame_prod,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            out_video_path=out_video_path,
            normalize=args.normalize
        )

    print("Saving numpy array to '{}'".format(out_curve_path))
    np.save(file=out_curve_path, arr=motion_curve, allow_pickle=False)

    print("All done.")
