import cv2
import mediapipe as mp
import numpy as np
from .common import reflect
from .common import clamp

from .datagen import FrameProducer
from .datagen import create_frame_producer

import json
from typing import Tuple
from typing import Union

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def _get_head_region_info(image: np.ndarray, pose_detector: mp_pose.Pose)\
        -> Union[None, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
    """
        Find the ROI containing the face from an input image

        :param image: input BGR (cv2 format) image of type np.ndarray and shape [height, width, 3]
        :param pose_detector: the MediaPipe pose detector

        :returns List[nose, rshoulder, lshoulder]. Each element is a 2-size ndarray with 2D landmark coordinates in pixel space.
        Or returns None if no body could be detected
    """

    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose_detector.process(image)

    # Get the pose keypoint of the whole body
    if not results.pose_landmarks:
        return None

    # Selection of the necessary part from landmarks
    nose = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height)
    rshoulder = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)
    lshoulder = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width,
                          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height)

    return nose, rshoulder, lshoulder


def extract_face_bounds(frames_in: FrameProducer, head_focus: bool = False) -> Tuple[int, int, int, int]:
    """
        Get the global face boundingbox throughout the video

        :param frames_in: generator of input frames
        :param head_focus: if true, use the body detection phase to find the shoulder/head zone

        :returns The bounding box containing the face throughout the whole video
         as a 4-tuple of int elements, in order: x, y, width, height
    """

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
        mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose_detector:

        xs = []
        ys = []

        for image in frames_in.frames():

            # MediaPipe wants images in OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape
            x, y = 0, 0
            if head_focus:

                # Try to find information about the shoulder/nose area
                head_info = _get_head_region_info(image=image, pose_detector=pose_detector)

                if head_info is not None:
                    nose, _, lshoulder = head_info
                    # print("nose,rshoulder",nose,rshoulder)
                    # lshoulder is the bottom-right point
                    x2, y2 = lshoulder
                    # Compute the top-left
                    x, y = reflect(c=nose, P=lshoulder)
                    # Ensure that the reflected point is in the image boundaries
                    x = clamp(x, 0, w)
                    y = clamp(y, 0, h)

                    # Compute bbox and round to integers
                    w = int(x2 - x)
                    h = int(y2 - y)
                    x = int(x)
                    y = int(y)

                    # crop the bbox, if a body was visible from the front
                    if w > 0 and h > 0:
                        image = image[y:y+h, x:x+w]
                        # the `face_detection.process()` requires a contiguous array
                        image = np.ascontiguousarray(image)

            # Process the image with MediaPipe Face Detection.
            results = face_detection.process(image)

            if not results.detections:  # if nothing detected
                continue

            bbox_face = results.detections[0].location_data.relative_bounding_box  # we assume that there is always 1 face at least
            xm, ym, wm, hm = bbox_face.xmin*w, bbox_face.ymin*h, bbox_face.width*w, bbox_face.height*h  # map back to the original pixel space
            xs += [int(x+xm), int(x+xm+wm)]
            ys += [int(y+ym), int(y+ym+hm)]

    assert len(xs) == len(ys)

    if len(xs) == 0:
        raise Exception("No faces found through the whole video")

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)
    wx = max_x - min_x
    hy = max_y - min_y

    return min_x, min_y, wx, hy


if __name__ == '__main__':
    import argparse
    from .common import bbox_to_dict

    parser = argparse.ArgumentParser(description='Get the bounding box of the face throughout a video')
    parser.add_argument('--inframes', '--invideo',
                        help='Path to a video or directory showing a sign language interpreter.'
                             ' Hence, we assume that there is a face always present and visible.',
                        required=True)
    parser.add_argument('--outbounds',
                        help='Path for a JSON output file: a JSON structure containing the pixel-coordinates'
                             ' of the smallest rectangle containing the face of the person throughout the whole video.'
                             ' The rectangle must hold the same proportions of the original video (e.g.: 4:3, 16:9).'
                             ' Output has the format: { "x": int, "y": int, "width": int, "height": int}.',
                        required=True)
    parser.add_argument('--head-focus',
                        action='store_true',
                        help='Before trying to recognize the face, try to recognize the head zone of a full body.'
                             ' Useful when the face is too small but the body is visible.'
                             ' However, body recognition is much slower.',
                        required=False)

    args = parser.parse_args()

    # Extract the bounds and save them
    with create_frame_producer(args.inframes) as frames_in:
        bounds = extract_face_bounds(frames_in=frames_in, head_focus=args.head_focus)
        bounds_dict = bbox_to_dict(bounds)
        with open(args.outbounds, "w", encoding="utf-8") as outfile:
            json.dump(obj=bounds_dict, fp=outfile)
