import cv2
import mediapipe as mp
from mtcnn import MTCNN
import numpy as np

from .common import reflect
from .common import clamp

from .datagen import FrameProducer
from .datagen import create_frame_producer

import json
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

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


def _get_face_bounds_mediapipe(image: np.ndarray,
                               face_detector: mp_face_detection.FaceDetection,
                               head_focus: bool = False,
                               pose_detector: mp_pose.Pose = None) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the bounds of the face using the mediapipe framework

    :param image: RGB image pixels
    :param face_detection: The MediaPipe face detection solution
    :param head_focus: If True, first we try to get the body location (shoulders, nose), and search there
    :param pose_detector: If head_focus if used, this must be an initialized MediaPipe Pose instance
    :return: A 4-tuple with [x, y, width, height] of the bounds containing the face, or None if no face was found.
    """

    # Setup original full-frame bounds
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
                image = image[y:y + h, x:x + w]
                # the `face_detection.process()` requires a contiguous array
                image = np.ascontiguousarray(image)

    # Process the image with MediaPipe Face Detection.
    results = face_detector.process(image)

    if not results.detections:  # if no face can be detected -> None
        return None

    # we assume that there is always at least 1 face
    bbox_face = results.detections[0].location_data.relative_bounding_box
    # Map [0,1] bbox_face range to pixel space
    xm, ym, wm, hm = bbox_face.xmin * w, bbox_face.ymin * h, bbox_face.width * w, bbox_face.height * h
    # shift back to the original pixel space
    x += int(xm)
    y += int(ym)

    return x, y, int(wm), int(hm)


def _get_face_bounds_mtcnn(image: np.ndarray, detector: MTCNN) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the bounds of the face using the mediapipe framework

    :param image: RGB image pixels
    :return: A 4-tuple with [x, y, width, height] of the bounds containing the face, or None if no face was found.
    """

    face_list = detector.detect_faces(image, threshold_onet=0.85)
    if len(face_list) == 0:
        return None

    # Take the first in the list
    face = face_list[0]
    # bbox format is already [x, y, width, height]
    bbox = face['box']
    if bbox is None:
        return None

    x, y, w, h = bbox

    return x, y, w, h


#
#
#
def extract_face_bounds(frames_in: FrameProducer,
                        head_focus: bool = False,
                        method: str = "mediapipe") -> Tuple[int, int, int, int]:
    """
        Get the global face boundingbox throughout the video

        :param frames_in: generator of input frames
        :param head_focus: if true, use the body detection phase to find the shoulder/head zone
        :param method: Specify the head lookup framework: "mediapipe", "mtcnn".

        :returns The bounding box containing the face throughout the whole video
         as a 4-tuple of int elements, in order: x, y, width, height
    """

    # Init MediaPipe classes
    mp_face_detector = None
    mp_pose_detector = None
    if method == "mediapipe":
        mp_face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        if head_focus:
            mp_pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    # Init MTCNN class
    mtcnn_detector = None
    if method == "mtcnn":
        # See https://github.com/ipazc/mtcnn
        mtcnn_detector = MTCNN()

    try:

        # Accumulators for min and max coordinates of the face bbox.
        # For each frame, we add two new elements with x_min, x_max  coordinates for the detected face
        xs: List[int] = []
        # Same for y coordinates
        ys: List[int] = []

        for image in frames_in.frames():

            if method == "mediapipe":
                assert mp_face_detector is not None
                assert (head_focus and mp_pose_detector is not None) or ((not head_focus) and mp_pose_detector is None)

                bbox = _get_face_bounds_mediapipe(image=image,
                                                  face_detector=mp_face_detector,
                                                  head_focus=head_focus,
                                                  pose_detector=mp_pose_detector)

            elif method == "mtcnn":
                assert mtcnn_detector is not None
                bbox = _get_face_bounds_mtcnn(image=image, detector=mtcnn_detector)

            else:
                raise Exception("For face detection, method '{}' is not supported.".format(method))

            if bbox is None:
                continue

            x, y, w, h = bbox

            xs += [x, x+w]
            ys += [y, y+h]

    finally:
        if mp_face_detector is not None:
            mp_face_detector.close()
            mp_face_detector = None
        if mp_pose_detector is not None:
            mp_pose_detector.close()
            mp_pose_detector = None
        if mtcnn_detector is not None:
            mtcnn_detector = None

    assert len(xs) == len(ys)

    if len(xs) == 0:
        raise Exception("No faces found throughout the whole video")

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
    parser.add_argument('--method',
                        type=str,
                        help='Select the framework used to extract the face boundaries.'
                             ' Possible values: "mediapipe", "mtcnn"',
                        default="mediapipe")
    parser.add_argument('--head-focus',
                        action='store_true',
                        help='Used only with the "mediapipe" method.'
                             ' Before trying to recognize the face, try to recognize the head zone of a full body.'
                             ' Useful when the face is too small but the body is visible.'
                             ' However, body recognition is much slower.',
                        required=False)

    args = parser.parse_args()

    # Extract the bounds and save them
    with create_frame_producer(args.inframes) as frames_in:
        bounds = extract_face_bounds(frames_in=frames_in, head_focus=args.head_focus, method=args.method)
        bounds_dict = bbox_to_dict(bounds)
        with open(args.outbounds, "w", encoding="utf-8") as outfile:
            json.dump(obj=bounds_dict, fp=outfile)
