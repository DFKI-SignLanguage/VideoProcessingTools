import cv2
import mediapipe as mp
from .common import *
import ffmpeg

from typing import Tuple

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def get_roi(image):
    """
        Find the ROI containing the face from an input image

        Args:
          image : input image

        Returns: List [nose,rshoulder,lshoulder]
          nose: nose coordinates
          rshoulder: right shoulder coordinates
          lshoulder: left shoulder coordinates

    """
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5) as pose:
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Get the pose keypoint of the whole body
        if not results.pose_landmarks:
            raise RuntimeError("no body roi detected")

        # Selectiong the necessary part from landmarks
        nose = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height])
        rshoulder = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width,
                              results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height])
        lshoulder = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width,
                              results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height])

        return [nose, rshoulder, lshoulder]


def extract_face_bounds(input_video_path: str, output_video_path: str = None, head_focus: bool = False)\
        -> Tuple[int, int, int, int]:
    """
        Get the global face boundingbox throughout the video

        Args:
          input_video_path: path to the input video
          [optional]
          output_video_path: path to the output video
          head_focus: if true, use the body detection phase to find the shoulder/head zone

        Returns:
            a 4-tuple of int elements, in order: x, y, width, height
    """

    cap = cv2.VideoCapture(input_video_path)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        xs = []
        ys = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # print("End of video")
                break

            h, w, _ = image.shape
            x, y = 0, 0
            if head_focus:
                nose, rshoulder, _ = get_roi(image)
                # print("nose,rshoulder",nose,rshoulder)
                pts = get_bbox_pts(nose, rshoulder)
                # print(pts)
                x, y, w, h = cv2.boundingRect(pts.astype(int))
                # crop the bbox
                image = image[y:y+h, x:x+w]

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.detections:  # if nothing detected
                continue

            bbox_face = results.detections[0].location_data.relative_bounding_box  # we assume that there is always 1 face at least
            xm, ym, wm, hm = bbox_face.xmin*w, bbox_face.ymin*h, bbox_face.width*w, bbox_face.height*h  # map back to the original
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

    if output_video_path:  # if output is defined
        stream = ffmpeg.input(input_video_path)
        stream = ffmpeg.drawbox(stream, min_x, min_y, wx, hy, color='red', thickness=2)
        stream = ffmpeg.output(stream, output_video_path)
        ffmpeg.run(stream)

    return min_x, min_y, wx, hy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get the bounding box of the face throughout a video')
    parser.add_argument('--invideo',
                        help='Path to a video file showing a sign language interpreter.'
                             ' Hence, we assume that there is a face always present and visible.',
                        required=True)
    parser.add_argument('--outbounds',
                        help='Path for a JSON output file: a JSON structure containing the pixel-coordinates'
                             ' of the smallest rectangle containing the face of the person throughout the whole video.'
                             ' The rectangle must hold the same proportions of the original video (e.g.: 4:3, 16:9).'
                             ' Output has the format: { "x": int, "y": int, "width": int, "height": int}.',
                        required=True)
    parser.add_argument('--outvideo',
                        default=None,
                        help='Path for an (optional) videofile showing the original video'
                             'and an overlay of the region selected as bounds.',
                        required=False)
    parser.add_argument('--head-focus',
                        action='store_true',
                        help='Before trying to recognize the face, try to recognize the head zone of a full body.'
                             ' Useful when the face is too small but the body is visible.'
                             ' However, body recognition is much slower.',
                        required=False)

    args = parser.parse_args()

    # Extract the bounds and save them
    bounds = extract_face_bounds(args.invideo, args.outvideo, args.head_focus)
    bounds_json = format_json_bbox(bounds)
    with open(args.outbounds, "w", encoding="utf-8") as outfile:
        print(bounds_json, file=outfile)
