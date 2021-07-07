import cv2
import mediapipe as mp
from .common import *
import ffmpeg


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


def get_bbox_face(input_video_path, output_video_path=None, skip_focus=False):
    """
        Get the global face boundingbox throughout the video

        Args:
          input_video_path: path to the input video
          [optional]
          output_video_path: path to the output video
          skip_focus: skip the body detection phase if True

        Returns:
          json(bbox): the global face bbox into json { "x": int, "y": int, "width": int, "height": int }
    """

    cap = cv2.VideoCapture(input_video_path)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        bboxes = []  # Will accumulate the bboxes across all frames
        # Frame-by-frame
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # print("End of video")
                break
            # img = image.copy()
            h, w, _ = image.shape

            if not skip_focus:
                nose, rshoulder, _ = get_roi(image)
                pts = get_bbox_pts(nose, rshoulder)
                # print(pts)
                x, y, w, h = cv2.boundingRect(pts.astype(int))
                image = crop_bbox(image, x, y, w, h)

            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.detections:  # if nothing detected
                continue
            # we assume that there is always 1 face at least
            bbox_face = results.detections[0].location_data.relative_bounding_box
            # map back to the original
            xm, ym, wm, hm = bbox_face.xmin*w, bbox_face.ymin*h, bbox_face.width*w, bbox_face.height*h
            bboxes.append([int(x+xm), int(y+ym), int(wm), int(hm)])

    if len(bboxes) == 0:
        return format_json_bbox(np.zeros(4))

    bboxes = np.array(bboxes)
    maxs = bboxes.max(axis=0)  # for coord x,y
    mins = bboxes.min(axis=0)  # for width and height

    if output_video_path:  # if output is defined
        stream = ffmpeg.input(input_video_path)
        stream = ffmpeg.drawbox(stream, mins[0], mins[1], maxs[2], maxs[3], color='red', thickness=2)
        stream = ffmpeg.output(stream, output_video_path)
        ffmpeg.run(stream)

    return format_json_bbox(np.array([mins[0], mins[1], maxs[2], maxs[3]]))


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
                             ' Output has the format: { "x": int, "y": int, "width": int, "height": int}',
                        required=True)
    parser.add_argument('--outvideo',
                        default=None,
                        help='Path for an (optional) videofile showing the original video'
                             'and an overlay of the region selected as bounds',
                        required=False)
    parser.add_argument('--skip-focus',
                        action='store_true',
                        help='Skip the body localisation phase.'
                             ' Useful when the face is already big enough and no body is really visible.',
                        required=False)

    args = parser.parse_args()
    print(get_bbox_face(args.invideo, args.outvideo, args.skip_focus),
          file=open(args.outbounds, "w", encoding="utf-8"))
