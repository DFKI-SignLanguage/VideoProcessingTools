import cv2
import mediapipe as mp
import numpy as np

# TODO -- use the mp_drawing to compose overlay images
# mp_drawing = mp.solutions.drawing_utils


def extract_face_data(videofilename: str) -> np.ndarray:
    """
    Extract the MediaPipe face mesh data from the specified videofile.

    :param videofilename: Path to a video containing a face
    :return: a numpy array with the face data in shape ([N][468][3]), where N is the number of frames in the video.
    """

    # For video input:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(videofilename)

    if not cap.isOpened():
        raise Exception("Couldn't open video file '{}'".format(videofilename))

    frames = []
    frame_num = 0
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            # The video is finished
            break

        # Debug: Save the frame
        # cv2.imwrite("testframe.png", image)

        # Flip the image horizontally for a later selfie-view display
        # image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        if results is None:
            raise Exception("Face couldn't be recognized in frame {}".format(frame_num))

        # Assume there is only one face
        landmarks = results.multi_face_landmarks[0]

        # Map the list of landmarks into a bi-dimensional array (and convert it into a list)
        lm_list = list(map(lambda l: [l.x, l.y, l.z], landmarks.landmark))
        assert len(lm_list) == 468

        # Append to frames container
        frames.append(lm_list)
        frame_num += 1

    cap.release()

    out_array = np.asarray(frames)
    return out_array


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Uses mediapipe to extract the face mesh data from the frames of a video.")
    parser.add_argument('--invideo',
                        help='Path to a videofile containing the face of a person.',
                        required=True)
    parser.add_argument('--outfaceanimation',
                        help='Path to the output numpy array of size [N][468][3],'
                             ' where N is the number of video frames,'
                             ' 468 are the number of landmarks of the [MediaPipe](https://mediapipe.dev) face mesh,'
                             ' and 3 is to store <x,y,z> 3D coords.',
                        required=True)
    parser.add_argument('--outheadanimation',
                        help='Path to the output numpy array of size [N][6] with the movement of the head in space.'
                             ' N is the number of video frames and 6 (3+3) are the 3-tuple translation'
                             ' and 3-tuple angles moving and rotating the face in space.'
                             ' TODO: check, maybe the rotation can be a quaternion.',
                        required=False)
    parser.add_argument('--outcompositevideo',
                        help='Path to a (optional) videofile with the same resolution and frames of the original video,'
                             ' plus the overlay of the face landmarks',
                        required=False)
    parser.add_argument('--no-head-movement',
                        help='TODO -- If specified, neutralizes the head movement,'
                             ' i.e., the face landmarks are saved without translation and rotation,'
                             ' as if the person\'s nose is always facing the front, in the direction of the camera',
                        required=False)

    #
    # Extract arguments
    args = parser.parse_args()

    video_filename = args.invideo
    faceanimation_filename = args.outfaceanimation
    # TODO -- consider also the other parameters:
    # outheadanimation
    # outcompositevideo
    # no-head-movement

    #
    print("Extracting face landmarks from '{}' and save into '{}'...".format(video_filename, faceanimation_filename))
    facedata = extract_face_data(videofilename=video_filename)
    # Save numpy array to a file
    facedata.dump(faceanimation_filename)

    print("Done.")
