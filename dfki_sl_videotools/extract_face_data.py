import cv2
import mediapipe as mp
import numpy as np

import ffmpeg

from typing import List

# Code to overlay the face mesh point taken from https://google.github.io/mediapipe/solutions/holistic.html
mp_drawing = mp.solutions.drawing_utils

# Vertices numbers derived from uv texture or from FBX model.
# Vertices on the front. Will be usd to compute the reference horizontal vector
VERTEX_ID_FRONT_TOP_RIGHT = 54
VERTEX_ID_FRONT_TOP_LEFT = 284
# Vertices at the side of the face, next to the ears. Will be usd to compute the reference vertical vector
VERTEX_ID_EAR_TOP_L = 356
VERTEX_ID_JAW_BASE_L = 361
VERTEX_ID_EAR_TOP_R = 127
VERTEX_ID_JAW_BASE_R = 132

VERTEX_ID_NOSE_BASE = 168
VERTEX_ID_NOSE_TIP = 4

VERTICES_TO_DRAW = {
    VERTEX_ID_FRONT_TOP_RIGHT, VERTEX_ID_FRONT_TOP_LEFT,
    VERTEX_ID_EAR_TOP_R, VERTEX_ID_EAR_TOP_L,
    VERTEX_ID_JAW_BASE_R, VERTEX_ID_JAW_BASE_L,
    VERTEX_ID_NOSE_TIP
}


def vec_len(a: np.ndarray) -> float:

    a = np.power(a, 2)  # type: np.ndarray
    a = a.sum()
    a = np.sqrt(a)
    # returns the result as a scalar
    return a.item()


def normalize_face_landmarks(landmarks: List[List[float]], frame_width_px: int, frame_height_px: int) -> List[List[float]]:
    """Performs a normalizatiopn of the orientation of the Mediapipe face landmarks using forehead and lateral
     keypoints to build a rigid reference system
     """

    # This is the proportion between the H and W of the input frame.
    y_ratio = 1.0 * frame_height_px / frame_width_px
    # This is needed because the frame coordinates are normalized in range [0,1] for both width and height.
    # It means that plain x,y landmark coordinates are not using the same scale
    # We will scale along the y axis in order to go back to an orthogonal system
    # Scale along the y axis to go to an orthogonal system
    landmarks = [[lm[0], lm[1] * y_ratio, lm[2]] for lm in landmarks]

    #
    # Build the new reference X axis
    top_front_right = np.asarray(a=landmarks[VERTEX_ID_FRONT_TOP_RIGHT], dtype=np.float32)
    top_front_left = np.asarray(a=landmarks[VERTEX_ID_FRONT_TOP_LEFT], dtype=np.float32)
    X = top_front_left - top_front_right  # type: np.ndarray

    #
    # Build the reference Y axis
    ear_top_left = np.asarray(a=landmarks[VERTEX_ID_EAR_TOP_L], dtype=np.float32)
    jaw_base_left = np.asarray(a=landmarks[VERTEX_ID_JAW_BASE_L], dtype=np.float32)
    ear_top_right = np.asarray(a=landmarks[VERTEX_ID_EAR_TOP_R], dtype=np.float32)
    jaw_base_right = np.asarray(a=landmarks[VERTEX_ID_JAW_BASE_R], dtype=np.float32)

    # A: vertical down vector from the midpoints between ears and jaw bases
    ears_midpoint = (ear_top_left + ear_top_right) / 2
    jaw_base_midpoint = (jaw_base_left + jaw_base_right) / 2
    A = jaw_base_midpoint - ears_midpoint

    # Now, project A on the new X axis in order to have exactly 90 degrees
    # First, the dot product can be used to retrieve the cos_theta (angle between the vectors)
    cos_theta = X.dot(A) / (vec_len(X) * vec_len(A))
    # Compute the size of the projection of A over X
    proj_A_over_X = vec_len(A) * cos_theta
    # Then compute the small vector a by multiplying the projection to a unit vector with the same direction of X
    a = proj_A_over_X * X / vec_len(X)
    # Now compute the tip of the vertical Y axis
    Y = A - a

    # Normalize everything
    new_x = X / vec_len(X)
    new_y = Y / vec_len(Y)
    # The new front axis is simply the cross product of x and y
    new_z = np.cross(new_x, new_y)

    # Compose the 3D transformation matrix able to align the "face coordinate system" back with the reference XYZ system
    mat = np.asarray(a=[
        [new_x[0], new_x[1], new_x[2]],
        [new_y[0], new_y[1], new_y[2]],
        [new_z[0], new_z[1], new_z[2]]
    ], dtype=np.float32)  # type: np.ndarray
    assert mat.shape == (3, 3)

    # # Debug variables:
    # should_be_I = mat.T @ mat
    # # cos of the angle between the new reference axes
    # should_be_zero = new_x.dot(new_y) / (len3D(new_x) * len3D(new_y))

    # Check through the determinant if this matrix is really orthogonal and invertible
    det = np.linalg.det(mat)
    if np.abs(1 - det) > 0.01:
        raise Exception("Rotation matrix determinant deviates too much from 1 ({:06f}). Probably the computed landmarks are too distorted.".format(det))

    # Take the position of the tip of the nose to build the centering vector
    nose_tip = np.asarray(a=landmarks[VERTEX_ID_NOSE_TIP], dtype=np.float32)

    # Finally, rotate all vertices to the new system
    # TODO - If we store the landmarks list as numpy array, we can vectorize also the following transformation
    out = []
    for i, lm in enumerate(landmarks):
        lm_column_vec = np.asarray(a=lm, dtype=np.float32)

        # translate the nose tip to 0,0,0
        lm_centered = lm_column_vec - nose_tip
        # Apply the rotation
        lm_rotated = np.matmul(mat, lm_centered)
        # re-scale to the original y proportions
        lm_rotated[1] /= y_ratio
        # Re-center with nose tip to 0.5, 0.5, 0.5
        lm_recentered = lm_rotated + 0.5  # Adds the offsets to recenter the nose tip to the camera center
        out.append(lm_recentered.tolist())

    assert len(landmarks) == len(out)

    return out


def extract_face_data(videofilename: str,
                      outcompositevideo: str = None,
                      normalize_landmarks: bool = False) -> np.ndarray:
    """
    Extract the MediaPipe face mesh data from the specified videofile.

    :param videofilename: Path to a video containing a face
    :return: a numpy array with the face data in shape ([N][468][3]), where N is the number of frames in the video.
    """

    # For video input:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5,
                                      static_image_mode=False, refine_landmarks=False)
    cap = cv2.VideoCapture(videofilename)

    if not cap.isOpened():
        raise Exception("Couldn't open video file '{}'".format(videofilename))

    composite_video_out_process = None

    width = None
    height = None

    frames = []
    frame_num = 0
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            # The video is finished
            break

        # Store the size of the input frames.
        if width is None:
            width = image.shape[1]
            height = image.shape[0]

        # Flip the image horizontally for a later selfie-view display
        # image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        rgb_image.flags.writeable = False
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks is None:
            continue
            # raise Exception("Face couldn't be recognized in frame {}".format(frame_num))

        # Assume there is only one face
        landmarks = results.multi_face_landmarks[0]

        # Map the list of landmarks into a bi-dimensional array (and convert it into a list)
        lm_list = list(map(lambda l: [l.x, l.y, l.z], landmarks.landmark))
        assert len(lm_list) == 468

        if normalize_landmarks:
            lm_list = normalize_face_landmarks(lm_list, width, height)

        # Append to frames container
        frames.append(lm_list)

        #
        # Manage composite video output
        if outcompositevideo is not None:

            if composite_video_out_process is None:

                composite_video_out_process = (
                    ffmpeg
                        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
                        .output(outcompositevideo, pix_fmt='yuv420p')
                        .overwrite_output()
                        .run_async(pipe_stdin=True)
                )

            # Print and draw face mesh landmarks on the image.
            annotated_image = image.copy()
            # print('face_landmarks:', face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=landmarks)

            #
            # DEBUG: save the landmarks to a file
            # import pickle
            # with open("landmarks-{:010d}.pck".format(frame_num), "wb") as outfile:
            #     pickle.dump(obj=lm_list, file=outfile)

            #
            # Print landmarks with custom routine
            # Fill the upper left quarter of the image using a orthographic projection (i.e., use only x and y)
            # and we use the depth to modulate the color intensity.

            # First compute the dynamic range of the z coordinate among all points
            zs = [p[2] for p in lm_list]
            z_min = min(zs)
            z_max = max(zs)
            z_range = z_max - z_min

            # Draw the landmarks
            for i, lm in enumerate(lm_list):
                lm_x, lm_y, lm_z = lm[:]
                # As the landmarks are already normalized in a range [0,1],
                # bring them to the half of the output frame resolution

                lm_x *= width / 2
                lm_y *= height / 2
                norm_z = 1 - ((lm_z - z_min) / z_range)

                cv2.circle(img=annotated_image, center=(int(lm_x), int(lm_y)), radius=3,
                           color=(int(254 * norm_z), 20, 20), thickness=2)

            # Finally, write the annotated frame to the output video
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            composite_video_out_process.stdin.write(
                annotated_image.astype(np.uint8).tobytes()
            )

        frame_num += 1

    cap.release()

    if composite_video_out_process is not None:
        composite_video_out_process.stdin.close()
        composite_video_out_process.wait()

    out_array = np.asarray(frames, dtype=np.float32)
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
                        help='TODO -- Path to the output numpy array of size [N][6] with the movement of the head in space.'
                             ' N is the number of video frames and 6 (3+3) are the 3-tuple translation'
                             ' and 3-tuple angles moving and rotating the face in space.'
                             ' TODO: check, maybe the rotation can be a quaternion.',
                        required=False)
    parser.add_argument('--outcompositevideo',
                        help='Path to a (optional) videofile with the same resolution and frames of the original video,'
                             ' plus the overlay of the face landmarks',
                        required=False)
    parser.add_argument('--no-head-movement',
                        help='If specified, neutralizes the head movement,'
                             ' i.e., at each frame a counter-rotation is applied in order to have the'
                             ' person\'s nose facing the front, in the direction of the camera, and the face vertical.',
                        required=False)

    #
    # Extract arguments
    args = parser.parse_args()

    video_filename = args.invideo
    faceanimation_filename = args.outfaceanimation
    outcompositevideo = args.outcompositevideo
    normalize_landmarks = args.no_head_movement
    # TODO -- consider also the other parameters:
    # outheadanimation

    #
    print("Extracting face landmarks from '{}' and save into '{}'...".format(video_filename, faceanimation_filename))
    facedata = extract_face_data(videofilename=video_filename,
                                 outcompositevideo=outcompositevideo,
                                 normalize_landmarks=normalize_landmarks)
    # Save numpy array to a file
    facedata.dump(faceanimation_filename)

    print("Done.")
