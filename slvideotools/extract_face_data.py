# Extracts the face mesh data from the frames of a video using MediaPipe.
# See: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

import math

import numpy as np

# Code to overlay the face mesh point taken from https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
VisionRunningMode = mp.tasks.vision.RunningMode

from PIL.Image import Image
import PIL.Image
from PIL import ImageDraw

from .datagen import create_frame_producer, create_frame_consumer
from .datagen import VideoFrameProducer, VideoFrameConsumer

from typing import List
from typing import Tuple


MEDIAPIPE_FACE_LANDMARKS_COUNT = 478 

MEDIAPIPE_FACE_BLENDSHAPES_COUNT = 52

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

VERTICES_TO_HIGHLIGHT = {
    VERTEX_ID_FRONT_TOP_RIGHT, VERTEX_ID_FRONT_TOP_LEFT,
    VERTEX_ID_EAR_TOP_R, VERTEX_ID_EAR_TOP_L,
    VERTEX_ID_JAW_BASE_R, VERTEX_ID_JAW_BASE_L,
    VERTEX_ID_NOSE_TIP
}


def vec_len(a: np.ndarray) -> float:
    """Computes the length of a vector."""

    a = np.power(a, 2)
    a = a.sum()
    a = np.sqrt(a)
    # returns the result as a scalar
    return a.item()


def normalize_face_landmarks(landmarks: List[List[float]], frame_width_px: int, frame_height_px: int,
                             nose_translation: np.ndarray, rot_mat: np.ndarray, scale: float) -> List[List[float]]:
    """Performs a normalization of the orientation of the Mediapipe face landmarks using forehead and lateral
     keypoints to build a rigid reference system
     """

    # This is the proportion between the H and W of the input frame.
    # This is needed because the frame coordinates are normalized in range [0,1] for both width and height.
    # It means that plain x,y landmark coordinates are not using the same scale
    # We will scale along the y axis in order to go back to an orthogonal system
    # Scale along the y axis to go to an orthogonal system
    y_ratio = 1.0 * frame_height_px / frame_width_px

    # Finally, rotate all vertices to the new system
    # TODO - If we store the landmarks list as numpy array, we can vectorize also the following transformation
    out = []
    for i, lm in enumerate(landmarks):
        lm_column_vec = np.asarray(a=lm, dtype=np.float32)

        # translate the nose tip to 0,0,0
        lm_centered = lm_column_vec - nose_translation
        # Scale along y to make the system orthogonal
        lm_centered[1] *= y_ratio
        # Apply the rotation
        lm_rotated = np.matmul(rot_mat.T, lm_centered)
        # re-scale to the original y proportions
        lm_rotated[1] /= y_ratio
        # Scale to normalize the head size
        lm_rotated /= scale
        # Re-center with nose tip to 0.5, 0.5, 0.5
        lm_recentered = lm_rotated + 0.5  # Adds the offsets to recenter the nose tip to the camera center

        out.append(lm_recentered.tolist())

    assert len(landmarks) == len(out)

    return out


def compute_normalization_params(landmarks: List[List[float]],
                                 frame_width_px: int,
                                 frame_height_px: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the normalization parameters.

    :param landmarks: List of 3D landmarks
    :param frame_width_px: video frame width
    :param frame_height_px: video frame height
    :return: The (3D position of the nose tip, 3x3 matrix transforming the face into a normalized pose,
     scaling factor to fix the size of the head.
    """

    #
    # Take the position of the tip of the nose to build the centering vector
    nose_tip = np.asarray(a=landmarks[VERTEX_ID_NOSE_TIP], dtype=np.float32)

    #
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
    R = np.asarray(a=[
        [new_x[0], new_y[0], new_z[0]],
        [new_x[1], new_y[1], new_z[1]],
        [new_x[2], new_y[2], new_z[2]]
    ], dtype=np.float32)  # type: np.ndarray
    assert R.shape == (3, 3)

    # # Debug variables:
    # should_be_I = R.T @ R
    # # cos of the angle between the new reference axes
    # should_be_zero = new_x.dot(new_y) / (len3D(new_x) * len3D(new_y))

    # Check through the determinant if this matrix is really orthogonal and invertible
    det = np.linalg.det(R)
    if np.abs(1 - det) > 0.01:
        raise Exception("Rotation matrix determinant deviates too much from 1 ({:06f})."
                        " Probably the computed landmarks are too distorted.".format(det))

    # A factor comparing how much the Y length increases with respect to 10% of the height, so that: |Y| / k = 0.1
    scale = vec_len(Y) / 0.1

    return nose_tip, R, scale


def extract_face_data(frames_in: VideoFrameProducer,
                      composite_frames_out: VideoFrameConsumer = None,
                      normalize_landmarks: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the MediaPipe face mesh data from the specified videofile.

    :param frames_in: A FrameProducer of some video material containing a face.
    :param composite_frames_out: If provided, the landmark information are overlayed to the original frames and saved here.
    :param normalize_landmarks: If set, the output landmarks are normalized.
    :return: a 4-tuple of numpy ndarrasys:
      1) ndarray with the face landmark data in shape [N][468][3];
      2) ndarray with the nose tip positions in shape [N][3];
      3) ndarray with the face rotation transforms in shape [N][3][3];
      4) ndarray with the face scaling in shape [N][1].
             Where N is the number of frames in the video.
    """

    # For video input:
    #mp_face_mesh = mp.solutions.face_mesh
    #face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5,
    #                                  static_image_mode=False, refine_landmarks=False)
    
    #
    # Initialize the Mediapipe Face Landmarker 
    base_options = mp_python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = mp_vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # Will store the H and W of the input video frame
    width = None
    height = None

    # Accumulators that wil contain the data for all frames.
    out_landmarks_list = []
    out_nose_tips = np.ndarray(shape=(0, 3), dtype=np.float32)
    out_Rs = np.ndarray(shape=(0, 3, 3), dtype=np.float32)
    out_scales = np.ndarray(shape=(0,), dtype=np.float32)

    frame_num = 0
    for rgb_image in frames_in.frames():

        # Store the size of the input frames.
        if width is None:
            width = rgb_image.shape[1]
            height = rgb_image.shape[0]

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = face_landmarker.detect(mp_image)

        # Check if at least one face is available
        if len(results.face_landmarks) == 0:
            landmarks = None
            # We just fill in the data with NaNs
            orig_frame_lm_list = [[float('nan')] * 3] * MEDIAPIPE_FACE_LANDMARKS_COUNT
            out_frame_lm_list = [[float('nan')] * 3] * MEDIAPIPE_FACE_LANDMARKS_COUNT
            nose_tip = np.asarray([float('nan')] * 3, dtype=np.float32)
            R = np.asarray([float('nan')] * 9, dtype=np.float32).reshape(3, 3)
            scale = float('nan')
        else:

            # Assume there is only one face
            landmarks = results.face_landmarks[0]

            assert len(landmarks) == MEDIAPIPE_FACE_LANDMARKS_COUNT


            # Map the list of landmarks into a bi-dimensional array (and convert it into a list)
            orig_frame_lm_list = list(map(lambda l: [l.x, l.y, l.z], landmarks))

            nose_tip, R, scale = compute_normalization_params(landmarks=orig_frame_lm_list, frame_width_px=width, frame_height_px=height)

            if normalize_landmarks:
                out_frame_lm_list = normalize_face_landmarks(landmarks=orig_frame_lm_list, frame_width_px=width, frame_height_px=height,
                                                   nose_translation=nose_tip, rot_mat=R, scale=scale)
            else:
                out_frame_lm_list = orig_frame_lm_list

        assert type(orig_frame_lm_list) == list
        assert type(out_frame_lm_list) == list
        assert len(orig_frame_lm_list) == MEDIAPIPE_FACE_LANDMARKS_COUNT
        assert len(out_frame_lm_list) == MEDIAPIPE_FACE_LANDMARKS_COUNT
        assert type(nose_tip) == np.ndarray
        assert nose_tip.shape == (3,)
        assert type(R) == np.ndarray
        assert R.shape == (3, 3)
        assert type(scale) == float

        # Store the transformation data
        out_nose_tips = np.append(out_nose_tips, [nose_tip], axis=0)
        out_Rs = np.append(out_Rs, [R], axis=0)
        out_scales = np.append(out_scales, [np.float32(scale)], axis=0)

        # Append to frames container
        out_landmarks_list.append(out_frame_lm_list)

        #
        # Manage composite video output
        if composite_frames_out is not None:

            # Prepare the overlay image
            #annotated_image = rgb_image.copy()
            pil_image: Image = PIL.Image.fromarray(obj=rgb_image)
            pil_draw = ImageDraw.Draw(pil_image)
            #draw.rectangle(xy=[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline=(220, 10, 10))


            # Draw face mesh landmarks on the overlay image.
            if landmarks is not None:

                # Let's use 1 pixel radius every 500 pixels of video.
                norm_landmark_radius = max(1, int(width / 600))

                # 
                # Draw the original landmarks over the face
                for i, lm in enumerate(orig_frame_lm_list):
                    lm_x, lm_y, lm_z = lm[:]

                    # If a coordinate is NaN, it's because the face was not found
                    if math.isnan(lm_x):
                        continue

                    # As the landmarks are already normalized in a range [0,1],
                    # bring them to the half of the output frame resolution

                    lm_x *= width
                    lm_y *= height

                    if i in VERTICES_TO_HIGHLIGHT:
                        vcol = (20, 220, 220)
                    else:
                        vcol = (20, 20, 220)

                    pil_draw.ellipse(xy=[lm_x - norm_landmark_radius, lm_y - norm_landmark_radius,  
                                        lm_x + norm_landmark_radius, lm_y + norm_landmark_radius],
                                     fill=vcol)

                #
                # DEBUG: save the landmarks to a file
                # import pickle
                # with open("landmarks-{:010d}.pck".format(frame_num), "wb") as outfile:
                #     pickle.dump(obj=lm_list, file=outfile)

                #
                # Draw the (normaized) landmarks in the upper left corner of the image using a orthographic projection (i.e., use only x and y)
                # and we use the depth to modulate the color intensity.

                # First compute the dynamic range of the z coordinate among all points
                zs = [p[2] for p in out_frame_lm_list]
                z_min = min(zs)
                z_max = max(zs)
                z_range = z_max - z_min

                # Draw the landmarks
                for i, lm in enumerate(out_frame_lm_list):
                    lm_x, lm_y, lm_z = lm[:]

                    # If a coordinate is NaN, it's because the face was not found
                    if math.isnan(lm_x):
                        continue

                    # As the landmarks are already normalized in a range [0,1],
                    # bring them to the half of the output frame resolution
                    lm_x *= width / 2
                    lm_y *= height / 2
                    # rescale z in [0,1]
                    norm_z = 1 - ((lm_z - z_min) / z_range)

                    pil_draw.ellipse(xy=[lm_x - norm_landmark_radius, lm_y - norm_landmark_radius,
                                        lm_x + norm_landmark_radius, lm_y + norm_landmark_radius],
                                     fill=(int(255 * norm_z), 20, 20))

            #
            # Finally, write the annotated frame to the output video
            annotated_image = np.asarray(pil_image)  # Back from PIL to numpy array
            composite_frames_out.consume(annotated_image)

        frame_num += 1

    out_landmarks = np.asarray(out_landmarks_list, dtype=np.float32)

    return out_landmarks, out_nose_tips, out_Rs, out_scales


#
# MAIN
#
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Uses mediapipe to extract the face mesh data from the frames of a video.")
    parser.add_argument('--inframes', '--invideo',
                        help='Path to a video or image directory providing the frames with the face of a person.',
                        required=True, type=str)
    parser.add_argument('--outlandmarks',
                        help='Path to the output numpy array of size [N][468][3],'
                             ' where N is the number of video frames,'
                             ' 468 are the number of landmarks of the [MediaPipe](https://mediapipe.dev) face mesh,'
                             ' and 3 is to store <x,y,z> 3D coords.'
                             ' If no faces are detected, all values are NaN!' 
                             ' If more faces are detected, only the first in the mediapipe list is used.',
                        required=True, type=str)
    parser.add_argument('--outnosetipposition',
                        help='Path to an output numpy array of shape [N][3] with the x,y,z movement of the nose tip in space.'
                             ' N is the number of video frames'
                             ' As for MediaPipe, X and Y coordinates are normalized in the range [0,1] in the frame size.',
                        required=False, type=str)
    parser.add_argument('--outfacerotation',
                        help='Path to an output numpy array of shape [N][3][3] with the 3x3 rotation of the face.'
                             ' N is the number of video frames',
                        required=False, type=str)
    parser.add_argument('--outfacescale',
                        help='Path to the output numpy array of shape [N] with the scaling of the face.'
                             ' N is the number of video frames.'
                             ' The scaling factor needed to resize the vertical distance within ear and jaw-base into 10 percent of the height of the frame.',
                        required=False, type=str)
    parser.add_argument('--outcompositeframes', '--outcompositevideo',
                        help='Path to a videofile or directory for image files. Will have the same resolution and content of the input frames,'
                             ' plus the overlay of the face landmarks.'
                             ' The blue landmarks are printed by mediapipe.'
                             ' The red landmarks, possibly normalized, and printed in the upper-left quadrant,'
                             ' are the outputted values',
                        required=False, type=str)
    parser.add_argument('--normalize-landmarks',
                        action='store_true',
                        help='If specified, neutralizes the head translation, rotation, and zoom.'
                             ' At each frame, a counter-rotation, -translation, and -scaling are applied in order to have:'
                             ' face nose facing the camera and head-up, nose tip at the center of the frame, head of the same size.',
                        required=False)

    #
    # Extract arguments
    args = parser.parse_args()

    source_path = args.inframes
    landmarkspath = args.outlandmarks
    nosetippositionpath = args.outnosetipposition
    facerotationpath = args.outfacerotation
    facescalepath = args.outfacescale
    compositeoutpath = args.outcompositeframes
    normalize_landmarks = args.normalize_landmarks

    #
    print("Extracting face landmarks from '{}' and save into '{}'...".format(source_path, landmarkspath))
    with create_frame_producer(source_path) as frames_prod:

        # Instantiate the frame consumer for the overlay video, if requested
        frames_cons = create_frame_consumer(compositeoutpath) if compositeoutpath is not None else None

        try:
            landmarksdata, nosetipdata, facerotdata, facescaledata =\
                extract_face_data(frames_in=frames_prod,
                                  composite_frames_out=frames_cons,
                                  normalize_landmarks=normalize_landmarks)
        except Exception as e:
            # Just forward the exception
            raise e
        finally:
            # Remember to close the frame consumer, if specified.
            if frames_cons is not None:
                frames_cons.close()

    # Save numpy arrays to a file
    for filepath, data in\
            [(landmarkspath, landmarksdata),
             (nosetippositionpath, nosetipdata),
             (facerotationpath, facerotdata),
             (facescalepath, facescaledata)]:

        if filepath is not None:
            print("Saving numpy array to '{}'".format(filepath))
            np.save(file=filepath, arr=data, allow_pickle=False)

    print("Done.")
