# DFKI - Sign Language - Video Processing Tools

This is a repository of a set of command-line tools to preprocess videos for sign language analysis.

These scripts rely on a number of body/face analysis libraries (e.g., MediaPipe, OpenPose, ...) to analyse and extract information of the body parts incolved in sign language utterances. For example, identifying the location of hands/face, cropping at specified bounds, extracting landmarks, ...and the like.

The scripts are heavily based on the [kkroening ffmpeg python](https://kkroening.github.io/ffmpeg-python/) bindings.

## Installation

Clone the repository and setup a python environment for it (Tested with v3.7).

```sh
python3 -m venv p3env-videotools
source p3env-videotools/bin/activate
git clone https://github.com/DFKI-SignLanguage/VideoProcessingTools.git
cd VideoProcessingTools
pip install -r requirements.txt
```

## Scripts

Here is the list of scripts and their description.

In general, all the scripts are designed to be executed as modules, but their core functionality is available also as function.

### Extract Face Bounds

This scripts analyse a video in order to identify the rectangle containing the face of the person throughout the whole video.
It is useful on videos showing the full body of the interpreter because some software, like MediaPipe, do not work well when the face occupies only a small portion of the video.

```
python -m dfki_sl_videotools.extract_face_bounds --help                             
usage: extract_face_bounds.py [-h] --invideo INVIDEO --outbounds OUTBOUNDS
                              [--outvideo OUTVIDEO] [--skip-focus]

Get the bounding box of the face throughout a video

optional arguments:
  -h, --help            show this help message and exit
  --invideo INVIDEO     Path to a video file showing a sign language
                        interpreter. Hence, we assume that there is a face
                        always present and visible.
  --outbounds OUTBOUNDS
                        Path for a JSON output file: a JSON structure
                        containing the pixel-coordinates of the smallest
                        rectangle containing the face of the person throughout
                        the whole video. The rectangle must hold the same
                        proportions of the original video (e.g.: 4:3, 16:9).
                        Output has the format: { "x": int, "y": int, "width":
                        int, "height": int}
  --outvideo OUTVIDEO   Path for an (optional) videofile showing the original
                        videoand an overlay of the region selected as bounds
  --skip-focus          Skip the body localisation phase. Useful when the face
                        is already big enough and no body is really visible.
```

### Crop Video

```
python -m dfki_sl_videotools.crop_video --help
usage: crop_video.py [-h] --invideo INVIDEO --inbounds INBOUNDS --outvideo
                     OUTVIDEO

Crop a video at a specified rectangular area.

optional arguments:
  -h, --help           show this help message and exit
  --invideo INVIDEO    Path to the input videofile
  --inbounds INBOUNDS  Path to a JSON file containing the bounds information
                       for cropping. Format is: { "x": int, "y": int, "width":
                       int, "height": int}
  --outvideo OUTVIDEO  Path for the output videofile, showing the cropped area
```

_Warning!!!_ The resolution of the output video might differ from the width/height specified in the JSON file. This is due to limitations of some codecs.

### Extract Face Mesh

```
python -m dfki_sl_videotools.extract_face_mesh --help
usage: extract_face_mesh.py [-h] --invideo INVIDEO --outfaceanimation
                            OUTFACEANIMATION
                            [--outheadanimation OUTHEADANIMATION]
                            [--outcompositevideo OUTCOMPOSITEVIDEO]
                            [--no-head-movement NO_HEAD_MOVEMENT]

Uses mediapipe to extract the face mesh data from the frames of a video.

optional arguments:
  -h, --help            show this help message and exit
  --invideo INVIDEO     Path to a videofile containing the face of a person.
  --outfaceanimation OUTFACEANIMATION
                        Path to the output numpy array of size [N][468][3],
                        where N is the number of video frames, 468 are the
                        number of landmarks of the
                        [MediaPipe](https://mediapipe.dev) face mesh, and 3 is
                        to store <x,y,z> 3D coords.
  --outheadanimation OUTHEADANIMATION
                        Path to the output numpy array of size [N][6] with the
                        movement of the head in space. N is the number of
                        video frames and 6 (3+3) are the 3-tuple translation
                        and 3-tuple angles moving and rotating the face in
                        space. TODO: check, maybe the rotation can be a
                        quaternion.
  --outcompositevideo OUTCOMPOSITEVIDEO
                        Path to a (optional) videofile with the same
                        resolution and frames of the original video, plus the
                        overlay of the face landmarks
  --no-head-movement NO_HEAD_MOVEMENT
                        TODO -- If specified, neutralizes the head movement,
                        i.e., the face mandmarks are saved without translation
                        and rotation, as if the person's nose is always facing
                        the front, in the direction of the camera
```

### Trim Video

```
python -m dfki_sl_videotools.trim_video --help                                      
usage: trim_video.py [-h] --invideo INVIDEO --outvideo OUTVIDEO --startframe
                     STARTFRAME --endframe ENDFRAME

Trims a video file.

optional arguments:
  -h, --help            show this help message and exit
  --invideo INVIDEO     Input video filepath
  --outvideo OUTVIDEO   Output video filepath
  --startframe STARTFRAME
                        First frame to retain (counting from 1)
  --endframe ENDFRAME   Last frame to retain (counting from 1)
```

## Testing

Test modules/functions are implemented using [pytest](https://docs.pytest.org/).
After setting up the python environment, open a terminal and... 

    cd .../VideoProcessingTools
    pytest
