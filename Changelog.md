# Changelog

Template and more info at <https://keepachangelog.com/en/1.0.0/>. 

Types of changes:

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities.


## [Planned]


## [Unreleased]

### Changed

## [0.6] - 2026-11-11

- Updated dependencies to the latest MediaPipe (0.10.30, then 1.2.2), TensorFlow 2.21.0, NumPy 2.x and OpenCV 5.0.
- Ported face bounds extraction (`extract_face_bounds`) from the legacy `mp.solutions` API (FaceDetection, Pose) to the MediaPipe Tasks API (`FaceDetector`, `PoseLandmarker`).
  - Two more models are needed in the `models/` directory: `blaze_face_short_range.tflite` and `pose_landmarker_heavy.task`. See `models/README.txt` for download links.
- Updated the installation instructions in the README.
- Added a `MediaPipeDocs/` folder with MediaPipe model cards (BlazeFace, Face Mesh V2, Blendshape V2) and the full-size face landmarks map, for quick reference.
- CI: tests now run on Ubuntu 24.04 with Python 3.11 and download all the required MediaPipe models.

## [0.5] - 2026-06-24

- Added the explicit list of the 52 blendshape names (`MEDIAPIPE_FACE_BLENDSHAPES`) to the `extract_face_data` module.
- Fixed: the face landmarker model is now loaded from the package `models/` directory instead of a path relative to the current working directory.
- Fixed many type-check issues (annotations in `extract_face_data` and in the `frames()` generator).
- CI: the GitHub test workflow now downloads the face landmarker model.

## [0.4] - 2025-08-20

- Switched from old mediapipe 0.8 FaceMesh solution to new 0.10 FaceLandmarker.
  - Number of landmasks increased from 468 to 478!Thus, the output shape of the landmark infomation numpy file is changed to [N][478][3].
- Removed cv2 dependency from face extraction.
  - Now OpenCV is needed only for the computation of the motion energy.
- Added code to extract blendshape data from faces. 52 blendshapes are optionally saved into a numyp array of shape [N][52]. See option --outblendshapes in `extract_face_data` module.

## [0.3] - 2025-08-14

- Switch to MediaPipe 0.10.x (from 0.8.9: API changes)
- Update MTCNN to 1.0.0
- Compatible with Python 3.11

## [0.2] - 2025-08-12

- Renamed package from `dfki_sl_videotools` to `slvideotools`
- Added more methods for face detection: in script `extract_face_bounds`, parameter `--method` can be either `mediapipe` or `mtcnn`
- Added a script to extract video motion energy using Optical Flow.
- Removed option --outvideo from the face detection script. Bounds can be overlayed in dedicated script `draw_bbox`
- Framework to "produce" and "consume" frames from either videos or directories of images


## [0.1] - 2022-01-26

- First release with command lines for face movement detection, crop, trim, facemesh extraction
- Added this Changelog :-)
- Face extraction tool can save save landmarks and includes a "normalization" option to transform the landmark to a camera-facing neutral position.
