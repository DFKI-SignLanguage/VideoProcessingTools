# Chengelog

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
