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

- Switch to MediaPipe 0.10.x (from 0.8.9: API changes)
- Update MTCNN to 1.0.0


## [Unreleased]

### Added

...

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
