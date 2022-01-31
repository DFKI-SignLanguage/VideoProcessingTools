# Chengelog

Template and more info at <https://keepachangelog.com/en/1.0.0/>. 

Types of changes:

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities.


## [Unreleased]

### Added

- Framework to "produce" and "consume" frames from either videos or directories of images
- Renamed package from `dfki_sl_videotools` to `slvideotools`
- Removed option --outvideo from the face detection script. Bounds can be overlayed in another dedicated script.

## [0.1] - 2022-01-26

- First release with command lines for face movement detection, crop, trim, facemesh extraction
- Added this Changelog :-)
- Face extraction tool can save save landmarks and includes a "normalization" option to transform the landmark to a camera-facing neutral position.
