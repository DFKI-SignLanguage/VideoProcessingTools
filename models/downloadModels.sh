#!/usr/bin/env bash
#
# Downloads the external models listed in README.txt into this directory.
# Usage: bash models/downloadModels.sh  (can be invoked from any directory)
#

set -euo pipefail

# Directory containing this script (the models dir)
MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_URLS=(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

for url in "${MODEL_URLS[@]}"; do
    filename="$(basename "$url")"
    dest="$MODELS_DIR/$filename"
    echo "Downloading $filename ..."
    # Download to a temp file first, so an interrupted download doesn't leave a broken model.
    curl --fail --location --show-error --progress-bar --output "$dest.part" "$url"
    mv "$dest.part" "$dest"
done

echo "All models downloaded to $MODELS_DIR"
