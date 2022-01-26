#!/bin/bash

commands=("extract_face_bounds" "crop_video" "extract_face_data" "trim_video")

## now loop through the above array
for cmd in "${commands[@]}"
do
   echo "Updating help text for command $cmd"
   python -m slvideotools.$cmd --help > Docs/Help/$cmd.txt
done
