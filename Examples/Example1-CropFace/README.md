# Example - cropping faces

Small example based on a Makefile to convert all videos inside a directory.
For each video, the face is identified and the video cropped.

## Usage

Create first a `videos` link to a directory with MP4 (.mp4) video files

    cd Examples/Example1-CropFace
    ln -s path/to/my_videos_directory/ videos

Then launch the Make process

    make
   
The `make` will create a new directory with cropped videos

    ls -l cropped-faces
