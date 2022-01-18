#
# Classes to produce and consume videoframes from videos or directories into videos or directories

from abc import ABC
from abc import abstractmethod

import PIL
from PIL.Image import Image
import ffmpeg
import numpy as np

from typing import List

import os

# Set of image extensions supported while reading frames from image files
IMAGE_FORMATS = {'png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG'}


class FrameProducer(ABC):

    @abstractmethod
    def frames(self) -> np.ndarray:
        """Generator. Returns, one-by-one, all of the available frames."""
        pass

    def __enter__(self):
        """Allow the object to be used in a context with a 'with' statement"""
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_tb):
        """Clean up resources when exiting a context"""
        pass


class ImageFileProducer(FrameProducer):
    """A concrete frame producer getting image files from a directory. The files are processed in alphabetical order.
    The scanning is NOT recursive. Alpha channel is dropped. Only recognized image formats are processed.
    If an unrecognized image format is found, an exception is raised."""

    def __init__(self, directory_path: str):

        if not os.path.exists(directory_path):
            raise Exception("Path {} doesn't exist")

        if not os.path.isdir(directory_path):
            raise Exception("Path {} is not a directory")

        self.dir_path: str = directory_path

        # Take the list of all files in the directory and sort them alphabetically
        self.dir_filenames: List[str] = sorted(os.listdir(self.dir_path))

    def frames(self) -> np.ndarray:

        for file_name in self.dir_filenames:

            if file_name.startswith("."):
                continue

            extension = file_name.split('.')[-1]
            if extension not in IMAGE_FORMATS:
                raise Exception("Extension {} not supported".format(extension))

            file_path = os.path.join(self.dir_path, file_name)

            if os.path.isdir(file_path):
                continue

            #
            # Load the image and put into numpy array format
            img: Image = PIL.Image.open(file_path)
            img_np: np.ndarray = np.asarray(img)

            # Check for the presence of the alpha channel
            # In case, remove it.
            depth = img_np.shape[2]
            if depth == 4:
                # Drop the alpha channel
                img_np = img_np[:, :, :3]
            elif depth == 3:
                pass
            else:
                raise Exception("Unsupported image depth {}".format(depth))

            yield img_np

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Nothing to be closed when processing a directory
        pass


class VideoFrameProducer(FrameProducer):
    """A concrete frame producer getting frames from a video file"""

    def __init__(self, videofile_path: str):

        from .common import video_info

        if not os.path.exists(videofile_path):
            raise Exception("File {} doesn't exist".format(videofile_path))

        if not os.path.isfile(videofile_path):
            raise Exception("Path {} is not a file".format(videofile_path))

        self._video_w, self._video_h, _ = video_info(videofile_path)

        self._ffmpeg_read_process = (
            ffmpeg
            .input(videofile_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )

    def frames(self) -> np.ndarray:
        while True:
            in_bytes = self._ffmpeg_read_process.stdout.read(self._video_w * self._video_h * 3)
            if not in_bytes:
                break

            in_frame = (
                np
                .frombuffer(in_bytes, np.uint8)
                .reshape([self._video_w, self._video_h, 3])
                       )

            yield in_frame

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._ffmpeg_read_process.wait()
