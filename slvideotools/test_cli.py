# Tests to check that the command line interface commands are at least present and loading all the libraries
import os


def test_cli_extract_face_bounds():
    exit_status = os.system('python -m slvideotools.extract_face_bounds --help')
    assert exit_status == 0


def test_cli_draw_bbox():
    exit_status = os.system('python -m slvideotools.draw_bbox --help')
    assert exit_status == 0


def test_cli_crop_video():
    exit_status = os.system('python -m slvideotools.crop_video --help')
    assert exit_status == 0


def test_cli_extract_face_data():
    exit_status = os.system('python -m slvideotools.extract_face_data --help')
    assert exit_status == 0


def test_cli_trim_video():
    exit_status = os.system('python -m slvideotools.trim_video --help')
    assert exit_status == 0

