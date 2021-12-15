#
# Tests to check that the command line interface commands are at least present and loading all the libraries
import os


def test_cli_extract_face_bounds():
    exit_status = os.system('python -m dfki_sl_videotools.extract_face_bounds --help')
    assert exit_status == 0


def test_cli_crop_video():
    exit_status = os.system('python -m dfki_sl_videotools.crop_video --help')
    assert exit_status == 0


def test_cli_extract_face_data():
    exit_status = os.system('python -m dfki_sl_videotools.extract_face_data --help')
    assert exit_status == 0


def test_cli_trim_video():
    exit_status = os.system('python -m dfki_sl_videotools.trim_video --help')
    assert exit_status == 0

