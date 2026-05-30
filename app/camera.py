# camera.py
import cv2
import tempfile
import os


def open_laptop_camera(index: int = 0):
    """Open laptop webcam (default index 0)."""
    return cv2.VideoCapture(index)


def open_iriun_camera(index: int = 1):
    """
    Open Iriun virtual webcam.
    Change index if your system uses a different one.
    """
    return cv2.VideoCapture(index)


def open_video_file(path: str):
    """Open a video file."""
    return cv2.VideoCapture(path)


def save_uploaded_to_temp(uploaded_file) -> str:
    """
    Save a Streamlit uploaded_file to a temp path and return the path.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name
