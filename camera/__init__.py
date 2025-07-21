"""
Camera interface module that provides a unified interface for different camera backends.
"""
from .base import CameraBase
from .opencv_camera import OpenCVCamera
from .realsense_camera import RealSenseCamera
from .factory import CameraFactory

__all__ = [
    'CameraBase', 
    'OpenCVCamera', 
    'RealSenseCamera',
    'CameraFactory'
]
