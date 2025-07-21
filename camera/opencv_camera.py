"""
基于OpenCV的相机实现。
"""
import cv2
import numpy as np
from typing import Optional, Tuple

from .base import CameraBase


class OpenCVCamera(CameraBase):
    """
    使用OpenCV的VideoCapture实现的相机类。
    """
    
    def __init__(self, camera_id: int = 0, **kwargs):
        """
        初始化OpenCV相机。
        
        参数:
            camera_id: 相机设备ID（默认：0）
            width: 期望的帧宽度（默认：1280）
            height: 期望的帧高度（默认：720）
            fps: 期望的每秒帧数（默认：30）
        """
        width = kwargs.get('width', 1280)
        height = kwargs.get('height', 720)
        fps = kwargs.get('fps', 30)
        
        self.camera_id = camera_id
        self._cap = None
        self._width = width
        self._height = height
        self._fps = fps
        self._is_opened = False
    
    def open(self) -> bool:
        """打开相机连接。"""
        if self._is_opened:
            return True
            
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            return False
            
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        
        self._is_opened = True
        return True
    
    def close(self) -> None:
        """关闭相机连接。"""
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._is_opened = False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        从相机读取一帧图像。
        
        返回:
            tuple: (success, frame)，其中success是布尔值表示是否成功读取帧，
                  frame是捕获的图像，以numpy数组形式返回。
        """
        if not self.is_opened and not self.open():
            return False, None
            
        ret, frame = self._cap.read()
        if not ret:
            self._is_opened = False
            return False, None
            
        return True, frame
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取当前相机分辨率。"""
        if not self.is_opened:
            return self._width, self._height
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置相机分辨率。
        
        参数:
            width: 期望的帧宽度
            height: 期望的帧高度
            
        返回:
            bool: 如果分辨率设置成功返回True，否则返回False
        """
        self._width = width
        self._height = height
        
        if self.is_opened:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 验证分辨率是否设置成功
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (actual_width, actual_height) == (width, height)
            
        return True
    
    def get_fps(self) -> int:
        """获取当前帧率(FPS)设置。"""
        if not self.is_opened:
            return self._fps
        return int(self._cap.get(cv2.CAP_PROP_FPS))
    
    def set_fps(self, fps: int) -> bool:
        """
        设置相机的帧率(FPS)。
        
        参数:
            fps: 期望的每秒帧数
            
        返回:
            bool: 如果FPS设置成功返回True，否则返回False
        """
        self._fps = fps
        if self.is_opened:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            return int(self._cap.get(cv2.CAP_PROP_FPS)) == fps
        return True
    
    @property
    def is_opened(self) -> bool:
        """检查相机是否已打开并准备就绪可以读取帧。"""
        return self._is_opened and self._cap is not None and self._cap.isOpened()
    
    def __enter__(self):
        """上下文管理器入口。"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出。"""
        self.close()
