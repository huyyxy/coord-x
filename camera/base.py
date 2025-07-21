"""
相机基础接口定义。
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class CameraBase(ABC):
    """
    相机接口的抽象基类。
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        使用可选参数初始化相机。
        
        参数:
            **kwargs: 相机特定参数
        """
        pass
    
    @abstractmethod
    def open(self) -> bool:
        """
        打开相机连接。
        
        返回:
            bool: 如果相机成功打开返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        关闭相机连接并释放资源。
        """
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        从相机读取一帧图像。
        
        返回:
            tuple: (success, frame)，其中success是布尔值表示是否成功读取帧，
                  frame是捕获的图像，以numpy数组形式返回。
        """
        pass

    @abstractmethod
    def read_frames(self) -> Tuple[bool, Optional[Dict[str, np.ndarray]]]:
        """
        从相机读取一帧。
        
        返回:
            tuple: (success, frames) 其中 success 是布尔值，frames 是包含 'color' 和/或 'depth' 帧的字典，
                  值为 numpy 数组
        """
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """
        获取当前相机分辨率。
        
        返回:
            tuple: 相机分辨率的 (宽度, 高度)
        """
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置相机分辨率。
        
        参数:
            width: 期望的帧宽度
            height: 期望的帧高度
            
        返回:
            bool: 如果分辨率设置成功返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def get_fps(self) -> int:
        """
        获取当前帧率(FPS)设置。
        
        返回:
            int: 当前FPS值
        """
        pass
    
    @abstractmethod
    def set_fps(self, fps: int) -> bool:
        """
        设置相机的帧率(FPS)。
        
        参数:
            fps: 期望的每秒帧数
            
        返回:
            bool: 如果FPS设置成功返回True，否则返回False
        """
        pass
    
    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """
        检查相机是否已打开并准备就绪可以读取帧。
        
        返回:
            bool: 如果相机已打开返回True，否则返回False
        """
        pass
    
    def get_properties(self) -> Dict[str, Any]:
        """
        获取相机属性和设置。
        
        返回:
            dict: 包含相机属性的字典
        """
        return {
            'resolution': self.get_resolution(),
            'fps': self.get_fps(),
            'is_opened': self.is_opened
        }
