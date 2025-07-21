"""
Intel RealSense 相机的实现。
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any

# RealSense 是一个可选的依赖项
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from .base import CameraBase


class RealSenseCamera(CameraBase):
    """
    Intel RealSense 相机的实现类。
    """
    
    def __init__(self, device_id: str = None, **kwargs):
        """
        初始化 RealSense 相机。
        
        参数:
            device_id: 设备序列号，如果为 None 则使用第一个可用的设备
            width: 期望的帧宽度（默认：1280）
            height: 期望的帧高度（默认：720）
            fps: 期望的帧率（默认：30）
            enable_depth: 是否启用深度流（默认：False）
            enable_color: 是否启用彩色流（默认：True）
        """
        if rs is None:
            raise ImportError("pyrealsense2 is not installed. Please install it with 'pip install pyrealsense2'")
        width = kwargs.get('width', 1280)
        height = kwargs.get('height', 720)
        fps = kwargs.get('fps', 30)
        enable_depth = kwargs.get('enable_depth', False)
        enable_color = kwargs.get('enable_color', True)
            
        self.device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._enable_depth = enable_depth
        self._enable_color = enable_color
        
        self._pipeline = None
        self._config = None
        self._is_opened = False
        self._depth_scale = 1.0
        self._intrinsics = None
    
    def open(self) -> bool:
        """打开相机连接。"""
        if self._is_opened:
            return True
            
        try:
            self._pipeline = rs.pipeline()
            self._config = rs.config()
            
            # 根据配置启用流
            if self.device_id is not None:
                self._config.enable_device(self.device_id)
                
            if self._enable_color:
                self._config.enable_stream(
                    rs.stream.color, 
                    self._width, 
                    self._height, 
                    rs.format.bgr8, 
                    self._fps
                )
                
            if self._enable_depth:
                self._config.enable_stream(
                    rs.stream.depth, 
                    self._width, 
                    self._height, 
                    rs.format.z16, 
                    self._fps
                )
            
            # 开始流传输
            profile = self._pipeline.start(self._config)
            
            # 获取深度帧处理的深度比例
            if self._enable_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                self._depth_scale = depth_sensor.get_depth_scale()

            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._intrinsics = color_profile.get_intrinsics()
            
            self._is_opened = True
            return True
            
        except Exception as e:
            print(f"Failed to open RealSense camera: {e}")
            self.close()
            return False
    
    def close(self) -> None:
        """关闭相机连接。"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except:
                pass
            self._pipeline = None
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
            
        try:
            frames = self._pipeline.wait_for_frames()
            result = None
            
            if self._enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    result = np.asanyarray(color_frame.get_data())
                else:
                    return False, None
            
            return True, result
            
        except Exception as e:
            print(f"Error reading from RealSense camera: {e}")
            self._is_opened = False
            return False, None

    def read_frames(self) -> Tuple[bool, Optional[Dict[str, np.ndarray]]]:
        """
        从相机读取一帧。
        
        返回:
            tuple: (success, frames) 其中 success 是布尔值，frames 是包含 'color' 和/或 'depth' 帧的字典，
                  值为 numpy 数组
        """
        if not self.is_opened and not self.open():
            return False, None
            
        try:
            frames = self._pipeline.wait_for_frames()
            result = {}
            
            if self._enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    result['color'] = np.asanyarray(color_frame.get_data())
                else:
                    return False, None
            
            if self._enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    result['depth'] = np.asanyarray(depth_frame.get_data())
                else:
                    return False, None
            
            return True, result
            
        except Exception as e:
            print(f"Error reading from RealSense camera: {e}")
            self._is_opened = False
            return False, None
    
    def get_resolution(self) -> Tuple[int, int]:
        """获取当前相机分辨率。"""
        return self._width, self._height
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置相机分辨率。
        
        注意：对于 RealSense，更改分辨率需要重新初始化管道。
        """
        if width == self._width and height == self._height:
            return True
            
        was_opened = self.is_opened
        if was_opened:
            self.close()
            
        self._width = width
        self._height = height
        
        if was_opened:
            return self.open()
            
        return True
    
    def get_fps(self) -> int:
        """获取当前帧率(FPS)设置。"""
        return self._fps
    
    def set_fps(self, fps: int) -> bool:
        """
        设置相机的帧率(FPS)。
        
        注意：对于 RealSense，更改帧率需要重新初始化管道。
        """
        if fps == self._fps:
            return True
            
        was_opened = self.is_opened
        if was_opened:
            self.close()
            
        self._fps = fps
        
        if was_opened:
            return self.open()
            
        return True
    
    @property
    def is_opened(self) -> bool:
        """检查相机是否已打开并准备好读取帧。"""
        return self._is_opened and self._pipeline is not None
    
    def get_depth_scale(self) -> float:
        """
        获取深度比例因子。
        
        返回：
            float：深度比例因子，单位为米/单位。
        """
        return self._depth_scale
    
    def get_properties(self) -> Dict[str, Any]:
        """
        获取相机的内参。
        
        返回：
            dict：包含内参的字典。
        """
        props = super().get_properties()
        props.update({
            'device_id': self.device_id,
            'enable_depth': self._enable_depth,
            'enable_color': self._enable_color,
            'depth_scale': self._depth_scale,
            'intrinsics': self._intrinsics
        })
        return props
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
