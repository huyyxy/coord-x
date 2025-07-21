"""
相机工厂类，用于创建相机实例。
"""
from typing import Dict, Any, Type, Optional, Union

from .base import CameraBase
from .opencv_camera import OpenCVCamera
from .realsense_camera import RealSenseCamera


class CameraFactory:
    """
    用于创建相机实例的工厂类。
    """
    
    # 支持的相机类型
    CAMERA_TYPES = {
        'opencv': OpenCVCamera,
        'realsense': RealSenseCamera,
        # 在此处添加更多实现的相机类型
    }
    
    @classmethod
    def create_camera(cls, 
                     camera_type: str = 'opencv',
                     **kwargs) -> CameraBase:
        """
        创建指定类型的相机实例。
        
        参数:
            camera_type: 要创建的相机类型 ('opencv' 或 'realsense')
            **kwargs: 传递给相机构造函数的额外参数
            
        返回:
            CameraBase: 请求的相机类型的实例
            
        异常:
            ValueError: 如果指定的相机类型不受支持
        """
        camera_type = camera_type.lower()
        if camera_type not in cls.CAMERA_TYPES:
            raise ValueError(f"Unsupported camera type: {camera_type}. "
                           f"Available types: {list(cls.CAMERA_TYPES.keys())}")
        
        camera_class = cls.CAMERA_TYPES[camera_type]
        return camera_class(**kwargs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CameraBase:
        """
        从配置字典创建相机实例。
        
        配置示例:
            {
                'type': 'opencv',  # 或 'realsense'
                'camera_id': 0,
                'width': 1280,
                'height': 720,
                'fps': 30,
                'enable_depth': True  # 仅适用于 RealSense
            }
            
        参数:
            config: 包含相机配置的字典
            
        返回:
            CameraBase: 配置好的相机实例
            
        异常:
            ValueError: 如果配置无效
            KeyError: 如果缺少必需的配置键
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        if 'type' not in config:
            raise KeyError("Configuration must include 'type' key")
            
        # 创建配置的副本以避免修改输入
        config = config.copy()
        camera_type = config.pop('type')
        
        return cls.create_camera(camera_type, **config)
    
    @classmethod
    def register_camera_type(cls, 
                           name: str, 
                           camera_class: Type[CameraBase], 
                           override: bool = False) -> None:
        """
        向工厂注册新的相机类型。
        
        参数:
            name: 注册相机类型的名称
            camera_class: 要注册的相机类（必须是 CameraBase 的子类）
            override: 如果为 True，则允许覆盖已存在的相机类型
            
        异常:
            TypeError: 如果 camera_class 不是 CameraBase 的子类
            ValueError: 如果相机类型已注册且 override 为 False
        """
        if not issubclass(camera_class, CameraBase):
            raise TypeError(f"Camera class must be a subclass of CameraBase, got {camera_class}")
            
        name = name.lower()
        if not override and name in cls.CAMERA_TYPES:
            raise ValueError(f"Camera type '{name}' is already registered")
            
        cls.CAMERA_TYPES[name] = camera_class
    
    @classmethod
    def list_available_cameras(cls) -> Dict[str, Type[CameraBase]]:
        """
        获取可用相机类型的字典。
        
        返回:
            dict: 相机类型名称到对应类的映射
        """
        return cls.CAMERA_TYPES.copy()
    
    @classmethod
    def detect_available_cameras(cls) -> Dict[str, Dict[str, Any]]:
        """
        检测系统上可用的相机。
        
        返回：
            dict：包含检测到的相机信息的字典
        """
        available = {}
        
        # 检测 OpenCV 相机
        opencv_cameras = {}
        for i in range(10):  # 检查前10个索引
            cap = None
            try:
                cap = OpenCVCamera(camera_id=i)
                if cap.open():
                    opencv_cameras[f'opencv_{i}'] = {
                        'type': 'opencv',
                        'camera_id': i,
                        'resolution': cap.get_resolution(),
                        'fps': cap.get_fps()
                    }
                    cap.close()
            except Exception as e:
                if cap:
                    cap.close()
                break
        
        if opencv_cameras:
            available['opencv'] = opencv_cameras
        
        # 检测 RealSense 相机
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            realsense_devices = {}
            
            for i, device in enumerate(ctx.devices):
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                realsense_devices[f'realsense_{i}'] = {
                    'type': 'realsense',
                    'serial': serial,
                    'name': name,
                    'device_id': serial
                }
            
            if realsense_devices:
                available['realsense'] = realsense_devices
                
        except ImportError:
            pass  # pyrealsense2 未安装
        except Exception as e:
            print(f"检测 RealSense 相机时出错: {e}")
        
        return available
