"""
使用摄像头进行ArUco标记检测的示例

本示例演示了如何使用相机工厂获取相机实例，
在视频流中检测ArUco标记，并实时显示结果。
"""
import cv2
import numpy as np
import argparse
from typing import Tuple, List, Optional

# Import the camera factory and ArucoMarker
from camera.factory import CameraFactory
from marker_utils import ArucoMarker

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='ArUco Marker Detection with Camera')
    parser.add_argument('--camera-type', type=str, default='opencv',
                       choices=['opencv', 'realsense'],
                       help='Type of camera to use (default: opencv)')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera frame height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera FPS (default: 30)')
    parser.add_argument('--dict-type', type=str, default='DICT_6X6_250',
                       help='ArUco dictionary type (default: DICT_6X6_250)')
    return parser.parse_args()

def draw_marker_info(frame: np.ndarray, corners: List[np.ndarray], ids: List[int], 
                    rejected: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    在帧上绘制检测到的标记及其信息
    
    参数:
        frame: 输入图像帧
        corners: 检测到的标记角点列表
        ids: 检测到的标记ID列表
        rejected: 被拒绝的候选标记列表
        
    返回:
        绘制了标记和信息的图像帧
    """
    # 绘制检测到的标记
    if len(corners) > 0:
        # 绘制检测到的标记
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # 为每个标记绘制附加信息
        for i, marker_id in enumerate(ids):
            # 获取标记中心点
            c = corners[i][0]
            center = c.mean(axis=0).astype(int)
            
            # 绘制标记ID和中心点
            cv2.putText(frame, f"ID: {marker_id[0]}", 
                       (center[0] - 30, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
            
            # 绘制四个角点
            for j, corner in enumerate(c):
                cv2.circle(frame, tuple(corner.astype(int)), 4, (255, 0, 0), -1)
                cv2.putText(frame, str(j+1), 
                           tuple((corner + np.array([10, 5])).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # 绘制被拒绝的标记（用于调试）
    if rejected is not None and len(rejected) > 0:
        for rejected_marker in rejected:
            cv2.polylines(frame, [rejected_marker.astype(int)], True, (0, 0, 255), 2)
    
    return frame

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 使用工厂模式初始化相机
    try:
        camera = CameraFactory.create_camera(
            camera_type=args.camera_type,
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        # 支持的相机类型
        if not camera.open():
            print(f"错误：无法打开{args.camera_type}相机")
            return
            
        print(f"相机打开成功：{camera.get_resolution()} {camera.get_fps()}")
        
        # 初始化ArUco标记检测器
        aruco_detector = ArucoMarker(dict_type=args.dict_type)
        
        print("Press 'q' to quit")
        
        while True:
            # 从相机捕获帧
            ret, frame = camera.read()
            if not ret:
                print("获取帧失败")
                break
                
            # 检测ArUco标记
            corners, ids, rejected = aruco_detector.detect_markers(frame)
            
            # 在帧上绘制标记信息
            frame = draw_marker_info(frame, corners, ids, rejected)
            
            # 显示帧
            cv2.imshow('ArUco标记检测', frame)
            
            # 按'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # 清理资源
        if 'camera' in locals():
            camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
