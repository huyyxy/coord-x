"""
ArUco 标记检测与姿态估计

本脚本实现了使用普通摄像头或RealSense深度相机进行ArUco标记的检测与姿态估计功能。
主要功能包括：
1. 支持OpenCV和RealSense两种相机类型
2. 加载相机内参和畸变系数进行相机标定
3. 实时检测图像中的ArUco标记
4. 估计标记的3D姿态（位置和旋转）
5. 在图像上可视化标记的边界框和坐标轴

使用方法：
python aruco_detection_example.py [--camera-type {opencv,realsense}] [--camera-id CAMERA_ID]
"""

import traceback
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).parent.absolute()))

from marker_utils import ArucoMarker
from camera.factory import CameraFactory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ArUco Marker Detection with Camera')
    parser.add_argument('--camera-type', type=str, default='opencv',
                        choices=['opencv', 'realsense'],
                        help='Camera type (opencv or realsense)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera frame height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera FPS (default: 30)')
    parser.add_argument('--marker-length', type=float, default=0.068,
                        help='Length of the ArUco marker in meters (default: 0.068m)')
    parser.add_argument('--dict-type', type=str, default='DICT_6X6_250',
                        help='ArUco dictionary type (default: DICT_6X6_250)')
    return parser.parse_args()

def load_camera_matrix(camera_matrix_path='camera_matrix.npy', 
                      dist_coeffs_path='dist_coeffs.npy'):
    """从文件加载相机矩阵和畸变系数。"""
    try:
        camera_matrix = np.load(camera_matrix_path)
        dist_coeffs = np.load(dist_coeffs_path)
        print("Loaded camera calibration data")
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        print("Warning: Camera calibration files not found. Using default values.")
        # 默认值（替换为你的相机内参）
        width, height = 1280, 720
        fx = width * 1.2  # 以像素为单位的近似焦距
        fy = height * 1.2
        cx = width / 2.0  # 图像中心x坐标
        cy = height / 2.0  # 图像中心y坐标
        camera_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        return camera_matrix, dist_coeffs

def draw_axis(img, corners, imgpts):
    """
    在图像上绘制3D坐标轴。
    
    参数:
        img: 输入图像
        corners: 标记的角点
        imgpts: 3D坐标轴在图像平面上的投影点
    
    返回:
        绘制了坐标轴的图像
    """
    # 将角点转换为整数元组(x, y)
    corner = tuple(corners[0].astype(int).ravel())
    # 将imgpts转换为整数元组并确保是2D点
    imgpts = imgpts.astype(int).reshape(-1, 2)
    # 绘制三个坐标轴（X: 红色, Y: 绿色, Z: 蓝色）
    img = cv2.line(img, corner, tuple(imgpts[0]), (0, 0, 255), 5)  # X轴（红色）
    img = cv2.line(img, corner, tuple(imgpts[1]), (0, 255, 0), 5)  # Y轴（绿色）
    img = cv2.line(img, corner, tuple(imgpts[2]), (255, 0, 0), 5)  # Z轴（蓝色）
    return img

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化ArUco检测器
    aruco_detector = ArucoMarker(dict_type=args.dict_type)
    
    # 加载相机标定数据
    camera_matrix, dist_coeffs = load_camera_matrix()
    
    # 创建相机实例
    try:
        camera = CameraFactory.create_camera(
            camera_type=args.camera_type,
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        if not camera.open():
            print(f"Error: Could not open {args.camera_type} camera")
            return
            
        print(f"Camera opened successfully: {camera.get_resolution()} {camera.get_fps()}")
        
        while True:
            # 从相机读取帧
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # 检测ArUco标记
            corners, ids, _ = aruco_detector.detect_markers(frame)
            
            if ids is not None and len(ids) > 0:
                # 使用ArucoMarker类的方法绘制检测到的标记
                frame = aruco_detector.draw_detected_markers(frame, corners, ids)
                
                # 获取所有检测到标记的3D中心坐标
                marker_centers = aruco_detector.get_marker_centers_3d(
                    corners, 
                    ids,
                    args.marker_length,
                    camera_matrix,
                    dist_coeffs
                )

                # 将检测结果打印到控制台及显示在图像上
                i = 0
                for marker_id, tvec in marker_centers.items():
                    print(f"Marker ID {marker_id}: X={tvec[0]:.3f}m, Y={tvec[1]:.3f}m, Z={tvec[2]:.3f}m")

                    # 显示3D位置信息
                    pos_text = f"ID {marker_id}: X={tvec[0]:.3f}m, Y={tvec[1]:.3f}m, Z={tvec[2]:.3f}m"
                    frame = cv2.putText(
                        frame, 
                        pos_text, 
                        (10, 30 + 30 * i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    i += 1
            
            # 显示帧
            cv2.imshow('ArUco Marker Detection', frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        traceback.print_exc()
    finally:
        # 清理资源
        if 'camera' in locals():
            camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
