"""
计算机视觉应用中的坐标变换工具。

本模块提供了在不同坐标系之间转换的辅助函数，
特别是在像素坐标和图像坐标之间进行转换。
"""

import numpy as np
import cv2
from typing import Tuple, Union, Optional, List


def pixel_to_image_coordinates(
    pixel_coords: Union[Tuple[float, float], np.ndarray],
    depth: float,
    camera_matrix: np.ndarray,
    distortion_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    将像素坐标 (u, v) 转换为图像坐标 (x, y, z)
    
    坐标原点说明:
        - 像素坐标系: 原点 (0, 0) 在图像左上角，u轴向右为正，v轴向下为正
        - 图像坐标系: 原点位于图像中心，x轴向右为正，y轴向下为正，z轴沿光轴向前为正
    
    参数:
        pixel_coords: 像素坐标（单位：像素），可以是 (u, v) 元组或形状为 (2,) 或 (N, 2) 的numpy数组
        depth: 深度值（单位：米），表示沿光轴方向的距离
        camera_matrix: 3x3 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        distortion_coeffs: 可选的畸变系数 [k1, k2, p1, p2, k3, ...]
        
    返回:
        numpy.ndarray: 图像坐标 (x, y, z)（单位：米）或形状为 (3,) 或 (N, 3) 的坐标数组
    """
    # 确保输入是numpy数组
    pixel_coords = np.asarray(pixel_coords)
    
    # 处理单个点和多个点的情况
    single_point = pixel_coords.ndim == 1
    if single_point:
        pixel_coords = pixel_coords.reshape(1, -1)
    
    # 如果提供了畸变系数，则去畸变
    if distortion_coeffs is not None:
        pixel_coords = cv2.undistortPoints(
            pixel_coords.astype(np.float32),
            camera_matrix,
            distortion_coeffs,
            P=camera_matrix
        ).reshape(-1, 2)
    
    # 提取内参
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # 转换为归一化图像坐标
    x = (pixel_coords[:, 0] - cx) / fx
    y = (pixel_coords[:, 1] - cy) / fy
    
    # 通过深度值缩放到3D坐标
    z = np.full(len(pixel_coords), depth)
    x_3d = x * depth
    y_3d = y * depth
    
    # 组合坐标
    image_coords = np.column_stack((x_3d, y_3d, z))
    
    return image_coords[0] if single_point else image_coords


def image_to_pixel_coordinates(
    image_coords: Union[Tuple[float, float, float], np.ndarray],
    camera_matrix: np.ndarray,
    distortion_coeffs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    将图像坐标 (x, y, z) 转换为像素坐标 (u, v)
    
    坐标原点说明:
        - 图像坐标系: 原点位于图像中心，x轴向右为正，y轴向下为正，z轴沿光轴向前为正
        - 像素坐标系: 原点 (0, 0) 在图像左上角，u轴向右为正，v轴向下为正
    
    参数:
        image_coords: 图像坐标（单位：米），可以是 (x, y, z) 元组或形状为 (3,) 或 (N, 3) 的numpy数组
        camera_matrix: 3x3 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        distortion_coeffs: 可选的畸变系数 [k1, k2, p1, p2, k3, ...]
        
    返回:
        numpy.ndarray: 像素坐标 (u, v)（单位：像素）或形状为 (2,) 或 (N, 2) 的坐标数组
    """
    # 确保输入是numpy数组
    image_coords = np.asarray(image_coords)
    
    # 处理单个点和多个点的情况
    single_point = image_coords.ndim == 1
    if single_point:
        image_coords = image_coords.reshape(1, -1)
    
    # 通过z坐标(深度)归一化
    x = image_coords[:, 0] / image_coords[:, 2]
    y = image_coords[:, 1] / image_coords[:, 2]
    
    # 如果提供了畸变系数，则应用畸变
    if distortion_coeffs is not None:
        # 转换为齐次坐标
        points = np.column_stack((x, y, np.ones_like(x)))
        
        # 应用畸变
        r2 = x**2 + y**2
        k1, k2, p1, p2 = distortion_coeffs[:4]
        radial = 1 + k1 * r2 + k2 * r2**2
        
        if len(distortion_coeffs) > 4:  # 处理径向畸变k3
            k3 = distortion_coeffs[4]
            radial += k3 * r2**3
            
        x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_distorted = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        
        x, y = x_distorted, y_distorted
    
    # 应用相机内参矩阵
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    u = x * fx + cx
    v = y * fy + cy
    
    pixel_coords = np.column_stack((u, v))
    
    return pixel_coords[0] if single_point else pixel_coords


def image_to_camera_coordinates(
    image_coords: Union[Tuple[float, float, float], np.ndarray],
    camera_matrix: np.ndarray
) -> np.ndarray:
    """
    将图像坐标 (x, y, z) 转换为相机坐标 (Xc, Yc, Zc)
    
    图像坐标系: 以图像中心为原点，x轴向右，y轴向下，z轴向前
    相机坐标系: 以相机光心为原点，x轴向右，y轴向下，z轴向前
    
    参数:
        image_coords: 图像坐标（单位：米），可以是 (x, y, z) 元组或形状为 (3,) 或 (N, 3) 的numpy数组
        camera_matrix: 3x3 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        
    返回:
        numpy.ndarray: 相机坐标 (Xc, Yc, Zc)（单位：米）或形状为 (3,) 或 (N, 3) 的坐标数组
    """
    # 确保输入是numpy数组
    image_coords = np.asarray(image_coords)
    
    # 处理单个点和多个点的情况
    single_point = image_coords.ndim == 1
    if single_point:
        image_coords = image_coords.reshape(1, -1)
    
    # 提取内参
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # 归一化平面坐标 (x/z, y/z, 1)
    x_img, y_img, z = image_coords[:, 0], image_coords[:, 1], image_coords[:, 2]
    
    # 转换为相机坐标
    Xc = (x_img - cx) * z / fx
    Yc = (y_img - cy) * z / fy
    Zc = z
    
    # 组合结果
    camera_coords = np.column_stack((Xc, Yc, Zc))
    
    return camera_coords[0] if single_point else camera_coords


def camera_to_image_coordinates(
    camera_coords: Union[Tuple[float, float, float], np.ndarray],
    camera_matrix: np.ndarray
) -> np.ndarray:
    """
    将相机坐标 (Xc, Yc, Zc) 转换为图像坐标 (x, y, z)
    
    相机坐标系: 以相机光心为原点，x轴向右，y轴向下，z轴向前
    图像坐标系: 以图像中心为原点，x轴向右，y轴向下，z轴向前
    
    参数:
        camera_coords: 相机坐标（单位：米），可以是 (Xc, Yc, Zc) 元组或形状为 (3,) 或 (N, 3) 的numpy数组
        camera_matrix: 3x3 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        
    返回:
        numpy.ndarray: 图像坐标 (x, y, z)（单位：米）或形状为 (3,) 或 (N, 3) 的坐标数组
    """
    # 确保输入是numpy数组
    camera_coords = np.asarray(camera_coords)
    
    # 处理单个点和多个点的情况
    single_point = camera_coords.ndim == 1
    if single_point:
        camera_coords = camera_coords.reshape(1, -1)
    
    # 提取内参
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # 转换为图像坐标
    Xc, Yc, Zc = camera_coords[:, 0], camera_coords[:, 1], camera_coords[:, 2]
    x = Xc * fx / Zc + cx
    y = Yc * fy / Zc + cy
    z = Zc
    
    # 组合结果
    image_coords = np.column_stack((x, y, z))
    
    return image_coords[0] if single_point else image_coords


def transform_coordinates(
    points: np.ndarray,
    transformation_matrix: np.ndarray
) -> np.ndarray:
    """
    将4x4变换矩阵应用于3D点
    
    参数:
        points: 输入的3D点（单位：米），形状为 (3,) 或 (N, 3) 的numpy数组
        transformation_matrix: 4x4 变换矩阵
        
    返回:
        numpy.ndarray: 变换后的3D点（单位：米），形状与输入相同
    """
    # 确保点是2D数组
    points = np.asarray(points)
    single_point = points.ndim == 1
    if single_point:
        points = points.reshape(1, -1)
    
    # 转换为齐次坐标
    homogeneous = np.column_stack((points, np.ones(len(points))))
    
    # 应用变换
    transformed = (transformation_matrix @ homogeneous.T).T
    
    # 转换回3D坐标
    transformed_3d = transformed[:, :3] / transformed[:, 3:4]
    
    return transformed_3d[0] if single_point else transformed_3d
