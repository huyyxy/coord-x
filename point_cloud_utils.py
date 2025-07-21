"""
点云工具集

本模块提供了处理3D点云的实用函数，包括：
- 从深度图生成点云
- 点云可视化
- 点云处理操作
- 坐标变换
"""
import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Union, List
import cv2


def create_point_cloud_from_depth(depth_image: np.ndarray,
                               camera_matrix: np.ndarray,
                               depth_scale: float = 1.0,
                               depth_trunc: float = 3.0,
                               remove_nan: bool = True) -> o3d.geometry.PointCloud:
    """
    从深度图和相机内参矩阵创建点云
    
    参数:
        depth_image: 包含深度值的2D numpy数组（单位：米）
        camera_matrix: 3x3相机内参矩阵
        depth_scale: 深度值的缩放因子（默认：1.0）
        depth_trunc: 超过此距离的深度值将被截断（单位：米）
        remove_nan: 是否从结果点云中移除NaN/Inf点
        
    返回:
        o3d.geometry.PointCloud: 从深度图创建的点云
    """
    depth_image = depth_image.astype(np.float32)
    
    # 创建Open3D深度图像
    o3d_depth = o3d.geometry.Image(depth_image)
    
    # 创建Open3D相机内参对象
    # Create Open3D intrinsic object
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    
    # Create point cloud from depth
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d_depth,
        intrinsic=intrinsic,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        project_valid_depth_only=remove_nan
    )
    
    return pcd


def visualize_point_cloud(pcd: o3d.geometry.PointCloud,
                        window_name: str = "Point Cloud") -> None:
    """
    使用Open3D可视化点云
    
    参数:
        pcd: 要可视化的Open3D点云
        window_name: 可视化窗口的名称
    """
    if not isinstance(pcd, list):
        pcd = [pcd]
    
    o3d.visualization.draw_geometries(
        pcd,
        window_name=window_name,
        width=1024,
        height=768,
        left=50,
        top=50
    )


def filter_point_cloud(pcd: o3d.geometry.PointCloud,
                     min_bound: Tuple[float, float, float] = (-np.inf, -np.inf, -np.inf),
                     max_bound: Tuple[float, float, float] = (np.inf, np.inf, np.inf)) -> o3d.geometry.PointCloud:
    """
    过滤点云，只保留指定边界内的点
    
    参数:
        pcd: 输入点云
        min_bound: 最小(x, y, z)边界
        max_bound: 最大(x, y, z)边界
        
    返回:
        o3d.geometry.PointCloud: 过滤后的点云
    """
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)


def downsample_point_cloud(pcd: o3d.geometry.PointCloud,
                         voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """
    使用体素网格滤波对点云进行下采样
    
    参数:
        pcd: 输入点云
        voxel_size: 体素网格单元的大小
        
    返回:
        o3d.geometry.PointCloud: 下采样后的点云
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def estimate_normals(pcd: o3d.geometry.PointCloud,
                   radius: float = 0.1,
                   max_nn: int = 30) -> None:
    """
    估计点云的法线
    
    参数:
        pcd: 输入点云（原地修改）
        radius: 法线估计的搜索半径
        max_nn: 考虑的最大邻居数量
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_towards_camera_location()


def transform_point_cloud(pcd: o3d.geometry.PointCloud,
                        transform: np.ndarray) -> o3d.geometry.PointCloud:
    """
    将4x4变换矩阵应用于点云
    
    参数:
        pcd: 输入点云
        transform: 4x4变换矩阵
        
    返回:
        o3d.geometry.PointCloud: 变换后的点云
    """
    return pcd.transform(transform)


def merge_point_clouds(pcds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    """
    将多个点云合并为单个点云
    
    参数:
        pcds: 要合并的点云列表
        
    返回:
        o3d.geometry.PointCloud: 合并后的点云
    """
    if not pcds:
        raise ValueError("No point clouds provided to merge")
        
    merged = o3d.geometry.PointCloud()
    points = []
    colors = []
    
    for pcd in pcds:
        points.append(np.asarray(pcd.points))
        if pcd.has_colors():
            colors.append(np.asarray(pcd.colors))
    
    merged.points = o3d.utility.Vector3dVector(np.vstack(points))
    if colors and len(colors) == len(pcds):
        merged.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    
    return merged


def save_point_cloud(pcd: o3d.geometry.PointCloud, filename: str) -> None:
    """
    将点云保存到文件
    
    支持格式: .ply, .pcd, .pts, .xyz, .xyzn, .xyzrgb
    
    参数:
        pcd: 要保存的点云
        filename: 输出文件名（带扩展名）
    """
    o3d.io.write_point_cloud(filename, pcd)


def load_point_cloud(filename: str) -> o3d.geometry.PointCloud:
    """
    从文件加载点云
    
    支持格式: .ply, .pcd, .pts, .xyz, .xyzn, .xyzrgb
    
    参数:
        filename: 输入文件名（带扩展名）
        
    返回:
        o3d.geometry.PointCloud: 加载的点云
    """
    return o3d.io.read_point_cloud(filename)


# Example usage
if __name__ == "__main__":
    # Example: Create a simple point cloud from a synthetic depth image
    width, height = 640, 480
    fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Create a simple depth image (a slanted plane)
    y, x = np.mgrid[:height, :width]
    depth = 1.0 + 0.5 * (x / width + y / height)
    
    # Create point cloud
    pcd = create_point_cloud_from_depth(depth, camera_matrix)
    
    # Visualize
    visualize_point_cloud(pcd, "Example Point Cloud")
