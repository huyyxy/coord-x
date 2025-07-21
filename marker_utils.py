import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import os

# 预定义的ArUco字典
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
}

class ArucoMarker:
    """
    ArUco标记工具类，用于生成和检测ArUco标记
    """
    def __init__(self, dict_type: str = "DICT_6X6_250"):
        """
        初始化ArUco标记检测器
        
        参数:
            dict_type: ArUco字典类型，默认为"DICT_6X6_250"
        """
        self.dict_type = dict_type
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
    
    def generate_marker(self, marker_id: int, size_pixels: int = 300, 
                       border_bits: int = 1, save_path: Optional[str] = None) -> np.ndarray:
        """
        生成单个ArUco标记图像
        
        参数:
            marker_id: 标记ID，必须在字典范围内
            size_pixels: 输出图像的尺寸（像素）
            border_bits: 标记边框的宽度（以位为单位）
            save_path: 保存路径，如果提供则保存图像
            
        返回:
            numpy数组格式的标记图像
        """
        marker_img = cv2.aruco.generateImageMarker(
            self.aruco_dict, 
            id=marker_id, 
            sidePixels=size_pixels,
            borderBits=border_bits
        )
        
        if save_path:
            cv2.imwrite(save_path, marker_img)
            
        return marker_img
    
    def generate_markers(self, num_markers: int, start_id: int = 0, 
                        size_pixels: int = 300, output_dir: str = "aruco_markers") -> List[np.ndarray]:
        """
        批量生成ArUco标记图像
        
        参数:
            num_markers: 要生成的标记数量
            start_id: 起始ID
            size_pixels: 每个标记的尺寸（像素）
            output_dir: 输出目录
            
        返回:
            标记图像列表
        """
        os.makedirs(output_dir, exist_ok=True)
        markers = []
        
        for i in range(num_markers):
            marker_id = start_id + i
            save_path = os.path.join(output_dir, f"aruco_marker_{marker_id}.png")
            marker = self.generate_marker(marker_id, size_pixels, save_path=save_path)
            markers.append(marker)
            
        return markers
    
    def detect_markers(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[int], Optional[List[np.ndarray]]]:
        """
        检测图像中的ArUco标记
        
        参数:
            image: 输入图像（BGR或灰度）
            
        返回:
            Tuple 包含三个元素 (corners 角点列表, ids ID列表, rejected 拒绝的候选列表):
            - corners: List[np.ndarray], 每个元素是一个形状为 (1,4,2) 的numpy数组
            * 1: 表示检测到的标记数量（每个标记一个数组）
            * 4: 标记的四个角点
            * 2: 每个角点的 (x, y) 像素坐标
            * 角点顺序: 顺时针方向，依次为 [左上, 右上, 右下, 左下]
            * 示例: np.array([[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]], dtype=np.float32)
            
            - ids: List[int], 每个检测到的标记的ID，顺序与corners中的标记一一对应
            * 示例: [0, 1] 表示检测到ID为0和1的两个标记
            
            - rejected: List[np.ndarray] | None, 被拒绝的候选标记（未通过编码检查的标记）
            * 格式与corners相同，每个元素是形状为(1,4,2)的numpy数组
            * 可能为None，如果没有检测到被拒绝的候选
            
            返回值示例:
                (
                    [
                        array([[[100., 200.], [150., 200.], [150., 250.], [100., 250.]]], dtype=float32),
                        array([[[300., 400.], [350., 400.], [350., 450.], [300., 450.]]], dtype=float32)
                    ],
                    [0, 1],  # 两个标记的ID
                    None     # 没有被拒绝的候选
                )
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        return corners, ids, rejected
    
    def get_marker_centers_3d(self, corners: List[np.ndarray], ids: List[int], 
                            marker_length: float, 
                            camera_matrix: np.ndarray,
                            dist_coeffs: np.ndarray = None) -> dict:
        """
        获取检测到的ArUco标记在相机坐标系中的3D中心坐标
        
        参数:
            corners: 检测到的标记角点列表
            ids: 检测到的标记ID列表
            marker_length: 标记的实际物理尺寸（单位：米）
            camera_matrix: 相机内参矩阵 [3x3]
            dist_coeffs: 相机畸变系数 [1x5] 或 [5x1]，如果为None则使用零畸变
            
        返回:
            dict: 包含标记ID和对应3D坐标的字典，格式为 {marker_id: (x, y, z), ...}
                  x, y, z 是相机坐标系中的坐标（单位：米）
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1))
            
        marker_centers_3d = {}
        
        for i, corner in enumerate(corners):
            # 估计标记的位姿
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, marker_length, camera_matrix, dist_coeffs
            )
            
            # 获取平移向量（即标记中心在相机坐标系中的3D位置）
            tvec = tvecs[0][0]  # shape: (3,)
            
            marker_id = int(ids[i][0]) if ids is not None else i
            marker_centers_3d[marker_id] = tuple(tvec)
            
        return marker_centers_3d
        
    def draw_detected_markers(self, image: np.ndarray, corners: List[np.ndarray], 
                            ids: List[int] = None) -> np.ndarray:
        """
        在图像上绘制检测到的标记
        
        参数:
            image: 原始图像
            corners: 检测到的标记角点
            ids: 检测到的标记ID
            
        返回:
            绘制了标记的图像
        """
        if ids is not None:
            return cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
        else:
            return cv2.aruco.drawDetectedMarkers(image.copy(), corners)
    
    def estimate_pose_single_marker(self, corners: np.ndarray, marker_length: float, 
                                  camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计单个标记的姿态（位置和旋转）
        
        参数:
            corners: 标记的角点坐标
            marker_length: 标记的实际物理尺寸（单位：米）
            camera_matrix: 相机内参矩阵
            dist_coeffs: 相机畸变系数
            
        返回:
            (旋转向量, 平移向量)
        """
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )
        return rvecs[0][0], tvecs[0][0]


def create_grid_board(num_markers_x: int, num_markers_y: int, 
                     marker_length: float, marker_separation: float, 
                     dict_type: str = "DICT_6X6_250", first_id: int = 0) -> Tuple[cv2.aruco.Board, Dict]:
    """
    创建ArUco网格板
    
    参数:
        num_markers_x: X轴方向的标记数量
        num_markers_y: Y轴方向的标记数量
        marker_length: 标记的实际物理尺寸（单位：米）
        marker_separation: 标记之间的间距（单位：米）
        dict_type: ArUco字典类型
        first_id: 第一个标记的ID
        
    返回:
        (ArUco板对象, 参数字典)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    
    board = cv2.aruco.GridBoard(
        size=(num_markers_x, num_markers_y),
        markerLength=marker_length,
        markerSeparation=marker_separation,
        dictionary=aruco_dict,
        firstMarker=first_id
    )
    
    return board, {"aruco_dict": aruco_dict, "parameters": cv2.aruco.DetectorParameters()}


def detect_and_draw_markers(image_path: str, dict_type: str = "DICT_6X6_250", 
                          output_path: Optional[str] = None) -> np.ndarray:
    """
    检测图像中的ArUco标记并绘制结果
    
    参数:
        image_path: 输入图像路径
        dict_type: ArUco字典类型
        output_path: 输出图像路径（可选）
        
    返回:
        绘制了检测结果的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 初始化检测器
    aruco_detector = ArucoMarker(dict_type)
    
    # 检测标记
    corners, ids, _ = aruco_detector.detect_markers(image)
    
    # 绘制检测结果
    if ids is not None and len(ids) > 0:
        output_image = aruco_detector.draw_detected_markers(image, corners, ids)
        print(f"检测到 {len(ids)} 个标记: {ids.flatten()}")
    else:
        output_image = image.copy()
        print("未检测到任何标记")
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, output_image)
        print(f"结果已保存到: {output_path}")
    
    return output_image


# 使用示例
if __name__ == "__main__":
    # 1. 创建ArUco检测器
    aruco = ArucoMarker("DICT_6X6_250")
    
    # 2. 生成并保存标记
    output_dir = "aruco_markers"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成5个标记，ID从0到4
    aruco.generate_markers(5, 0, 300, output_dir)
    print(f"已生成标记到目录: {output_dir}")
    
    # 3. 检测图像中的标记
    # 注意：需要准备一张包含ArUco标记的测试图像
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        output_image = detect_and_draw_markers(
            test_image_path, 
            "DICT_6X6_250",
            "detected_markers.jpg"
        )
        print("标记检测完成，结果已保存")
    else:
        print(f"未找到测试图像: {test_image_path}")
        print(f"请准备一张包含ArUco标记的图像并保存为: {test_image_path}")
