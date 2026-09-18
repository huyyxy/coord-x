"""
RealSense YOLO 杯子检测示例

本示例演示了如何使用Intel RealSense深度相机和YOLOv8模型进行实时杯子检测和3D定位。
主要功能包括：
1. 初始化RealSense相机并获取彩色和深度图像
2. 使用YOLOv8模型检测图像中的杯子
3. 计算检测到的杯子在相机坐标系中的3D位置
4. 实时显示检测结果和3D坐标

使用方法：
1. 确保已安装必要的依赖（pyrealsense2, ultralytics, opencv-python, numpy）
2. 连接Intel RealSense相机
3. 运行脚本：python realsense_yolo_cup_detection.py
4. 按'q'键退出程序

注意：
- 本示例使用COCO数据集预训练的YOLOv8模型，其中杯子的类别ID为41
- 3D位置计算基于深度相机的深度信息和相机内参
- 确保相机已正确安装并授予相应权限
"""


import cv2
import numpy as np
from ultralytics import YOLO
import time
import pyrealsense2 as rs
from camera import CameraFactory


def get_depth_colormap(depth_image):
    # 步骤 1：设置深度范围（单位：米）
    # min_depth 和 max_depth 用于控制深度图像的显示范围
    # min_depth = depth_image.min()
    # max_depth = depth_image.max()
    min_depth = 0.3  # 最小深度阈值
    max_depth = 1.0  # 最大深度阈值
    
    # 步骤 2：将深度值从米转换为毫米（因为RealSense默认使用毫米为单位）
    min_depth_units = min_depth * 1000
    max_depth_units = max_depth * 1000

    # 步骤 3：将深度值限制在指定范围内，超出范围的值会被截断
    # 这样可以确保所有深度值都在[min_depth_units, max_depth_units]之间
    depth_image_clipped = np.clip(depth_image, min_depth_units, max_depth_units)

    # 步骤 4：将深度值归一化到0-255范围，并转换为8位无符号整型
    # 这是为了后续应用颜色映射做准备
    depth_image_8bit = cv2.normalize(depth_image_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # 步骤 5：应用JET颜色映射，将灰度深度图转换为彩色图像
    # 不同颜色代表不同的深度值，便于直观理解深度信息
    depth_colormap = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
    # 如果需要，可以进行颜色反转
    # 因为 realsense 的 colorizer 是近处红/远处蓝，而 OpenCV 的 JET 是低值蓝/高值红
    # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_RGB2BGR)
    return depth_colormap

def get_depth_colormap4(depth_image):
    # 步骤 1: 反转深度值方向（关键！）
    # 因为 realsense 的 colorizer 是近处红/远处蓝，而 OpenCV 的 JET 是低值蓝/高值红
    min_depth = 300  # 毫米
    max_depth = 3000 # 毫米
    depth_inverted = np.where(
        (depth_image > min_depth) & (depth_image < max_depth),
        max_depth - (depth_image - min_depth),
        0
    )

    # 步骤 2: 归一化处理
    depth_normalized = cv2.normalize(
        depth_inverted, 
        None, 
        0, 
        255, 
        cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )

    # 步骤 3: 应用 JET 色彩映射
    # 应用颜色映射（JET 映射与 RealSense 默认的彩色映射类似）
    # cv2.COLORMAP_JET - 类似 RealSense 默认的彩色映射
    # cv2.COLORMAP_RAINBOW - 彩虹色映射
    # cv2.COLORMAP_HOT - 热力图效果
    # cv2.COLORMAP_COOL - 冷色调映射
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # 步骤 4: 将无效点设为黑色（可选）
    depth_colormap[depth_image == 0] = [0, 0, 0]
    return depth_colormap

def get_depth_colormap3(depth_image):
    # 获取有效深度值（忽略0值）
    valid_depths = depth_image[depth_image > 0]

    if len(valid_depths) > 0:
        # 计算实际深度范围（最近点到最远点）
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        
        # 反转深度方向（使近处值大，远处值小）
        depth_inverted = np.where(
            depth_image > 0,
            max_depth - (depth_image - min_depth),  # 关键：基于实际范围的反转
            0
        )
        
        # 归一化到0-255范围（基于实际深度范围）
        depth_normalized = np.uint8(255 * (depth_inverted - np.min(depth_inverted[depth_inverted > 0])) / 
                                (max_depth - min_depth))
        
        # 应用直方图均衡化（模拟RealSense默认行为）
        depth_equ = cv2.equalizeHist(depth_normalized)
        
        # 应用JET色彩映射
        depth_colormap = cv2.applyColorMap(depth_equ, cv2.COLORMAP_JET)
        
        # 将无效点设为黑色
        depth_colormap[depth_image == 0] = [0, 0, 0]
    else:
        depth_colormap = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
    return depth_colormap

def get_depth_colormap2(depth_image):
    # 将深度数据归一化到 0-255 范围
    depth_colormap = cv2.normalize(
        depth_image, 
        None, 
        0, 
        255, 
        cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )

    depth_colormap = 255 - depth_colormap  # 反转颜色
    # 应用颜色映射（JET 映射与 RealSense 默认的彩色映射类似）
    # cv2.COLORMAP_JET - 类似 RealSense 默认的彩色映射
    # cv2.COLORMAP_RAINBOW - 彩虹色映射
    # cv2.COLORMAP_HOT - 热力图效果
    # cv2.COLORMAP_COOL - 冷色调映射
    depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_SPRING)
    return depth_colormap

def get_depth_colormap1(depth_image):
    # 步骤 1: 反转深度值方向（关键！）
    # 因为 realsense 的 colorizer 是近处红/远处蓝，而 OpenCV 的 JET 是低值蓝/高值红
    depth_inverted = np.where(
        depth_image > 0,                   # 忽略无效点（深度=0）
        np.max(depth_image) - depth_image, # 反转有效深度值
        0
    )

    # 步骤 2: 归一化处理
    depth_normalized = cv2.normalize(
        depth_inverted, 
        None, 
        0, 
        255, 
        cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )

    # 步骤 3: 应用 JET 色彩映射
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # 步骤 4: 将无效点设为黑色（可选）
    depth_colormap[depth_image == 0] = [0, 0, 0]
    return depth_colormap

class RealSenseYOLOCupDetector:
    def __init__(self, model_name='yolov8m', conf_threshold=0.5, width=640, height=480, fps=30):
        """
        初始化RealSense相机和YOLO模型。
        
        参数:
            model_name (str): YOLO模型名称 (默认: 'yolov8m')
            conf_threshold (float): 检测的置信度阈值
            width (int): 图像宽度 (默认: 640)
            height (int): 图像高度 (默认: 480)
            fps (int): 帧率 (默认: 30)
        """
        self.conf_threshold = conf_threshold
        self.cup_class_id = 41  # COCO数据集中'cup'类别的ID
        
        # 使用CameraFactory初始化RealSense相机
        self.camera = CameraFactory.create_camera(
            camera_type='realsense',
            width=width,
            height=height,
            fps=fps,
            enable_depth=True,
            enable_color=True
        )
        if self.camera.open():
            # 获取相机内参
            self.intrinsics = self.camera.get_properties().get('intrinsics', None)
            
            # 获取深度传感器的深度比例
            self.depth_scale = self.camera.get_properties().get('depth_scale', None)
        
        # 加载YOLO模型
        self.model = YOLO(f'{model_name}.pt')  # Loads official YOLOv8 model
        self.conf_threshold = conf_threshold
    
    def get_frames(self):
        """
        从RealSense相机获取彩色和深度帧。
        
        返回：
            tuple: (成功标志, 彩色图像, 深度图像, 深度比例)
        """
        try:
            # 使用camera factory获取帧
            success, frames = self.camera.read_frames()
            if not success or 'color' not in frames or 'depth' not in frames:
                return False, None, None, None
                
            # 获取彩色和深度图像
            color_image = frames['color']
            depth_image = frames['depth']

            if self.depth_scale is None:
                # 获取深度传感器的深度比例
                self.depth_scale = self.camera.get_properties().get('depth_scale', None)
            
            return True, color_image, depth_image, self.depth_scale
            
        except Exception as e:
            print(f"获取帧时出错: {e}")
            return False, None, None, None
    
    def detect_cups(self, color_image):
        """
        使用YOLO在彩色图像中检测杯子。
        
        参数：
            color_image: 输入的彩色图像(BGR格式)
            
        返回：
            list: 检测到的杯子的边界框列表 [x1, y1, x2, y2, 置信度, 类别ID]
        """
        # 运行YOLO推理(Ultralytics YOLO默认期望BGR格式)
        results = self.model(color_image, conf=self.conf_threshold)
        
        # 处理检测结果
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # 合并边界框、置信度和类别ID
            for box, conf, class_id in zip(boxes, confs, class_ids):
                if class_id == self.cup_class_id:
                    detections.append([*box, conf, class_id])
        
        return detections
    
    def get_3d_position(self, bbox, depth_image, depth_scale):
        """
        计算边界框中心点在相机坐标系中的3D位置。
        
        参数：
            bbox: 边界框 [x1, y1, x2, y2, 置信度, 类别ID]
            depth_image: 来自RealSense的深度图像
            depth_scale: 深度比例
            
        返回：
            tuple: 相机坐标系中的(x, y, z)坐标，单位为米

        注意：在 RealSense 相机的坐标系中：
            原点（Origin）：位于相机的光心（optical center）
            Z 轴：从相机指向正前方（与光轴重合）
            X 轴：向右（从相机视角看）
            Y 轴：向下（从相机视角看）
        """
        if self.intrinsics is None:
            self.intrinsics = self.camera.get_properties().get('intrinsics', None)
        # 1. 获取图像尺寸
        h, w = depth_image.shape[:2]
        
        # 2. 确保边界框在图像范围内
        x1 = max(0, min(int(bbox[0]), w-1))
        y1 = max(0, min(int(bbox[1]), h-1))
        x2 = max(0, min(int(bbox[2]), w-1))
        y2 = max(0, min(int(bbox[3]), h-1))
        
        # 3. 计算中心点（使用浮点运算）
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # 4. 获取中心点周围区域的中值深度（更鲁棒）
        half_size = 5  # 可以调整
        x_start = max(0, int(center_x) - half_size)
        y_start = max(0, int(center_y) - half_size)
        x_end = min(w, int(center_x) + half_size + 1)
        y_end = min(h, int(center_y) + half_size + 1)
        
        roi = depth_image[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            return None
        
        # 使用非零深度值的中位数
        valid_depths = roi[roi > 0]
        if valid_depths.size == 0:
            return None
        
        depth = np.median(valid_depths) * depth_scale  # 转换为米
        
        # 5. 将像素坐标转换为相机坐标
        x = (center_x - self.intrinsics.ppx) / self.intrinsics.fx * depth
        y = (center_y - self.intrinsics.ppy) / self.intrinsics.fy * depth
        z = depth
        
        return (x, y, z)
    
    def visualize(self, color_image, depth_colormap, detections, positions):
        """
        可视化检测结果。
        
        参数：
            color_image: 彩色图像(BGR格式)
            depth_colormap: 上色后的深度图像
            detections: 检测结果列表
            positions: 与检测结果对应的3D位置列表
        """
        if detections and positions:
            # 在彩色图像上绘制检测结果
            for i, (det, pos) in enumerate(zip(detections, positions)):
                if det is None or pos is None:
                    continue
                x1, y1, x2, y2, conf, _ = map(int, det[:6])
                
                # 绘制边界框
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # 显示3D位置
                pos_text = f'({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m'
                cv2.putText(color_image, pos_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 水平堆叠图像
        images = np.hstack((color_image, depth_colormap))
        
        # 显示结果
        cv2.imshow('RealSense YOLO Cup Detection', images)
    
    def run(self):
        """
        检测和可视化的主循环。
        """
        try:
            while True:
                # 获取帧
                success, color_image, depth_image, depth_scale = self.get_frames()
                if not success or color_image is None or depth_image is None:
                    continue
                
                # 检测杯子
                detections = self.detect_cups(color_image)
                
                # 计算3D位置
                positions = []
                for det in detections:
                    pos = self.get_3d_position(det, depth_image, depth_scale)
                    positions.append(pos)
                    print(f"Cup detected at 3D position (x, y, z): {pos} meters")

                depth_colormap = get_depth_colormap(depth_image)
                
                # 可视化结果
                self.visualize(color_image.copy(), depth_colormap, detections, positions)
                
                # 按'q'键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # 停止视频流
            self.camera.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # 初始化检测器
    detector = RealSenseYOLOCupDetector(model_name='yolov8m', conf_threshold=0.5)
    
    # 运行检测
    detector.run()
