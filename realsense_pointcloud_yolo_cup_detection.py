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
3. 运行脚本：python realsense_cup_detection.py
4. 按'q'键退出程序

注意：
- 本示例使用COCO数据集预训练的YOLOv8模型，其中杯子的类别ID为41
- 3D位置计算基于深度相机的深度信息和相机内参
- 确保相机已正确安装并授予相应权限
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time
import open3d as o3d


class RealSenseYOLOCupDetector:
    def __init__(self, model_name='yolov8m', conf_threshold=0.5):
        """
        初始化RealSense相机和YOLO模型。
        
        参数:
            model_name (str): YOLO模型名称 (默认: 'yolov8m')
            conf_threshold (float): 检测的置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.cup_class_id = 41  # COCO数据集中'cup'类别的ID
        
        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 开始视频流
        self.profile = self.pipeline.start(config)
        
        # 获取深度传感器的深度比例
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 创建一个对齐对象，用于将深度帧与彩色帧对齐
        self.align = rs.align(rs.stream.color)
        
        # 加载YOLO模型
        self.model = YOLO(f'{model_name}.pt')  # 加载YOLOv8模型
        self.conf_threshold = conf_threshold
        
        # 创建点云处理对象
        self.pc = rs.pointcloud()
        # 创建颜色映射对象，用于可视化深度数据
        self.colorizer = rs.colorizer()
        
        # 创建点云可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Point Cloud', width=640, height=480)
        self.pcd = o3d.geometry.PointCloud()
        self.points_added = False
    
    def get_frames(self):
        """
        从RealSense获取对齐的彩色和深度帧，并生成点云。
        """
        # 等待获取一对连贯的帧
        frames = self.pipeline.wait_for_frames()
        
        # 将深度帧与彩色帧对齐
        aligned_frames = self.align.process(frames)
        
        # 获取对齐后的帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None, None
        
        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 为深度帧上色以便可视化
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        
        # 生成点云
        points = self.pc.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        vtx = np.array([[v[0], v[1], v[2]] for v in vtx], dtype=np.float32)
        
        return color_image, depth_image, depth_colormap, vtx
    
    def detect_cups(self, color_image):
        """
        使用YOLO在彩色图像中检测杯子。
        
        参数:
            color_image: 输入的彩色图像(BGR格式)
            
        返回:
            list: 检测到的杯子的边界框列表 [x1, y1, x2, y2, 置信度, 类别ID]
        """
        # 运行YOLO推理
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
    
    def get_3d_position(self, bbox, point_cloud, image_width=640, image_height=480):
        """
        使用点云数据计算边界框中心点在相机坐标系中的3D位置。
        
        参数:
            bbox: 边界框 [x1, y1, x2, y2, 置信度, 类别ID]
            point_cloud: 点云数据 (Nx3数组)
            image_width: 图像宽度
            image_height: 图像高度
            
        返回:
            tuple: 相机坐标系中的(x, y, z)坐标，单位为米
        """
        # 1. 确保边界框在图像范围内
        x1 = max(0, min(int(bbox[0]), image_width-1))
        y1 = max(0, min(int(bbox[1]), image_height-1))
        x2 = max(0, min(int(bbox[2]), image_width-1))
        y2 = max(0, min(int(bbox[3]), image_height-1))
        
        # 2. 计算中心点（使用浮点运算）
        center_x = int((x1 + x2) / 2.0)
        center_y = int((y1 + y2) / 2.0)
        
        # 3. 获取边界框内所有有效点
        valid_points = []
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                idx = y * image_width + x
                if idx < len(point_cloud):
                    point = point_cloud[idx]
                    # 过滤掉无效点（深度为0的点）
                    if not np.isnan(point[0]) and not np.isnan(point[1]) and not np.isnan(point[2]) and \
                       point[0] != 0 and point[1] != 0 and point[2] != 0:
                        valid_points.append(point)
        
        if not valid_points:
            return None
            
        # 4. 计算有效点的中心位置
        valid_points = np.array(valid_points)
        center_point = np.median(valid_points, axis=0)
        
        return tuple(center_point)
        
    def update_point_cloud_visualization(self, point_cloud, detections):
        """
        更新点云可视化。
        
        参数:
            point_cloud: 点云数据 (Nx3数组)
            detections: 检测结果列表
        """
        if len(point_cloud) == 0:
            return
            
        # 过滤掉无效点
        mask = ~np.isnan(point_cloud).any(axis=1) & (point_cloud != 0).all(axis=1)
        filtered_points = point_cloud[mask]
        
        # 更新点云数据
        self.pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # 为点云着色（这里使用统一的颜色）
        colors = np.ones((len(filtered_points), 3)) * 0.5  # 灰色
        
        # 高亮显示检测到的杯子区域
        for det in detections:
            if det is None:
                continue
                
            # 获取边界框内的点
            x1, y1, x2, y2, _, _ = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 计算边界框内点的索引
            indices = []
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    idx = y * 640 + x  # 假设图像宽度为640
                    if idx < len(mask) and mask[idx]:
                        indices.append(np.where(mask)[0] == idx)
            
            if indices:
                indices = np.where(np.any(np.array(indices), axis=0))[0]
                if len(indices) > 0:
                    # 为检测到的杯子区域设置不同颜色
                    colors[indices] = [1, 0, 0]  # 红色
        
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 更新可视化
        if not self.points_added:
            self.vis.add_geometry(self.pcd)
            self.points_added = True
        else:
            self.vis.update_geometry(self.pcd)
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def visualize(self, color_image, depth_colormap, detections, positions):
        """
        可视化检测结果。
        
        参数:
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
                # 获取帧和点云
                color_image, depth_image, depth_colormap, point_cloud = self.get_frames()
                if color_image is None or depth_image is None or point_cloud is None:
                    continue
                
                # 检测杯子
                detections = self.detect_cups(color_image)
                
                # 计算3D位置（使用点云）
                positions = []
                for det in detections:
                    pos = self.get_3d_position(det, point_cloud, color_image.shape[1], color_image.shape[0])
                    positions.append(pos)
                    if pos is not None:
                        print(f"检测到杯子，3D位置 (x, y, z): {pos} 米")
                
                # 更新点云可视化
                self.update_point_cloud_visualization(point_cloud, detections)
                
                # 可视化结果
                self.visualize(color_image.copy(), depth_colormap, detections, positions)
                
                # 按'q'键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # 停止视频流
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.vis.destroy_window()

if __name__ == "__main__":
    # 初始化检测器
    detector = RealSenseYOLOCupDetector(model_name='yolov8m', conf_threshold=0.5)
    
    # 运行检测
    detector.run()
