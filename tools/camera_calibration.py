"""
相机标定工具

使用说明:
1. 打印一个棋盘格标定板（推荐使用8x6的棋盘格，每个方格的尺寸为2.5cm x 2.5cm）
2. 将标定板放在不同位置和角度，让程序采集多张图片（建议15-20张）
3. 按空格键保存当前帧，按'q'键完成标定
"""
import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional, Tuple

class CameraCalibrator:
    def __init__(self, board_size=(8, 6), square_size=0.025):
        """
        初始化相机标定器
        
        参数:
            board_size: 棋盘格内部角点数量 (width, height)
            square_size: 每个方格的实际大小（单位：米）
        """
        self.board_size = board_size
        self.square_size = square_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 准备对象点，如 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
        
        # 存储对象点和图像点的数组
        self.objpoints = []  # 3D点（世界坐标系）
        self.imgpoints = []  # 2D点（图像平面）
        
        # 创建保存标定图像的目录
        self.calib_dir = "calibration_images"
        os.makedirs(self.calib_dir, exist_ok=True)
        
    def find_corners(self, img: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        在图像中查找并优化棋盘格角点位置
        
        参数:
            img: 输入的BGR彩色图像，形状为(H, W, 3)
            
        返回:
            tuple[bool, Optional[np.ndarray]]: 
                - 第一个元素(bool): 是否成功检测到棋盘格角点
                - 第二个元素: 如果检测成功，返回优化后的角点坐标数组，形状为(N, 1, 2)，
                  其中N是角点数量；如果检测失败，返回None
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        
        if ret:
            # 提高角点检测精度
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            return True, corners2
        return False, None
    
    def add_calibration_image(self, img: np.ndarray, corners: np.ndarray) -> None:
        """
        添加标定图像及其角点数据到标定数据集
        
        参数:
            img: 包含棋盘格的BGR彩色图像，形状为(H, W, 3)
            corners: 检测到的棋盘格角点坐标数组，形状为(N, 1, 2)，
                    其中N是角点数量，通常为board_size[0] * board_size[1]
            
        返回:
            None
            
        副作用:
            - 将3D对象点(self.objp)添加到objpoints列表
            - 将2D图像点(corners)添加到imgpoints列表
            - 将图像保存到calibration_images目录下
        """
        self.objpoints.append(self.objp)
        self.imgpoints.append(corners)
        
        # 保存标定图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.calib_dir, f"calib_{timestamp}.jpg")
        cv2.imwrite(filename, img)
        print(f"已保存标定图像: {filename}")
    
    def calibrate(self, img_size: tuple[int, int]) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        执行相机标定计算
        
        参数:
            img_size: 图像尺寸，格式为(宽度, 高度)
            
        返回:
            tuple: 包含三个元素的元组:
                - camera_matrix (np.ndarray | None): 3x3相机内参矩阵，格式为:
                    [[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,   1]]
                      其中(fx, fy)是焦距，(cx, cy)是主点坐标
                      
                - dist_coeffs (np.ndarray | None): 畸变系数，格式为[k1, k2, p1, p2, k3, ...]
                  - k1, k2, k3: 径向畸变系数
                  - p1, p2: 切向畸变系数
                  
                - mean_error (float | None): 平均重投影误差（像素）
                
                如果标定失败（如图像数量不足），则返回(None, None, None)
        """
        if len(self.objpoints) < 5:
            print("错误：需要至少5张标定图像")
            return None, None, None
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None
        )
        
        # 计算重投影误差
        mean_error = 0.0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(self.objpoints)
        print(f"重投影误差: {mean_error:.8f} 像素")
        print("\n相机内参矩阵:")
        print(mtx)
        print("\n畸变系数 (k1, k2, p1, p2, k3, ...):")
        print(dist[0])
        
        return mtx, dist[0], mean_error

def main():
    # 初始化标定器 (8x6 棋盘格，每个方格2.5cm x 2.5cm)
    calibrator = CameraCalibrator(board_size=(8, 6), square_size=0.025)
    
    # 打开相机 (0 通常是内置摄像头)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n相机标定程序")
    print("1. 准备一个8x6的棋盘格标定板")
    print("2. 将棋盘格放在相机前不同位置和角度")
    print("3. 按空格键保存当前帧，按'q'键完成标定")
    print("4. 建议保存15-20张不同角度的图像")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取相机画面")
            break
            
        # 查找棋盘格角点
        ret_corners, corners = calibrator.find_corners(frame)
        
        # 如果找到角点，绘制出来
        if ret_corners:
            cv2.drawChessboardCorners(frame, calibrator.board_size, corners, ret_corners)
        
        # 显示已保存的图像数量
        cv2.putText(frame, f"Saved: {len(calibrator.objpoints)}/20", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display help text
        cv2.putText(frame, "Press SPACE to save, 'q' to finish", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Camera Calibration (SPACE to save, 'q' to finish)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键保存当前帧
            if ret_corners:
                calibrator.add_calibration_image(frame.copy(), corners)
                if len(calibrator.objpoints) >= 20:
                    print("已保存20张图像，可以按'q'键完成标定")
            else:
                print("未检测到完整的棋盘格，请调整位置后重试")
        elif key == ord('q'):  # 'q'键退出
            break
    
    # 执行标定
    if len(calibrator.objpoints) >= 5:
        print("\n正在计算相机参数...")
        mtx, dist, error = calibrator.calibrate((frame.shape[1], frame.shape[0]))
        
        if mtx is not None:
            # 保存相机参数到文件
            np.savez("camera_params.npz", 
                    camera_matrix=mtx, 
                    dist_coeffs=dist,
                    reprojection_error=error)
            print("\n相机参数已保存到 camera_params.npz")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n标定完成！")

if __name__ == "__main__":
    main()
