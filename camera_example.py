"""
相机模块使用示例
"""
import cv2
import sys
from camera import OpenCVCamera, RealSenseCamera


WIDTH = 1280
HEIGHT = 720
FPS = 30

def test_opencv_camera():
    """测试OpenCV相机实现"""
    print("Testing OpenCV camera...")
    
    with OpenCVCamera(width=WIDTH, height=HEIGHT, fps=FPS) as cam:
        print(f"Camera properties: {cam.get_properties()}")
        
        while True:
            success, frame = cam.read()
            if not success:
                print("Failed to read frame")
                break
                
            # 显示帧
            cv2.imshow('OpenCV Camera', frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

def test_realsense_camera():
    """测试RealSense相机实现"""
    print("Testing RealSense camera...")
    
    try:
        with RealSenseCamera(width=WIDTH, height=HEIGHT, fps=FPS, enable_depth=True, enable_color=True) as cam:
            print(f"Camera properties: {cam.get_properties()}")
            
            while True:
                success, frames = cam.read()
                if not success or 'color' not in frames:
                    print("Failed to read frame")
                    break
                    
                # 显示彩色帧
                cv2.imshow('RealSense Color', frames['color'])
                
                # 如果可用则显示深度帧
                if 'depth' in frames:
                    # 对深度图进行归一化以便可视化
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(frames['depth'], alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('RealSense Depth', depth_colormap)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"Error with RealSense camera: {e}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Camera Module Example")
    print("1. 测试OpenCV相机")
    print("2. 测试RealSense相机")
    
    choice = input("请输入您的选择 (1 或 2): ")
    
    if choice == '1':
        test_opencv_camera()
    elif choice == '2':
        test_realsense_camera()
    else:
        print("无效选择。请重新运行并输入1或2。")
        sys.exit(1)
