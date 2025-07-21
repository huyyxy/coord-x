"""
相机工厂使用示例

这个示例展示了如何使用 CameraFactory 来创建和管理不同类型的相机实例。
"""
import cv2
import argparse
from camera import CameraFactory

def display_camera_feed(camera, window_name):
    """
    显示相机画面
    
    参数:
        camera: 相机实例
        window_name: 窗口名称
    """
    print(f"\n显示 {window_name} 画面 (按 'q' 键退出)...")
    print(f"相机属性: {camera.get_properties()}")
    
    try:
        while True:
            # 读取帧
            success, frame = camera.read()
            if not success:
                print("无法读取帧")
                break
                
            # 对于 RealSense，frame 是一个字典
            if isinstance(frame, dict):
                if 'color' in frame:
                    cv2.imshow(window_name, frame['color'])
                if 'depth' in frame:
                    # 将深度图转换为可视化的灰度图
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(frame['depth'], alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow(f"{window_name} - 深度", depth_colormap)
            else:
                # 对于 OpenCV 相机，直接显示帧
                cv2.imshow(window_name, frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        camera.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相机工厂使用示例')
    parser.add_argument('--type', type=str, default='opencv',
                       choices=['opencv', 'realsense'],
                       help='相机类型 (opencv 或 realsense)')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='相机ID (对于OpenCV相机)')
    parser.add_argument('--width', type=int, default=1280,
                       help='图像宽度')
    parser.add_argument('--height', type=int, default=720,
                       help='图像高度')
    parser.add_argument('--fps', type=int, default=30,
                       help='帧率')
    args = parser.parse_args()
    
    try:
        # 1. 使用工厂创建相机实例
        print(f"正在创建 {args.type} 相机...")
        camera = CameraFactory.create_camera(
            camera_type=args.type,
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
            enable_depth=(args.type == 'realsense'),  # 仅为 RealSense 启用深度
            enable_color=True
        )
        
        # 2. 显示相机信息
        print("\n=== 相机信息 ===")
        print(f"类型: {args.type}")
        print(f"分辨率: {args.width}x{args.height}")
        print(f"帧率: {args.fps}")
        
        # 3. 显示相机画面
        display_camera_feed(camera, f"Camera - {args.type.upper()}")
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        print("程序结束")

if __name__ == "__main__":
    main()
