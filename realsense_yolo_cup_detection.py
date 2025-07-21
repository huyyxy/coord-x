import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

class RealSenseYOLOCupDetector:
    def __init__(self, model_name='yolov8m', conf_threshold=0.5):
        """
        Initialize the RealSense camera and YOLO model.
        
        Args:
            model_name (str): YOLO model name (default: 'yolov8m')
            conf_threshold (float): Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.cup_class_id = 41  # COCO class ID for 'cup'
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start streaming
        self.profile = self.pipeline.start(config)
        
        # Get depth sensor's depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Create an align object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
        # Load YOLO model
        self.model = YOLO(f'{model_name}.pt')  # Loads official YOLOv8 model
        self.conf_threshold = conf_threshold
        
        # Create a colorizer object to visualize depth data
        self.colorizer = rs.colorizer()
    
    def get_frames(self):
        """
        Get aligned color and depth frames from RealSense.
        """
        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Colorize depth frame for visualization
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        
        return color_image, depth_image, depth_colormap
    
    def detect_cups(self, color_image):
        """
        Detect cups in the color image using YOLO.
        
        Args:
            color_image: Input color image (BGR format)
            
        Returns:
            list: List of bounding boxes [x1, y1, x2, y2, conf, class_id] for detected cups
        """
        # Run YOLO inference (Ultralytics YOLO expects BGR by default)
        results = self.model(color_image, conf=self.conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Combine boxes, confidences, and class IDs
            for box, conf, class_id in zip(boxes, confs, class_ids):
                if class_id == self.cup_class_id:
                    detections.append([*box, conf, class_id])
        
        return detections
    
    def get_3d_position(self, bbox, depth_image):
        """
        Calculate the 3D position of the center of a bounding box in camera coordinates.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2, conf, class_id]
            depth_image: Depth image from RealSense
            
        Returns:
            tuple: (x, y, z) coordinates in meters in camera coordinate system
        """
        # Calculate center of the bounding box
        x1, y1, x2, y2 = map(int, bbox[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get depth at the center point
        depth = depth_image[center_y, center_x] * self.depth_scale  # Convert to meters
        
        # Get camera intrinsics
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        
        # Convert from pixel coordinates to camera coordinates
        # Note: This is a simplified version that doesn't account for lens distortion
        x = (center_x - intrinsics.ppx) / intrinsics.fx * depth
        y = (center_y - intrinsics.ppy) / intrinsics.fy * depth
        z = depth
        
        return (x, y, z)
    
    def visualize(self, color_image, depth_colormap, detections, positions):
        """
        Visualize the results.
        
        Args:
            color_image: Color image (BGR)
            depth_colormap: Colorized depth image
            detections: List of detections
            positions: List of 3D positions corresponding to detections
        """
        # Draw detections on color image
        for i, (det, pos) in enumerate(zip(detections, positions)):
            x1, y1, x2, y2, conf, _ = map(int, det[:6])
            
            # Draw bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Display 3D position
            pos_text = f'({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m'
            cv2.putText(color_image, pos_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Stack images horizontally
        images = np.hstack((color_image, depth_colormap))
        
        # Display the result
        cv2.imshow('RealSense YOLO Cup Detection', images)
    
    def run(self):
        """
        Main loop for detection and visualization.
        """
        try:
            while True:
                # Get frames
                color_image, depth_image, depth_colormap = self.get_frames()
                if color_image is None or depth_image is None:
                    continue
                
                # Detect cups
                detections = self.detect_cups(color_image)
                
                # Calculate 3D positions
                positions = []
                for det in detections:
                    pos = self.get_3d_position(det, depth_image)
                    positions.append(pos)
                    print(f"Cup detected at 3D position (x, y, z): {pos} meters")
                
                # Visualize results
                self.visualize(color_image.copy(), depth_colormap, detections, positions)
                
                # Break the loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize the detector
    detector = RealSenseYOLOCupDetector(model_name='yolov8m', conf_threshold=0.5)
    
    # Run the detection
    detector.run()
