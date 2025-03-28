import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLO


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',  
            self.image_callback,
            10)
        self.bridge = CvBridge()

        
        self.model = YOLO("/home/haipy/ardu_ws/src/src/my_camera_pkg/my_camera_pkg/my_model_v8n.pt")

        # Real bucket dimensions (in meters)
        self.bucket_diameter = 1.2 # 31 cm ; sur gazebo ~1.2.m
        self.bucket_height = 1.42    # 38 cm ;sur gazebo 1.42m

        # 3D model points representing the bucket's base and top (in meters)
        self.object_points = np.array([
            [-self.bucket_diameter / 2, 0, 0],  # Bottom-left point of the base
            [self.bucket_diameter / 2, 0, 0],   # Bottom-right point of the base
            [-self.bucket_diameter / 2, 0, self.bucket_height],  # Top-left point
            [self.bucket_diameter / 2, 0, self.bucket_height]    # Top-right point
        ], dtype=np.float32)

        # Camera intrinsics (gazebo)
        self.camera_matrix = np.array([
            [205.47, 0, 320],
            [0, 205.47, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def image_callback(self, msg):
        try:
            # Convert ROS 2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        
        results = self.model(cv_image)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                # 2D image points (pixel coordinates of the detected bounding box corners)
                image_points = np.array([
                    [x1, y2],  # Bottom-left
                    [x2, y2],  # Bottom-right
                    [x1, y1],  # Top-left
                    [x2, y1]   # Top-right
                ], dtype=np.float32)

                # Apply PnP to estimate rotation and translation vectors
                success, rvec, tvec = cv2.solvePnP(self.object_points, image_points, self.camera_matrix, self.dist_coeffs)

                if success:
                    # Extract the translation vector (relative position of the bucket)
                    x, y, z = tvec.flatten()
                    position_text = f"Position: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m"
                    cv2.putText(cv_image, position_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw the bounding box and label
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(cv_image, f"{self.model.names[int(box.cls)]} {float(box.conf):.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the image 
        cv2.imshow("YOLOv8 Detection with PnP", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
