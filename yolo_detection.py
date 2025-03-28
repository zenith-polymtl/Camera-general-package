import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO  # Import YOLO
import os




class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',  
            self.image_callback,
            10)
        self.bridge = CvBridge()

        # Load the trained YOLOv8 model
        self.model = YOLO("/home/haipy/ardu_ws/src/src/my_camera_pkg/my_camera_pkg/my_model.pt")   

    def image_callback(self, msg):
        try:
            # Convert ROS 2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Perform YOLO detection
        results = self.model(cv_image)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red rectangle
                cv2.putText(cv_image, f"{self.model.names[int(box.cls)]} {float(box.conf):.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the image
        cv2.imshow("YOLOv8 Detection", cv_image)
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
