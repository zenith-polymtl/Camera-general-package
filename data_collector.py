import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')

        # Create output folder if it doesn't exist
        self.image_folder = 'captured_images'
        os.makedirs(self.image_folder, exist_ok=True)

        # Initialize CvBridge to convert ROS image messages to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)

        # Timer for controlling capture intervals
        self.capture_interval = 0.5  # Capture every 0.5 seconds
        self.capture_duration = 10  # Capture for 10 seconds
        self.image_count = 0
        self.start_time = None

    def start_capture(self):
        """Starts capturing images for a set duration."""
        self.start_time = time.time()  # Record the start time

        # Create a timer to capture images every 0.5 second
        self.timer = self.create_timer(self.capture_interval, self.capture_image)

    def capture_image(self):
        """Captures and saves the image."""
        # Check if the capture duration has elapsed
        if time.time() - self.start_time >= self.capture_duration:
            self.get_logger().info("Capture time finished. Stopping capture.")
            self.timer.cancel()  # Stop the timer
            rclpy.shutdown()  # Shutdown ROS2
            return

        # Save the captured image every 0.5 second
        if hasattr(self, 'cv_image') and self.cv_image is not None:
            # Add a timestamp to the filename to ensure uniqueness
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            filename = os.path.join(self.image_folder, f"image_{timestamp}_{self.image_count}.jpg")
            
            # Define save parameters: higher quality (95 is good)
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            cv2.imwrite(filename, self.cv_image, params)
            self.image_count += 1
            self.get_logger().info(f"Saved {filename}")
        else:
            self.get_logger().info("No image to save yet.")


    def image_callback(self, msg):
        """Updates the latest image from the camera."""
        # Convert the ROS image message to OpenCV format
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def main():
    rclpy.init()
    node = ImageCaptureNode()
    node.start_capture()  # Start the image capture process
    rclpy.spin(node)  # Keep the node running until capture is finished
    node.destroy_node()

if __name__ == '__main__':
    main()
