#!/usr/bin/env python3

import can
import sys
import time
from threading import Thread, Lock
from queue import Queue, Empty
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class CANWinchNode(Node):
    """ROS2 node for controlling a winch via CAN bus with reliability improvements."""

    def __init__(self):
        super().__init__('can_winch_node')
        
        # Configuration parameters
        self.declare_parameters(namespace='',
            parameters=[
                ('device', '/dev/ttyUSB0'),
                ('can_speed', 500000),
                ('baudrate', 2000000),
                ('motor_id', 1),
                ('movement_time', 2.0),
                ('keep_alive_interval', 0.3),  # 300ms between keep-alives
                ('bus_recovery_interval', 2.0),  # Check bus health every 2s
                ('max_reconnect_attempts', 5)
            ])

        # Get parameters
        self.device = self.get_parameter('device').value
        self.can_speed = self.get_parameter('can_speed').value
        self.baudrate = self.get_parameter('baudrate').value
        self.motor_id = self.get_parameter('motor_id').value
        self.movement_time = self.get_parameter('movement_time').value
        self.keep_alive_interval = self.get_parameter('keep_alive_interval').value
        self.bus_recovery_interval = self.get_parameter('bus_recovery_interval').value
        self.max_reconnect_attempts = self.get_parameter('max_reconnect_attempts').value

        # State management
        self.bus = None
        self.bus_lock = Lock()
        self.reconnect_attempts = 0
        self.response_queue = Queue()
        self.last_command = None
        self.current_operation = None
        self.operation_step = 0
        self.operation_data = {}
        self.last_command_time = 0
        self.receiver_thread = None
        self.keep_alive_active = False

        # Initialize system
        self.setup_can_bus()
        self.setup_timers()
        self.setup_subscriptions()

        self.get_logger().info("CAN Winch Node initialized")

    def setup_subscriptions(self):
        """Initialize ROS2 subscriptions with QoS settings."""
        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.go_sub = self.create_subscription(
            String, '/go_winch', self.go_callback, reliable_qos,
            callback_group=MutuallyExclusiveCallbackGroup())

    def setup_timers(self):
        """Initialize all timers with proper callback groups."""
        # Health check timer
        self.create_timer(
            self.bus_recovery_interval,
            self.check_bus_health,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        # Keep-alive timer
        self.create_timer(
            self.keep_alive_interval,
            self.send_keep_alive,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

    def setup_can_bus(self):
        """Initialize or reinitialize CAN bus connection."""
        with self.bus_lock:
            if self.bus:
                try:
                    self.bus.shutdown()
                except Exception as e:
                    self.get_logger().error(f"Error shutting down bus: {e}")
                self.bus = None

            try:
                self.bus = can.interface.Bus(
                    interface='seeedstudio',
                    channel=self.device,
                    bitrate=self.can_speed,
                    baudrate=self.baudrate
                )
                self.reconnect_attempts = 0
                self.get_logger().info(f"Connected to CAN bus on {self.device}")
                self.start_receiver_thread()
            except Exception as e:
                self.bus = None
                self.get_logger().error(f"CAN bus initialization failed: {e}")
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.get_logger().info("Scheduling reconnection attempt...")
                    self.reconnect_attempts += 1
                    self.create_timer(5.0, self.setup_can_bus)  # Retry after 5s

    def start_receiver_thread(self):
        """Start or restart the CAN message receiver thread."""
        if self.receiver_thread and self.receiver_thread.is_alive():
            return

        self.receiver_thread = Thread(target=self.receive_messages, daemon=True)
        self.receiver_thread.start()
        self.get_logger().info("CAN receiver thread started")

    def check_bus_health(self):
        """Periodically check and maintain bus connectivity."""
        if not self.bus or not self.receiver_thread.is_alive():
            self.get_logger().warning("CAN bus unhealthy, attempting recovery...")
            self.setup_can_bus()

    def send_keep_alive(self):
        """Maintain bus activity with empty messages when idle."""
        if self.current_operation is None and self.bus:
            try:
                empty_msg = can.Message(
                    arbitration_id=0x00,
                    data=[0x00]*8,
                    is_extended_id=False
                )
                self.bus.send(empty_msg)
                self.get_logger().debug("Sent keep-alive message", 
                                      throttle_duration_sec=10)
            except Exception as e:
                self.get_logger().error(f"Keep-alive failed: {e}")
                self.bus = None

    def receive_messages(self):
        """Continuous CAN message receiver with bus health monitoring."""
        self.get_logger().info("CAN receiver thread started")
        while rclpy.ok():
            if not self.bus:
                time.sleep(0.1)
                continue

            try:
                msg = self.bus.recv(0.1)
                if msg:
                    self.get_logger().debug(
                        f"RX id={msg.arbitration_id:#x} data={self.format_can_data(msg.data)}"
                    )
                    if self.last_command and msg.data[:2] == self.last_command:
                        self.response_queue.put(msg)
            except can.CanError as e:
                self.get_logger().error(f"CAN receive error: {e}")
                self.bus = None
            except Exception as e:
                self.get_logger().error(f"Receiver error: {e}")
                time.sleep(0.5)

    # ... (Keep existing message handling and command methods from original code) ...

    def send_message(self, data, arbitration_id=None):
        """Enhanced message sending with bus health checks."""
        if not self.bus:
            self.get_logger().error("Send failed: No active CAN bus")
            return False

        try:
            with self.bus_lock:
                msg = can.Message(
                    arbitration_id=arbitration_id or self.motor_id,
                    data=data,
                    is_extended_id=False
                )
                self.bus.send(msg)
                self.last_command = data[:2]
                self.last_command_time = time.time()
                self.get_logger().info(f"Sent message: {self.format_can_data(data)}")
                return True
        except can.CanError as e:
            self.get_logger().error(f"CAN send error: {e}")
            self.bus = None
            return False
        except Exception as e:
            self.get_logger().error(f"Unexpected send error: {e}")
            return False

    def go_callback(self, msg):
        """Command handler with bus availability check."""
        if not self.bus:
            self.get_logger().error("Ignoring command: No active CAN bus")
            return

        # ... (Keep existing command handling logic) ...

    def shutdown(self):
        """Comprehensive shutdown procedure."""
        self.get_logger().info("Shutting down node...")
        if self.bus:
            with self.bus_lock:
                try:
                    self.bus.shutdown()
                except Exception as e:
                    self.get_logger().error(f"Bus shutdown error: {e}")
        self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CANWinchNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()