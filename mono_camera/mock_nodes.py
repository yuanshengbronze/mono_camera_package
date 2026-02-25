import math
import rclpy
from rclpy.node import Node
from marti_common_msgs.msg import Float32Stamped
from geometry_msgs.msg import Vector3Stamped


class MockDepthPublisher(Node):
    """
    Publishes marti_common_msgs/Float32Stamped on /depth.

    depth_m here is interpreted by your optical flow node as:
      Z_m = POOL_DEPTH - depth_m
    so keep depth_m < POOL_DEPTH to avoid Z_m <= 0.
    """
    def __init__(self):
        super().__init__('mock_depth_publisher')

        # Parameters
        self.declare_parameter('topic', '/depth')
        self.declare_parameter('hz', 30.0)
        self.declare_parameter('mean_depth_m', 1.0)     # must be < pool_depth (2.0 in your node)
        self.declare_parameter('amplitude_m', 0.05)     # small oscillation
        self.declare_parameter('period_s', 10.0)        # slow wave
        self.declare_parameter('frame_id', 'depth_link')

        self.topic = str(self.get_parameter('topic').value)
        self.hz = float(self.get_parameter('hz').value)
        self.mean_depth = float(self.get_parameter('mean_depth_m').value)
        self.amp = float(self.get_parameter('amplitude_m').value)
        self.period = float(self.get_parameter('period_s').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        if self.hz <= 0:
            raise ValueError("hz must be > 0")
        if self.period <= 0:
            raise ValueError("period_s must be > 0")

        self.pub = self.create_publisher(Float32Stamped, self.topic, 10)
        self.t0 = self.get_clock().now()

        self.timer = self.create_timer(1.0 / self.hz, self._tick)
        self.get_logger().info(
            f"Publishing mock depth on {self.topic} at {self.hz:.1f} Hz "
            f"(mean={self.mean_depth:.3f} m, amp={self.amp:.3f} m, period={self.period:.1f} s)"
        )

    def _tick(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        # Smooth depth signal
        depth = self.mean_depth + self.amp * math.sin(2.0 * math.pi * t / self.period)

        msg = Float32Stamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame_id
        msg.data = float(depth)

        self.pub.publish(msg)


class MockRPYPublisher(Node):
    """
    Publishes geometry_msgs/Vector3Stamped on /rpy.
    vector.x = roll (rad), vector.y = pitch (rad), vector.z = yaw (rad)
    """
    def __init__(self):
        super().__init__('mock_rpy_publisher')

        # Parameters
        self.declare_parameter('topic', '/rpy')
        self.declare_parameter('hz', 30.0)
        self.declare_parameter('roll_rad', 0.0)
        self.declare_parameter('pitch_rad', 0.0)
        self.declare_parameter('yaw_rate_rad_s', 0.05)  # constant yaw rate
        self.declare_parameter('frame_id', 'base_link')

        self.topic = str(self.get_parameter('topic').value)
        self.hz = float(self.get_parameter('hz').value)
        self.roll = float(self.get_parameter('roll_rad').value)
        self.pitch = float(self.get_parameter('pitch_rad').value)
        self.yaw_rate = float(self.get_parameter('yaw_rate_rad_s').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        if self.hz <= 0:
            raise ValueError("hz must be > 0")

        self.pub = self.create_publisher(Vector3Stamped, self.topic, 10)
        self.t0 = self.get_clock().now()

        self.timer = self.create_timer(1.0 / self.hz, self._tick)
        self.get_logger().info(
            f"Publishing mock RPY on {self.topic} at {self.hz:.1f} Hz "
            f"(roll={self.roll:.3f}, pitch={self.pitch:.3f}, yaw_rate={self.yaw_rate:.3f} rad/s)"
        )

    def _tick(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        yaw = self.yaw_rate * t
        # Wrap yaw into [-pi, pi] to match your dyaw wrapping logic
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi

        msg = Vector3Stamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.frame_id
        msg.vector.x = float(self.roll)
        msg.vector.y = float(self.pitch)
        msg.vector.z = float(yaw)

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    depth_node = MockDepthPublisher()
    rpy_node = MockRPYPublisher()

    # Spin both nodes in one process
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(depth_node)
    executor.add_node(rpy_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(depth_node)
        executor.remove_node(rpy_node)
        depth_node.destroy_node()
        rpy_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()