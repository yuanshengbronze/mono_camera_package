import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from marti_common_msgs.msg import Float32Stamped
from geometry_msgs.msg import Vector3Stamped
from collections import deque

import cv2
import numpy as np

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')
        
        # === CONSTANTS ===
        self.POOL_DEPTH = 2 # in m
        self.FX = 522.94629
        self.FY = 525.422
        self.CX = 338.45741
        self.CY = 242.50987
        
        # === VARIABLES ===
        self.last_img_t = None
        self.prev_depth_t = None
        self.prev_depth_m = None
        self.prev_yaw_t = None
        self.prev_yaw = None
        self.depth_m = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.vz = 0.0
        self.w_yaw = 0.0

        # === PARAMETERS ===
        self.MAX_FEATURES = 350
        self.QUALITY_LEVEL = 0.03
        self.MIN_DISTANCE = 12
        self.DIRECTION_SMOOTH = 0.8
        self.ALPHA = 0.3
        self.REDETECT_INTERVAL = 50

        # === OPENCV BRIDGE ===
        self.bridge = CvBridge()

        # === SUBSCRIBERS ===
        self.image_sub = self.create_subscription(
            Image,
            '/image_rect',
            self.image_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Float32Stamped,
            '/depth',
            self.depth_callback,
            10
        )

        self.rpy_sub = self.create_subscription(
            Vector3Stamped,
            '/rpy',
            self.rpy_callback,
            10
        )

        # === PUBLISHERS ===
        self.speed_pub = self.create_publisher(Float32, '/optical_flow/speed', 10)
        self.image_pub = self.create_publisher(Image, '/optical_flow/annotated_image', 10)

        # === OPTICAL FLOW STATE ===
        self.prev_gray = None
        self.p0 = None
        self.prev_dirs = None
        self.frame_idx = 0
        self.mask = None

        # === LK PARAMETERS ===
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # === FEATURE TRACKING PARAMETERS ===
        self.feature_params = dict(
            maxCorners=self.MAX_FEATURES,
            qualityLevel=self.QUALITY_LEVEL,
            minDistance=self.MIN_DISTANCE,
            blockSize=7
        )

        # === Time Buffers ===
        self.depth_buffer = deque(maxlen=30)
        self.rpy_buffer = deque(maxlen=30)
        self.MAX_DEPTH_SKEW = 0.08
        self.MAX_RPY_SKEW = 0.08

        self.get_logger().info("Optical Flow Node started.")

    def _stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9
    
    def _nearest(self, buffer, t_ref, max_skew):
        if not buffer:
            return None
        best = min(buffer, key=lambda x: abs(x[0] - t_ref))
        if abs(best[0] - t_ref) > max_skew:
            return None
        return best
        
    def depth_callback(self, msg: Float32Stamped):
        z = float(msg.data)
        if z <= 0.0 or np.isnan(z) or np.isinf(z):
           return
        
        t = self._stamp_to_sec(msg.header.stamp)
        vz = 0.0 # Initialize vz
        if self.prev_depth_t is not None:
            dt = t - self.prev_depth_t
            if 1e-4 < dt < 1.0:
                vz = -(z - self.prev_depth_m) / dt
        
        self.prev_depth_t = t
        self.prev_depth_m = z
            
        self.depth_buffer.append((t, z, vz))
    
    def rpy_callback(self, msg: Vector3Stamped):
        r = float(msg.vector.x)
        p = float(msg.vector.y)
        y = float(msg.vector.z)
        t = self._stamp_to_sec(msg.header.stamp)

        w_yaw = 0.0 # Initialize w_yaw
        if self.prev_yaw_t is not None:
            dt = t - self.prev_yaw_t
            if 1e-4 < dt < 1.0:
                dyaw = (y - self.prev_yaw + np.pi) % (2 * np.pi) - np.pi
                w_yaw = dyaw / dt
        
        self.prev_roll = r
        self.prev_pitch = p
        self.prev_yaw = y
        self.prev_yaw_t = t
            
        self.rpy_buffer.append((t, r, p, y, w_yaw))

    def image_callback(self, msg):
        # Timing (based on img)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_img_t is None:
            self.last_img_t = t
            return
        dt = t - self.last_img_t
        self.last_img_t = t
        
        if dt <= 1e-4 or dt > 1.0:
            return
        
        # Get depth and rpy
        depth_entry = self._nearest(self.depth_buffer, t, self.MAX_DEPTH_SKEW)
        rpy_entry = self._nearest(self.rpy_buffer, t, self.MAX_RPY_SKEW)
        if depth_entry is None or rpy_entry is None:
            return
        _, self.depth_m, vz = depth_entry
        _, self.roll, self.pitch, self.yaw, w_yaw = rpy_entry
        
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize first frame
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            if self.p0 is None:
                self.p0 = np.empty((0, 1, 2), dtype=np.float32)
            self.prev_dirs = np.zeros_like(self.p0)
            self.mask = np.zeros_like(frame)
            return

        # --- Calculate optical flow ---
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.p0, None, **self.lk_params)
        if p1 is None or st is None or err is None:
            return

        good_new = p1[st == 1]
        good_old = self.p0[st == 1]
        valid_err = err[st == 1].flatten()

        # Least-Squares Matrix
        stacked_metric_vel = []
        stacked_weights = []

        # --- Reset mask fade ---
        self.mask = cv2.addWeighted(self.mask, self.ALPHA, np.zeros_like(self.mask), 1 - self.ALPHA, 0)

        # Update prev_dirs to match the number of good points
        # This needs to happen BEFORE the loop
        old_prev_dirs = self.prev_dirs
        new_prev_dirs = np.zeros((len(good_new), 1, 2), dtype=np.float32)
        
        # Map old directions to new positions where points were successfully tracked
        old_indices = np.where(st == 1)[0]
        for new_idx, old_idx in enumerate(old_indices):
            if old_idx < len(old_prev_dirs):
                new_prev_dirs[new_idx] = old_prev_dirs[old_idx]
        # --- Process each tracked point ---
        for idx, (new, old, error) in enumerate(zip(good_new, good_old, valid_err)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            Z_m = self.POOL_DEPTH - self.depth_m
            if Z_m <= 0:
                continue
            
            # PIXEL COORDINATES
            dx_px = a - c
            dy_px = b - d

            # NORMALIZED COORDINATES
            xnorm, ynorm = (a - self.CX) / self.FX, (b - self.CY) / self.FY 
            dxnorm, dynorm = dx_px / self.FX, dy_px / self.FY 

            # CAMERA COORDINATES
            X = xnorm * Z_m
            Y = ynorm * Z_m

            # AUV VELOCITY
            vx = -dxnorm / dt * Z_m + xnorm * vz - w_yaw * Y
            vy = -dynorm / dt * Z_m + ynorm * vz + w_yaw * X

            mag = np.sqrt(vx**2 + vy**2)
            if mag < 0.05:
                continue

            # Smooth direction
            if idx < len(new_prev_dirs):
                prev_dx_px, prev_dy_px = new_prev_dirs[idx][0]
                dx_px = self.DIRECTION_SMOOTH * prev_dx_px + (1 - self.DIRECTION_SMOOTH) * dx_px
                dy_px = self.DIRECTION_SMOOTH * prev_dy_px + (1 - self.DIRECTION_SMOOTH) * dy_px
                new_prev_dirs[idx] = [[dx_px, dy_px]]

            # Draw arrow
            end_x, end_y = int(c + 3 * dx_px), int(d + 3 * dy_px)
            cv2.arrowedLine(self.mask, (int(c), int(d)), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)

            stacked_metric_vel.append([vx, vy])
            weight = 1.0 / (error + 1e-6)
            stacked_weights.append(weight)

        self.prev_dirs = new_prev_dirs

        # --- Blend and publish annotated frame ---
        output = cv2.addWeighted(frame, 0.8, self.mask, 0.7, 0)
        msg_out = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        msg_out.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(msg_out)

        # Use Least-Squares
        if len(stacked_metric_vel) > 0:
            V = np.array(stacked_metric_vel)  # N x 2 (vx, vy for each point)
            w = np.diag(stacked_weights)  # N x N weight matrix
            
            # Set up least-squares problem: V = A * [v_uav_x, v_uav_y]^T
            # Since all points should have the same UAV velocity, we solve:
            # Minimize: ||W^(1/2) * (V - 1 * v_uav)||^2
            # where v_uav is the 2D UAV velocity we're estimating
            
            # Weighted least-squares solution
            v_uav_x = np.sum(w * V[:, 0]) / np.sum(w)
            v_uav_y = np.sum(w * V[:, 1]) / np.sum(w)
            
            # Calculate magnitude
            avg_speed = np.sqrt(v_uav_x**2 + v_uav_y**2)
            self.get_logger().info(f"Speed: {avg_speed:.2f} m/s (vx={v_uav_x:.2f}, vy={v_uav_y:.2f})")
            speed_msg = Float32()
            speed_msg.data = float(avg_speed) # in m/s
            self.speed_pub.publish(speed_msg)

        # --- Update for next frame ---
        self.prev_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        self.frame_idx += 1

        # --- Re-detect features if needed ---
        if self.frame_idx % self.REDETECT_INTERVAL == 0 or len(self.p0) < 30:
            new_points = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            if new_points is not None:
                self.p0 = np.vstack((self.p0, new_points))
                self.prev_dirs = np.zeros_like(self.p0)

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
