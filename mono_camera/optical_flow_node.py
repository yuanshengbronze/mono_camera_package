import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from marti_common_msgs.msg import Float32Stamped
from geometry_msgs.msg import Vector3Stamped
from collections import deque
from geometry_msgs.msg import Vector3

import cv2
import numpy as np

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')

        # === PARAMETERS ===
        # ros2 run mono_camera optical_flow_node --ros-args \
        #   -p clahe.clip_limit:=3.0 -p clahe.tile_x:=8 -p clahe.tile_y:=8
        
        # Scene / geometry
        self.declare_parameter('pool_depth', 2.0)

        # Feature detection (GFTT)
        self.declare_parameter('features.max', 350)
        self.declare_parameter('features.quality', 0.03)
        self.declare_parameter('features.min_distance', 12)
        self.declare_parameter('features.redetect_interval', 50)

        # LK optical flow
        self.declare_parameter('lk.win_size', 25)   # int, used as (win, win)  
        self.declare_parameter('lk.max_level', 2)
        self.declare_parameter('lk.max_error', 15.0)  # hard LK error cutoff

        # CLAHE
        self.declare_parameter('clahe.enable', True)
        self.declare_parameter('clahe.clip_limit', 2.0)
        self.declare_parameter('clahe.tile_x', 8)
        self.declare_parameter('clahe.tile_y', 8)

        # Outlier rejection
        self.declare_parameter('outlier.mad_threshold', 3.0)   # MAD multiplier
        self.declare_parameter('outlier.bright_threshold', 220) # pixel value to mask specular highlights
        self.declare_parameter('outlier.fb_threshold', 1.0)    # forward-backward error px

        # Yaw estimation
        self.declare_parameter('yaw.use_vision', True)           # use homography-based yaw
        self.declare_parameter('yaw.vision_alpha', 0.85)         # complementary filter weight for vision yaw
        self.declare_parameter('yaw.min_inliers', 8)             # minimum homography inliers

        # Velocity output
        self.declare_parameter('velocity.ema_alpha', 0.4)        # EMA smoothing for published velocity

        # === CONSTANTS ===
        self.POOL_DEPTH = float(self.get_parameter('pool_depth').value)
        self.FX = 522.94629
        self.FY = 525.422
        self.CX = 338.45741
        self.CY = 242.50987
        self.K = np.array([[self.FX, 0, self.CX],
                           [0, self.FY, self.CY],
                           [0,  0,  1]], dtype=np.float64)
        
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
        self.DIRECTION_SMOOTH = 0.8
        self.ALPHA = 0.3

        # Smoothed velocity output (EMA)
        self.ema_alpha = float(self.get_parameter('velocity.ema_alpha').value)
        self.ema_vx = 0.0
        self.ema_vy = 0.0

        # === CLAHE AND FEATURE TRACKING VARIABLES === 
        self.MAX_FEATURES = int(self.get_parameter('features.max').value)
        self.QUALITY_LEVEL = float(self.get_parameter('features.quality').value)
        self.MIN_DISTANCE = int(self.get_parameter('features.min_distance').value)
        self.REDETECT_INTERVAL = int(self.get_parameter('features.redetect_interval').value)

        self.CLAHE_ENABLE = bool(self.get_parameter('clahe.enable').value)
        self.clip = float(self.get_parameter('clahe.clip_limit').value)
        self.tx = int(self.get_parameter('clahe.tile_x').value)
        self.ty = int(self.get_parameter('clahe.tile_y').value)
        clip = max(0.1, self.clip)
        tx = max(1, self.tx)
        ty = max(1, self.ty)
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tx, ty))

        # Outlier rejection params
        self.MAD_THRESHOLD    = float(self.get_parameter('outlier.mad_threshold').value)
        self.BRIGHT_THRESHOLD = int(self.get_parameter('outlier.bright_threshold').value)
        self.FB_THRESHOLD     = float(self.get_parameter('outlier.fb_threshold').value)
        self.MAX_LK_ERROR     = float(self.get_parameter('lk.max_error').value)

        # Yaw estimation params
        self.USE_VISION_YAW   = bool(self.get_parameter('yaw.use_vision').value)
        self.YAW_VISION_ALPHA = float(self.get_parameter('yaw.vision_alpha').value)
        self.YAW_MIN_INLIERS  = int(self.get_parameter('yaw.min_inliers').value)
 
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
        self.speed_comp_pub = self.create_publisher(Vector3, '/optical_flow/speed_comp', 10)
        self.image_pub = self.create_publisher(Image, '/optical_flow/annotated_image', 10)

        # === OPTICAL FLOW STATE ===
        self.prev_gray = None
        self.p0 = None
        self.prev_dirs = None
        self.frame_idx = 0
        self.mask = None

        # === LK PARAMETERS ===
        self.lk_params = dict(
            winSize=(int(self.get_parameter('lk.win_size').value), int(self.get_parameter('lk.win_size').value)),
            maxLevel=int(self.get_parameter('lk.max_level').value),
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

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9
    
    def _nearest(self, buffer, t_ref, max_skew):
        if not buffer:
            return None
        best = min(buffer, key=lambda x: abs(x[0] - t_ref))
        if abs(best[0] - t_ref) > max_skew:
            return None
        return best

    def _bright_mask(self, gray):
        """Return a mask that excludes specular highlights (very bright pixels)."""
        _, bright = cv2.threshold(gray, self.BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_not(bright)

    # =========================================================================
    # VISION-BASED YAW ESTIMATION
    # =========================================================================

    def _estimate_yaw_rate_homography(self, pts_old, pts_new, dt):
        """
        Estimate yaw rate from the rotational component of the flow field by
        decomposing the inter-frame homography (valid for a planar scene).

        Returns w_yaw_vision (rad/s) or None if estimation fails.

        Convention: Z positive downward (camera facing the pool floor).
        For a downward-facing camera, yaw rotates around the Z axis.
        The returned value is positive for counter-clockwise rotation when
        viewed from above (i.e. w_yaw > 0 → vehicle turns left in NED/body).
        """
        if len(pts_old) < 4 or len(pts_new) < 4:
            return None

        H, inliers = cv2.findHomography(pts_old, pts_new, cv2.RANSAC, 3.0)
        if H is None:
            return None
        if inliers is None or int(inliers.sum()) < self.YAW_MIN_INLIERS:
            return None

        # Decompose homography into rotation / translation / plane normal.
        # cv2.decomposeHomographyMat returns up to 4 solution sets.
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, self.K)

        # Select the decomposition where the plane normal points *toward* the
        # camera, i.e. N_z < 0 in camera frame (floor is below the camera and
        # the normal of the floor points upward, which is -Z in camera coords).
        best_R = None
        for i in range(num):
            n = Ns[i].flatten()
            if n[2] < 0:          # normal pointing toward camera (up)
                best_R = Rs[i]
                break

        if best_R is None:
            # Fallback: pick solution with normal closest to -Z
            best_R = Rs[int(np.argmin([Ns[i][2] for i in range(num)]))]

        # Extract yaw (rotation about Z axis) from the rotation matrix.
        # For small angles: R ≈ [[cos θ, -sin θ, 0], [sin θ, cos θ, 0], [0, 0, 1]]
        yaw_delta = np.arctan2(best_R[1, 0], best_R[0, 0])
        w_yaw_vision = yaw_delta / dt
        return float(w_yaw_vision)

    def _estimate_yaw_rate_flow_curl(self, pts_old, pts_new, dt):
        """
        Lightweight fallback: estimate yaw rate from the curl (rotational
        component) of the optical flow field via least squares.

        For pure yaw w around Z, the induced normalised flow is:
            dx_norm = -w * y_norm * dt
            dy_norm =  w * x_norm * dt
        """
        A, b = [], []
        for new, old in zip(pts_new, pts_old):
            ax, ay = new.ravel()
            cx, cy = old.ravel()
            xnorm = (ax - self.CX) / self.FX
            ynorm = (ay - self.CY) / self.FY
            dx_norm = (ax - cx) / self.FX
            dy_norm = (ay - cy) / self.FY
            A.append([ xnorm * dt])   # from dy_norm =  w * xnorm * dt
            b.append(  dy_norm      )
            A.append([-ynorm * dt])   # from dx_norm = -w * ynorm * dt
            b.append(  dx_norm      )

        A = np.array(A)
        b = np.array(b)
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return float(result[0])

    def _fuse_yaw_rate(self, w_yaw_imu, pts_old, pts_new, dt):
        """
        Attempt homography-based yaw estimation and fuse with IMU yaw rate
        via a complementary filter.  Falls back to flow-curl if homography
        decomposition fails, and to IMU-only if everything else fails.
        """
        w_yaw_vision = self._estimate_yaw_rate_homography(pts_old, pts_new, dt)

        if w_yaw_vision is None and len(pts_old) >= 4:
            w_yaw_vision = self._estimate_yaw_rate_flow_curl(pts_old, pts_new, dt)

        if w_yaw_vision is None:
            return w_yaw_imu

        # Complementary filter: trust vision for slow/DC, IMU for fast transients
        alpha = self.YAW_VISION_ALPHA
        return alpha * w_yaw_vision + (1.0 - alpha) * w_yaw_imu

    # =========================================================================
    # CALLBACKS
    # =========================================================================
        
    def depth_callback(self, msg: Float32Stamped):
        z = float(msg.data)
        if z <= 0.0 or np.isnan(z) or np.isinf(z):
           return
        
        t = self._stamp_to_sec(msg.header.stamp)
        vz = 0.0
        if self.prev_depth_t is not None:
            dt = t - self.prev_depth_t
            if 1e-4 < dt < 1.0:
                # Convention: Z positive downward.
                # vz > 0 means vehicle moving downward (toward pool floor).
                vz = (z - self.prev_depth_m) / dt
        
        self.prev_depth_t = t
        self.prev_depth_m = z
        self.depth_buffer.append((t, z, vz))
    
    def rpy_callback(self, msg: Vector3Stamped):
        r = float(msg.vector.x)
        p = float(msg.vector.y)
        y = float(msg.vector.z)
        t = self._stamp_to_sec(msg.header.stamp)

        w_yaw = 0.0
        if self.prev_yaw_t is not None:
            dt = t - self.prev_yaw_t
            if 1e-4 < dt < 1.0:
                dyaw = (y - self.prev_yaw + np.pi) % (2 * np.pi) - np.pi
                w_yaw = dyaw / dt
        
        self.prev_roll  = r
        self.prev_pitch = p
        self.prev_yaw   = y
        self.prev_yaw_t = t
        self.rpy_buffer.append((t, r, p, y, w_yaw))

    def image_callback(self, msg):
        # --- Timing ---
        t = self._stamp_to_sec(msg.header.stamp)
        if self.last_img_t is None:
            self.last_img_t = t
            return
        dt = t - self.last_img_t
        self.last_img_t = t
        
        if dt <= 1e-4 or dt > 1.0:
            return
        
        # --- Sync depth and RPY to image timestamp ---
        depth_entry = self._nearest(self.depth_buffer, t, self.MAX_DEPTH_SKEW)
        rpy_entry   = self._nearest(self.rpy_buffer,   t, self.MAX_RPY_SKEW)
        if depth_entry is None or rpy_entry is None:
            return
        _, self.depth_m, vz      = depth_entry
        _, self.roll, self.pitch, self.yaw, w_yaw_imu = rpy_entry
        
        # --- Convert to grayscale + optional CLAHE ---
        frame      = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.CLAHE_ENABLE:
            frame_gray = self.clahe.apply(frame_gray)

        # --- Build mask that excludes specular highlights ---
        detect_mask = self._bright_mask(frame_gray)

        # --- First frame initialisation ---
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=detect_mask, **self.feature_params)
            if self.p0 is None:
                self.p0 = np.empty((0, 1, 2), dtype=np.float32)
            self.prev_dirs = np.zeros_like(self.p0)
            self.mask = np.zeros_like(frame)
            return

        # =====================================================================
        # OPTICAL FLOW  (forward pass)
        # =====================================================================
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.p0, None,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            **self.lk_params
        )
        if p1 is None or st is None or err is None:
            return

        # =====================================================================
        # FORWARD-BACKWARD CONSISTENCY CHECK
        # Discard points whose back-tracked position differs from the original
        # by more than FB_THRESHOLD pixels.
        # =====================================================================
        p_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            frame_gray, self.prev_gray, p1, None, **self.lk_params
        )
        if p_back is not None and st_back is not None:
            fb_error = np.linalg.norm(
                self.p0 - p_back, axis=2
            ).squeeze(axis=1)                       # shape (N,)
            fb_ok = (fb_error < self.FB_THRESHOLD)  # shape (N,)
        else:
            fb_ok = np.ones(len(self.p0), dtype=bool)

        # Combine LK status, hard LK error cutoff, and FB consistency
        lk_err_flat = err.flatten()
        combined_mask = (
            (st.flatten() == 1) &
            (lk_err_flat < self.MAX_LK_ERROR) &
            fb_ok
        )

        good_new  = p1[combined_mask]
        good_old  = self.p0[combined_mask]
        valid_err = lk_err_flat[combined_mask]

        # =====================================================================
        # YAW RATE  (fused IMU + vision)
        # =====================================================================
        if self.USE_VISION_YAW and len(good_old) >= 4:
            w_yaw = self._fuse_yaw_rate(w_yaw_imu, good_old, good_new, dt)
        else:
            w_yaw = w_yaw_imu

        # =====================================================================
        # PER-POINT VELOCITY COMPUTATION
        # =====================================================================
        stacked_metric_vel = []
        stacked_weights    = []

        # Fade the drawing mask
        self.mask = cv2.addWeighted(
            self.mask, self.ALPHA, np.zeros_like(self.mask), 1 - self.ALPHA, 0
        )

        # Carry over smoothed directions for surviving points
        old_prev_dirs = self.prev_dirs
        new_prev_dirs = np.zeros((len(good_new), 1, 2), dtype=np.float32)
        old_indices   = np.where(combined_mask)[0]
        for new_idx, old_idx in enumerate(old_indices):
            if old_idx < len(old_prev_dirs):
                new_prev_dirs[new_idx] = old_prev_dirs[old_idx]

        for idx, (new, old, error) in enumerate(zip(good_new, good_old, valid_err)):
            a, b = new.ravel()
            c, d = old.ravel()

            # Distance from camera to pool floor (Z positive downward).
            # depth_m = vehicle depth from surface; pool floor is at POOL_DEPTH.
            Z_m = self.POOL_DEPTH - self.depth_m
            if Z_m <= 0:
                continue

            # Pixel and normalised displacements
            dx_px  = a - c
            dy_px  = b - d
            xnorm  = (a - self.CX) / self.FX
            ynorm  = (b - self.CY) / self.FY
            dxnorm = dx_px / self.FX
            dynorm = dy_px / self.FY

            # 3-D position in camera frame
            X = xnorm * Z_m
            Y = ynorm * Z_m

            # AUV translational velocity in camera / body frame.
            # Optical flow equation (planar, Z+ downward):
            #   dxnorm/dt = -vx/Z  + xnorm*vz/Z  + w_yaw*ynorm  (approx, small angles)
            #   dynorm/dt = -vy/Z  + ynorm*vz/Z  - w_yaw*xnorm
            # Solving for vx, vy:
            vx = -dxnorm / dt * Z_m + xnorm * vz - w_yaw * Y
            vy = -dynorm / dt * Z_m + ynorm * vz + w_yaw * X

            mag = np.sqrt(vx**2 + vy**2)
            if mag < 0.05:
                continue

            # Smooth drawing direction
            if idx < len(new_prev_dirs):
                prev_dx_px, prev_dy_px = new_prev_dirs[idx][0]
                dx_px = (self.DIRECTION_SMOOTH * prev_dx_px +
                         (1 - self.DIRECTION_SMOOTH) * dx_px)
                dy_px = (self.DIRECTION_SMOOTH * prev_dy_px +
                         (1 - self.DIRECTION_SMOOTH) * dy_px)
                new_prev_dirs[idx] = [[dx_px, dy_px]]

            # Draw arrow
            end_x = int(c + 3 * dx_px)
            end_y = int(d + 3 * dy_px)
            cv2.arrowedLine(
                self.mask, (int(c), int(d)), (end_x, end_y),
                (0, 255, 0), 2, tipLength=0.3
            )

            stacked_metric_vel.append([vx, vy])
            weight = 1.0 / (error + 1e-6)
            stacked_weights.append(weight)

        self.prev_dirs = new_prev_dirs

        # --- Publish annotated frame ---
        output  = cv2.addWeighted(frame, 0.8, self.mask, 0.7, 0)
        msg_out = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        msg_out.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(msg_out)

        # =====================================================================
        # WEIGHTED LEAST-SQUARES + MAD OUTLIER REJECTION
        # =====================================================================
        if len(stacked_metric_vel) > 0:
            V = np.array(stacked_metric_vel)   # (N, 2)
            w = np.array(stacked_weights)       # (N,)

            # --- MAD-based outlier rejection ---
            if len(V) > 5:
                median_v  = np.median(V, axis=0)
                residuals = np.linalg.norm(V - median_v, axis=1)
                mad       = np.median(residuals) + 1e-6
                inlier_mask = residuals < self.MAD_THRESHOLD * mad
                V = V[inlier_mask]
                w = w[inlier_mask]

            if len(V) == 0:
                return

            # Weighted mean (closed-form WLS since all rows map to same unknowns)
            v_uav_x = np.sum(w * V[:, 0]) / np.sum(w)
            v_uav_y = np.sum(w * V[:, 1]) / np.sum(w)

            # --- EMA smoothing on output velocity ---
            a = self.ema_alpha
            self.ema_vx = a * v_uav_x + (1.0 - a) * self.ema_vx
            self.ema_vy = a * v_uav_y + (1.0 - a) * self.ema_vy

            avg_speed = np.sqrt(self.ema_vx**2 + self.ema_vy**2)

            self.get_logger().info(
                f"Speed: {avg_speed:.2f} m/s "
                f"(vx={self.ema_vx:.2f}, vy={self.ema_vy:.2f}, "
                f"w_yaw={w_yaw:.3f} rad/s)"
            )

            speed_msg      = Float32()
            speed_msg.data = float(avg_speed)
            self.speed_pub.publish(speed_msg)

            speed_comp_msg   = Vector3()
            speed_comp_msg.x = self.ema_vx
            speed_comp_msg.y = self.ema_vy
            speed_comp_msg.z = avg_speed
            self.speed_comp_pub.publish(speed_comp_msg)

        # =====================================================================
        # UPDATE STATE FOR NEXT FRAME
        # =====================================================================
        self.prev_gray = frame_gray.copy()
        self.p0        = good_new.reshape(-1, 1, 2)
        # Wrap frame counter to avoid overflow on long runs
        self.frame_idx = (self.frame_idx + 1) % self.REDETECT_INTERVAL

        # --- Periodic / low-count re-detection ---
        if self.frame_idx == 0 or len(self.p0) < 30:
            new_points = cv2.goodFeaturesToTrack(
                frame_gray, mask=detect_mask, **self.feature_params
            )
            if new_points is not None:
                self.p0        = np.vstack((self.p0, new_points))
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
