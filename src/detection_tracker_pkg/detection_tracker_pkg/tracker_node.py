# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, TwistStamped
from std_msgs.msg import ColorRGBA, Float64, Float64MultiArray, MultiArrayDimension
from rclpy.time import Time
import numpy as np
import struct
import sys
import os

"""
ROS2 Radar Point Classification Node

/radarFrame -> orientation -> velocity estimate -> bounds filter ->
static/dynamic classification -> per-frame DBSCAN -> CV Kalman filter.

Dynamic points are red, static points are blue.
"""

candidate_paths = [
    "/home/deep/Downloads/deep_ws/src/detection",
    "/home/deep/mavros_ws/src/detection",
    "/home/deep/src/detection",
    "/home/deep/mmwave_ws/src/detection",
]
detection_path = ''
for path in candidate_paths:
    if path and os.path.exists(path):
        detection_path = path
        break

if not detection_path:
    raise RuntimeError(
        'Could not locate detection modules. Checked: '
        + ', '.join([p for p in candidate_paths if p])
    )

if detection_path not in sys.path:
    sys.path.insert(0, detection_path)

print(f"[radar_classifier_node] Using detection modules from: {detection_path}")

try:
    from var_init import *
    from adjust_orientation import adjust_orientation
    from estimate_vel import estimate_vel
    from process_det import process_det
    from create_det import create_det
    from classify_det import classify_det
    from cluster_points import cluster_and_drop_noise, cluster_stats, CLUSTER_STATS_FIELDS
    from kalman_cv import CVTracker, KF_STATE_FIELDS
except ImportError as e:
    print(f"ERROR: Could not import detection modules from {detection_path}")
    print(f"Error: {e}")
    raise


class DetectionTrackerNode(Node):

    # Warm hues for dynamic, cool for static, so the two stay tellable apart
    DYNAMIC_CLUSTER_COLORS = [
        (1.00, 0.15, 0.15), (1.00, 0.55, 0.00), (1.00, 0.85, 0.10),
        (1.00, 0.30, 0.65), (0.85, 0.20, 0.40), (1.00, 0.70, 0.45),
    ]
    STATIC_CLUSTER_COLORS = [
        (0.15, 0.35, 1.00), (0.00, 0.75, 0.90), (0.20, 0.85, 0.50),
        (0.55, 0.40, 0.95), (0.10, 0.55, 0.65), (0.45, 0.70, 1.00),
    ]
    # Indexed by track id, which unlike a DBSCAN label is stable across frames
    TRACK_COLORS = [
        (1.00, 0.20, 0.20), (0.20, 1.00, 0.35), (0.30, 0.55, 1.00),
        (1.00, 0.80, 0.10), (0.90, 0.35, 1.00), (0.10, 0.90, 0.90),
    ]

    def __init__(self):
        super().__init__('radar_classifier_node')
        self.declare_parameter('radial_velocity_noise_floor', 0.3)
        self.declare_parameter('classified_point_size', 0.2)
        self.declare_parameter('velocity_arrow_seconds', 0.5)  # arrow length = |v| * this
        self.declare_parameter('kf_sphere_size', 0.3)
        self.radial_vel_noise_floor = self.get_parameter('radial_velocity_noise_floor').value
        self.classified_point_size = float(self.get_parameter('classified_point_size').value)
        self.velocity_arrow_seconds = float(self.get_parameter('velocity_arrow_seconds').value)
        self.kf_sphere_size = float(self.get_parameter('kf_sphere_size').value)

        self.frame_num = 0

        self.classified_points_pub = self.create_publisher(MarkerArray, 'classified_points', 10)
        self.static_clusters_pub = self.create_publisher(MarkerArray, 'static_clusters', 10)
        self.dynamic_clusters_pub = self.create_publisher(MarkerArray, 'dynamic_clusters', 10)
        self.static_cluster_state_pub = self.create_publisher(
            Float64MultiArray, 'static_cluster_state', 10)
        self.dynamic_cluster_state_pub = self.create_publisher(
            Float64MultiArray, 'dynamic_cluster_state', 10)
        self.kf_tracks_pub = self.create_publisher(MarkerArray, 'kf_tracks', 10)
        self.kf_track_state_pub = self.create_publisher(
            Float64MultiArray, 'kf_track_state', 10)
        self.latency_pub = self.create_publisher(Float64, 'latency/processing', 10)

        # Dynamic clusters only; static is not filtered
        self.tracker = CVTracker(kf_params)
        self.prev_frame_time = None

        self.ego_vel_map = np.zeros(3)   # ENU velocity from MAVROS
        self.ego_yaw = 0.0
        self.ego_vel_radar = None

        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import PoseStamped
        self.ego_vel_sub = self.create_subscription(
            TwistStamped,
            '/mavros/local_position/velocity_local',
            self._ego_vel_callback,
            qos_profile_sensor_data)
        self.ego_pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self._ego_pose_callback,
            qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            PointCloud2,
            '/radarFrame',
            self.pointcloud_callback,
            10)

        self.get_logger().info('Radar point classification node started')
        self.get_logger().info('Publishing classified_points (dynamic=red, static=blue)')

    def _ego_vel_callback(self, msg):
        self.ego_vel_map = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z
        ])
        self._update_ego_vel_radar()

    def _ego_pose_callback(self, msg):
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.ego_yaw = np.arctan2(siny_cosp, cosy_cosp)
        self._update_ego_vel_radar()

    def _update_ego_vel_radar(self):
        """ENU ego velocity -> radar frame (X=right, Y=forward, Z=up)."""
        cy, sy = np.cos(self.ego_yaw), np.sin(self.ego_yaw)
        vx_radar = sy * self.ego_vel_map[0] - cy * self.ego_vel_map[1]
        vy_radar = cy * self.ego_vel_map[0] + sy * self.ego_vel_map[1]
        vz_radar = self.ego_vel_map[2]
        self.ego_vel_radar = np.array([vx_radar, vy_radar, vz_radar])

    def extract_points(self, msg):
        """PointCloud2 -> Nx4 [x, y, z, radial_velocity]."""
        points = []
        point_step = msg.point_step
        for i in range(msg.width):
            offset = i * point_step
            x, y, z, vel = struct.unpack_from('ffff', msg.data, offset)
            points.append([x, y, z, vel])
        return np.array(points)

    def publish_classified_points(self, static_meas, dynamic_meas):
        """One POINTS marker per class: dynamic red, static blue."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        specs = (
            ('static_points', 0, static_meas, ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)),
            ('dynamic_points', 1, dynamic_meas, ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)),
        )

        for ns, marker_id, meas, color in specs:
            marker = Marker()
            marker.header.frame_id = "radar"
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.POINTS
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.classified_point_size  # POINTS uses x/y as width/height
            marker.scale.y = self.classified_point_size
            marker.color = color
            marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

            pts = np.asarray(meas) if meas is not None else np.empty((0, 3))
            if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 3:
                marker.action = Marker.ADD
                for row in pts:
                    marker.points.append(
                        Point(x=float(row[0]), y=float(row[1]), z=float(row[2]))
                    )
            else:
                marker.action = Marker.DELETE

            marker_array.markers.append(marker)

        self.classified_points_pub.publish(marker_array)

    def publish_clusters(self, points, labels, publisher, ns_prefix, palette):
        """One POINTS marker per cluster plus a velocity arrow, coloured by label."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Leading DELETEALL so a frame with fewer clusters leaves nothing stale
        clear = Marker()
        clear.header.frame_id = "radar"
        clear.header.stamp = stamp
        clear.ns = ns_prefix
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        points = np.asarray(points)
        labels = np.asarray(labels)

        if points.ndim == 2 and points.shape[0] > 0 and labels.size == points.shape[0]:
            for marker_id, label in enumerate(np.unique(labels)):
                cluster_pts = points[labels == label]
                r, g, b = palette[int(label) % len(palette)]

                marker = Marker()
                marker.header.frame_id = "radar"
                marker.header.stamp = stamp
                marker.ns = ns_prefix
                marker.id = marker_id
                marker.type = Marker.POINTS
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.classified_point_size
                marker.scale.y = self.classified_point_size
                marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
                marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                for row in cluster_pts:
                    marker.points.append(
                        Point(x=float(row[0]), y=float(row[1]), z=float(row[2]))
                    )
                marker_array.markers.append(marker)

            for marker_id, row in enumerate(cluster_stats(points, labels)):
                centroid = row[1:4]
                velocity = row[4:7]
                if float(np.linalg.norm(velocity)) < 1e-3:
                    continue

                r, g, b = palette[int(row[0]) % len(palette)]
                tip = centroid + velocity * self.velocity_arrow_seconds

                arrow = Marker()
                arrow.header.frame_id = "radar"
                arrow.header.stamp = stamp
                arrow.ns = ns_prefix + '_velocity'
                arrow.id = marker_id
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.orientation.w = 1.0
                arrow.scale.x = 0.05   # shaft
                arrow.scale.y = 0.12   # head diameter
                arrow.scale.z = 0.15   # head length
                arrow.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
                arrow.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                arrow.points.append(
                    Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])))
                arrow.points.append(
                    Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])))
                marker_array.markers.append(arrow)

        publisher.publish(marker_array)

    def publish_kf_tracks(self, tracks):
        """Filtered state as sphere + velocity arrow + id label per track."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for ns in ('kf_tracks', 'kf_tracks_velocity', 'kf_tracks_label'):
            clear = Marker()
            clear.header.frame_id = "radar"
            clear.header.stamp = stamp
            clear.ns = ns
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)

        for track in tracks:
            pos = track.position
            r, g, b = self.TRACK_COLORS[track.id % len(self.TRACK_COLORS)]

            sphere = Marker()
            sphere.header.frame_id = "radar"
            sphere.header.stamp = stamp
            sphere.ns = 'kf_tracks'
            sphere.id = track.id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(pos[0])
            sphere.pose.position.y = float(pos[1])
            sphere.pose.position.z = float(pos[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = self.kf_sphere_size
            sphere.color = ColorRGBA(r=r, g=g, b=b, a=0.85)
            sphere.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(sphere)

            if track.speed >= 1e-3:
                tip = pos + track.velocity * self.velocity_arrow_seconds
                arrow = Marker()
                arrow.header.frame_id = "radar"
                arrow.header.stamp = stamp
                arrow.ns = 'kf_tracks_velocity'
                arrow.id = track.id
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.orientation.w = 1.0
                arrow.scale.x = 0.05
                arrow.scale.y = 0.12
                arrow.scale.z = 0.15
                arrow.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
                arrow.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                arrow.points.append(Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
                arrow.points.append(Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2])))
                marker_array.markers.append(arrow)

            text = Marker()
            text.header.frame_id = "radar"
            text.header.stamp = stamp
            text.ns = 'kf_tracks_label'
            text.id = track.id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(pos[0])
            text.pose.position.y = float(pos[1])
            text.pose.position.z = float(pos[2]) + 0.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = f'ID{track.id} {track.speed:.2f}m/s'
            text.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(text)

        self.kf_tracks_pub.publish(marker_array)

    def _publish_rows(self, rows, fields, row_label, publisher):
        """Flatten an MxN array into a Float64MultiArray with a labelled layout."""
        msg = Float64MultiArray()
        n_rows, n_cols = rows.shape

        row_dim = MultiArrayDimension()
        row_dim.label = row_label
        row_dim.size = n_rows
        row_dim.stride = n_rows * n_cols
        col_dim = MultiArrayDimension()
        col_dim.label = ','.join(fields)
        col_dim.size = n_cols
        col_dim.stride = n_cols
        msg.layout.dim = [row_dim, col_dim]

        msg.data = rows.flatten().tolist()
        publisher.publish(msg)

    def publish_kf_state(self, tracks):
        self._publish_rows(self.tracker.state_rows(tracks), KF_STATE_FIELDS,
                           'tracks', self.kf_track_state_pub)

    def publish_cluster_state(self, points, labels, publisher):
        stats = cluster_stats(points, labels) if points is not None else np.zeros((0, 8))
        self._publish_rows(stats, CLUSTER_STATS_FIELDS, 'clusters', publisher)

    def _step_tracker(self, stats, frame_time_sec):
        """Advance the filter and publish. Called on empty frames too, so
        live tracks coast instead of dying on a single gap."""
        dt = (frame_time_sec - self.prev_frame_time) \
            if self.prev_frame_time is not None else 0.0
        self.prev_frame_time = frame_time_sec
        confirmed = self.tracker.step(stats, dt)
        self.publish_kf_tracks(confirmed)
        self.publish_kf_state(confirmed)
        return confirmed, dt

    def pointcloud_callback(self, msg):
        try:
            points = self.extract_points(msg)

            if len(points) == 0:
                return

            self.frame_num += 1
            msg_stamp = msg.header.stamp
            msg_time = Time.from_msg(msg_stamp)
            frame_time_sec = float(msg_stamp.sec) + float(msg_stamp.nanosec) * 1e-9
            if frame_time_sec <= 0.0:
                frame_time_sec = float(self.frame_num)
                msg_time = self.get_clock().now()
            frame_time_us = int(round(frame_time_sec * 1e6))

            x, y, z = adjust_orientation(elev_tilt, az_tilt, sensorHeight,
                                         points[:, 0], points[:, 1], points[:, 2])

            # Live radar reports small non-zero velocity for static objects
            radial_vel = points[:, 3].copy()
            radial_vel[np.abs(radial_vel) < self.radial_vel_noise_floor] = 0.0

            points_4col = np.column_stack([x, y, z, radial_vel])

            # Remove the drone's own radial velocity contribution
            if self.ego_vel_radar is not None and np.linalg.norm(self.ego_vel_radar) > 0.05:
                pts_xyz = points_4col[:, :3]
                norms = np.linalg.norm(pts_xyz, axis=1, keepdims=True)
                los = pts_xyz / np.maximum(norms, 1e-6)
                points_4col[:, 3] -= (los * self.ego_vel_radar).sum(axis=1)

            points_oriented = estimate_vel(points_4col)

            if self.frame_num % 50 == 1:
                raw_vel = points[:, 3]
                vel_mags = np.linalg.norm(points_oriented[:, 3:6], axis=1)
                self.get_logger().info(
                    f'[DIAG] Frame {self.frame_num}: {len(points)} pts | '
                    f'raw vel!=0: {np.count_nonzero(raw_vel)} '
                    f'(max|vel|={np.max(np.abs(raw_vel)):.3f}) | '
                    f'after noise floor: {np.count_nonzero(radial_vel)} | '
                    f'estimate_vel |v|>0: {np.count_nonzero(vel_mags)}'
                )

            processed_dets = process_det(points_oriented, frame_time_us,
                                         xLowerBound, xHigerBound,
                                         yLowerBound, yHigerBound,
                                         zLowerBound, zHigerBound)

            if len(processed_dets) == 0:
                self.publish_classified_points(None, None)
                self.publish_clusters(None, None, self.static_clusters_pub,
                                      'static_clusters', self.STATIC_CLUSTER_COLORS)
                self.publish_clusters(None, None, self.dynamic_clusters_pub,
                                      'dynamic_clusters', self.DYNAMIC_CLUSTER_COLORS)
                self.publish_cluster_state(None, None, self.static_cluster_state_pub)
                self.publish_cluster_state(None, None, self.dynamic_cluster_state_pub)
                self._step_tracker(np.zeros((0, 8)), frame_time_sec)
                return

            staticmeas, dynamicmeas = classify_det(create_det(processed_dets))

            static_clustered, static_labels, n_static_clusters = \
                cluster_and_drop_noise(staticmeas, clusterer_params)
            dynamic_clustered, dynamic_labels, n_dynamic_clusters = \
                cluster_and_drop_noise(dynamicmeas, clusterer_params)

            self.publish_classified_points(static_clustered, dynamic_clustered)
            self.publish_clusters(static_clustered, static_labels,
                                  self.static_clusters_pub, 'static_clusters',
                                  self.STATIC_CLUSTER_COLORS)
            self.publish_clusters(dynamic_clustered, dynamic_labels,
                                  self.dynamic_clusters_pub, 'dynamic_clusters',
                                  self.DYNAMIC_CLUSTER_COLORS)
            self.publish_cluster_state(static_clustered, static_labels,
                                       self.static_cluster_state_pub)
            self.publish_cluster_state(dynamic_clustered, dynamic_labels,
                                       self.dynamic_cluster_state_pub)

            confirmed, dt = self._step_tracker(
                cluster_stats(dynamic_clustered, dynamic_labels), frame_time_sec)

            if self.frame_num % 50 == 1:
                self.get_logger().info(
                    f'[DIAG] classify: {len(staticmeas)} static, {len(dynamicmeas)} dynamic | '
                    f'dbscan: {len(static_clustered)} pts/{n_static_clusters} cl static, '
                    f'{len(dynamic_clustered)} pts/{n_dynamic_clusters} cl dynamic'
                )
                self.get_logger().info(
                    f'[DIAG] kf: {len(confirmed)} confirmed / {len(self.tracker.tracks)} '
                    f'tracks, dt={dt:.3f}s, mean NIS={self.tracker.mean_nis():.2f} '
                    f'(target ~6.0)'
                )

            try:
                latency = max(0.0, (self.get_clock().now() - msg_time).nanoseconds * 1e-9)
                latency_msg = Float64()
                latency_msg.data = latency
                self.latency_pub.publish(latency_msg)
            except Exception as exc:
                self.get_logger().warn(f'Failed to publish latency sample: {exc}')

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = DetectionTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
