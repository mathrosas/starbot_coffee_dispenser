import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import pcl
import numpy as np
import tf2_ros
from tf2_ros import TransformException, ConnectivityException
from custom_msgs.msg import DetectedSurfaces, DetectedObjects
from typing import List, Tuple, Union

class ObjectDetection(Node):
    def __init__(self) -> None:
        super().__init__('cup_detection_node')  # like script 1
        # I/O to match script 1 + your RViz config
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera_depth_sensor/points', self.callback, 10)

        self.table_marker_pub = self.create_publisher(
            MarkerArray, '/table_marker', 10)  # plural for RViz
        self.objects_marker_pub = self.create_publisher(
            MarkerArray, '/cup_marker', 10)

        self.table_detected_pub = self.create_publisher(
            DetectedSurfaces, '/table_detected', 10)
        self.object_detected_pub = self.create_publisher(
            DetectedObjects, '/cup_detected', 10)

        self.marker_id = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def callback(self, msg: PointCloud2) -> None:
        try:
            cloud = self.from_ros_msg(msg)
            if cloud is None:
                return

            # ROI filters like script 1
            filtered_cloud_plane = self.filter_cloud(
                cloud, min_x_dist=-0.1, max_x_dist=0.4,
                min_y_dist=0.0, max_y_dist=0.75,
                min_height=-0.1, max_height=0.02)

            filtered_cloud_objects = self.filter_cloud(
                cloud, min_x_dist=-0.1, max_x_dist=0.4,
                min_y_dist=0.2, max_y_dist=0.7,
                min_height=0.0, max_height=0.1)

            # Plane segmentation
            plane_indices, plane_coefficients, plane_cloud = self.extract_plane(filtered_cloud_plane)

            # Clustering (tables/surfaces and objects)
            table_clusters, surface_centroids, surface_dimensions = self.extract_clusters(
                plane_cloud, "Table Surface")
            object_clusters, object_centroids, object_dimensions = self.extract_clusters(
                filtered_cloud_objects, "Cup")

            # Visuals
            self.pub_surface_marker(surface_centroids, surface_dimensions)
            self.pub_object_marker(object_centroids, object_dimensions)

            # Structured outputs
            self.pub_surface_detected(surface_centroids, surface_dimensions)
            self.pub_object_detected(object_centroids, object_dimensions)

        except (TransformException, ConnectivityException) as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def from_ros_msg(self, msg: PointCloud2) -> Union[pcl.PointCloud, None]:
        """Converts a ROS2 PointCloud2 message to a PCL point cloud (in base_link)."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', msg.header.frame_id,
                rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=1.0))

            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            rotation_quaternion = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])

            rotation_matrix = self.quaternion_to_rotation_matrix(rotation_quaternion)

            point_step = msg.point_step
            num_points = len(msg.data) // point_step
            points = []
            for i in range(num_points):
                start = i * point_step
                x = np.frombuffer(msg.data[start:start+4], dtype=np.float32)[0]
                y = np.frombuffer(msg.data[start+4:start+8], dtype=np.float32)[0]
                z = np.frombuffer(msg.data[start+8:start+12], dtype=np.float32)[0]
                p = np.array([x, y, z])
                p_rot = np.dot(rotation_matrix, p)
                p_rel = p_rot + translation
                points.append(p_rel)

            data = np.array(points, dtype=np.float32)
            assert data.shape[1] == 3, "Number of fields must be 3"
            cloud = pcl.PointCloud()
            cloud.from_array(data)
            return cloud

        except (TransformException, ConnectivityException) as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in from_ros_msg: {e}")
            return None

    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
            [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
            [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
        ])

    def filter_cloud(
        self, cloud: pcl.PointCloud,
        min_x_dist: float, max_x_dist: float,
        min_y_dist: float, max_y_dist: float,
        min_height: float, max_height: float
    ) -> Union[pcl.PointCloud, None]:
        """Axis-aligned ROI filter like script 1."""
        try:
            indices = []
            for i in range(cloud.size):
                px, py, pz = cloud[i]
                if (min_x_dist <= px <= max_x_dist and
                    min_y_dist <= py <= max_y_dist and
                    min_height <= pz <= max_height):
                    indices.append(i)
            return cloud.extract(indices)
        except Exception as e:
            self.get_logger().error(f"Error in filter_cloud: {e}")
            return None

    def extract_plane(self, cloud: pcl.PointCloud) -> Tuple[np.ndarray, np.ndarray, pcl.PointCloud]:
        """RANSAC plane extraction (params aligned with script 1)."""
        seg = cloud.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.01)
        indices, coefficients = seg.segment()
        plane_cloud = cloud.extract(indices)
        return indices, coefficients, plane_cloud

    def extract_clusters(
        self, cloud: pcl.PointCloud, cluster_type: str
    ) -> Tuple[List[pcl.PointCloud], List[List[float]], List[List[float]]]:
        """Euclidean clustering; larger max size like script 1."""
        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.02)
        ec.set_MinClusterSize(100)
        ec.set_MaxClusterSize(200000)
        ec.set_SearchMethod(tree)

        cluster_indices = ec.Extract()
        clusters, centroids, dims = [], [], []

        for indices in cluster_indices:
            cluster = cloud.extract(indices)
            centroid = np.mean(cluster, axis=0)
            min_coords = np.min(cluster, axis=0)
            max_coords = np.max(cluster, axis=0)
            dimensions = max_coords - min_coords

            clusters.append(cluster)
            centroids.append(centroid.tolist())
            dims.append(dimensions.tolist())

        if not clusters:
            self.get_logger().warning(f"No {cluster_type} clusters extracted...")
        return clusters, centroids, dims

    def pub_surface_marker(self, surface_centroids: List[List[float]], surface_dimensions: List[List[float]]) -> None:
        """Publish table surfaces as green cubes (like script 1)."""
        marker_array = MarkerArray()
        surface_thickness = 0.05

        for idx, (c, d) in enumerate(zip(surface_centroids, surface_dimensions)):
            length = float(d[0])
            width  = float(d[1])

            m = Marker()
            m.header.frame_id = "base_link"
            m.id = idx
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(c[0])
            m.pose.position.y = float(c[1])
            m.pose.position.z = float(c[2]) - surface_thickness / 2.0
            m.pose.orientation.w = 1.0

            m.scale.x = length
            m.scale.y = width
            m.scale.z = surface_thickness

            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.5
            marker_array.markers.append(m)

        if marker_array.markers:
            self.table_marker_pub.publish(marker_array)
        else:
            self.get_logger().warning("No table plane markers to publish.")

    def pub_surface_detected(self, centroids: List[List[float]], dimensions: List[List[float]]) -> None:
        """Publish structured surface detections (custom_msgs)."""
        for idx, (c, d) in enumerate(zip(centroids, dimensions)):
            msg = DetectedSurfaces()
            msg.surface_id = idx
            msg.position.x = float(c[0])
            msg.position.y = float(c[1])
            msg.position.z = float(c[2])
            msg.height = float(d[0])
            msg.width  = float(d[1])
            self.table_detected_pub.publish(msg)

    def pub_object_marker(self, object_centroids: List[List[float]], object_dimensions: List[List[float]]) -> None:
        """Publish objects as red CYLINDERs with small offsets (like script 1)."""
        marker_array = MarkerArray()
        for idx, (c, d) in enumerate(zip(object_centroids, object_dimensions)):
            diameter = float(max(d[0], d[1]))

            m = Marker()
            m.header.frame_id = "base_link"
            m.id = idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position.x = float(c[0]) - diameter / 4.0
            m.pose.position.y = float(c[1]) - 0.016
            m.pose.position.z = float(c[2])
            m.pose.orientation.w = 1.0

            m.scale.x = diameter
            m.scale.y = diameter
            m.scale.z = float(d[2])

            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.5
            marker_array.markers.append(m)

        if marker_array.markers:
            self.objects_marker_pub.publish(marker_array)
        else:
            self.get_logger().warning("No objects on flat surface markers to publish.")

    def pub_object_detected(self, centroids: List[List[float]], dimensions: List[List[float]]) -> None:
        """Publish structured object detections (cylindrical semantics)."""
        for idx, (c, d) in enumerate(zip(centroids, dimensions)):
            diameter = float(max(d[0], d[1]))
            msg = DetectedObjects()
            msg.object_id = idx
            msg.position.x = float(c[0]) - diameter / 4.0
            msg.position.y = float(c[1]) - 0.016
            msg.position.z = float(c[2])
            msg.width = diameter
            msg.thickness = diameter
            msg.height = float(d[2])
            self.object_detected_pub.publish(msg)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectDetection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
