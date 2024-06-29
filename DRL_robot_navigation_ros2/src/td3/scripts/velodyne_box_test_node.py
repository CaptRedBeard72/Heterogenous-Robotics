#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import matplotlib.pyplot as plt

class VelodyneTestNode(Node):
    def __init__(self):
        super().__init__('velodyne_test_node')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            10
        )
        self.get_logger().info("Velodyne test node initialized")

    def velodyne_callback(self, msg):
        points = list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")))
        self.visualize_points(points)

    def visualize_points(self, points):
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        z = [point[2] for point in points]

        plt.figure(figsize=(10, 7))
        plt.scatter(x, y, c=z, cmap='viridis')
        plt.colorbar(label='Height (m)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Velodyne Point Cloud')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = VelodyneTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()