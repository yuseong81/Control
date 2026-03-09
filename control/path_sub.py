import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import pandas as pd
import numpy as np
import os

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher_node')
        
        # 1. 경로 설정 (생성한 csv 파일 경로)
        # self.declare_parameter('path_file', '/home/yuseong/ros2_ws/ACCA_2025/src/ACCA_2024/erp42/erp42_control/erp42_control/reverse_path.csv')
        self.declare_parameter('path_file', '/home/yuseong/ros2_ws/ACCA_2025/src/ACCA_2024/erp42/erp42_control/erp42_control/complex_path.csv')
        # self.declare_parameter('path_file', '/home/yuseong/ros2_ws/ACCA_2025/src/ACCA_2024/erp42/erp42_control/erp42_control/path.csv')
        csv_path = self.get_parameter('path_file').get_parameter_value().string_value

        # 2. Publisher 설정
        self.path_pub = self.create_publisher(Path, '/local_path', 10)
        
        # 3. 데이터 로드
        if os.path.exists(csv_path):
            self.raw_data = pd.read_csv(csv_path)
            self.get_logger().info(f"Successfully loaded path from {csv_path}")
        else:
            self.get_logger().error(f"File not found: {csv_path}")
            return

        # 4. 타이머 설정 (10Hz로 경로 반복 발행)
        self.timer = self.create_timer(0.1, self.publish_path)

    def euler_to_quaternion(self, yaw):
        """Yaw 값을 Quaternion으로 변환"""
        return {
            'x': 0.0,
            'y': 0.0,
            'z': np.sin(yaw / 2.0),
            'w': np.cos(yaw / 2.0)
        }

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"  # IMM.py의 odom 프레임과 맞춰야 함
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for _, row in self.raw_data.iterrows():
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(row['x'])
            pose.pose.position.y = float(row['y'])
            pose.pose.position.z = 0.0
            
            # Yaw를 Quaternion으로 변환하여 채움
            q = self.euler_to_quaternion(row['yaw'])
            pose.pose.orientation.x = q['x']
            pose.pose.orientation.y = q['y']
            pose.pose.orientation.z = q['z']
            pose.pose.orientation.w = q['w']
            
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()