#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler


class PathPublisher:

    def __init__(self):
        # 1. ROS 노드 초기화
        rospy.init_node("path_publisher_node", anonymous=True)

        # 2. 퍼블리셔 설정 (Topic 명: /global_path, queue_size는 1이면 충분합니다)
        # latch=True로 설정하면 새로 켜지는 RViz가 이전에 발행된 Path를 바로 받아볼 수 있습니다.
        self.path_pub = rospy.Publisher(
            "/global_path", Path, queue_size=1, latch=True
        )

        # 3. CSV 파일 로드 (스크립트와 같은 디렉토리에 있는 csv 파일 읽기)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "path_topic.csv")

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"  # RViz 고정 프레임 기준

        if self.load_csv_and_build_msg(csv_path):
            rospy.loginfo("CSV 파일을 성공적으로 파싱하여 Path 메시지를 생성했습니다.")
        else:
            rospy.logerr(
                "CSV 파일을 읽는데 실패했습니다. 경로를 확인해주세요: %s",
                csv_path,
            )
            return

        # 4. 주기적 발행을 위한 타이머 설정 (1Hz, 1초에 한 번씩 발행)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

    def load_csv_and_build_msg(self, file_path):
        if not os.path.exists(file_path):
            return False

        with open(file_path, mode="r") as file:
            reader = csv.reader(file)
            header = next(reader)  # 헤더(x, y, yaw) 건너뛰기

            for row in reader:
                if not row:
                    continue
                x_val = float(row[0])
                y_val = float(row[1])
                yaw_val = float(row[2])

                # 개별 점을 위한 PoseStamped 메시지 생성
                pose = PoseStamped()
                pose.header.frame_id = "map"

                # 위치(Position) 대입
                pose.pose.position.x = x_val
                pose.pose.position.y = y_val
                pose.pose.position.z = 0.0

                # 방향(Orientation) 대입: Euler(Yaw) -> Quaternion 변환
                # Roll=0, Pitch=0, Yaw=yaw_val
                q = quaternion_from_euler(0.0, 0.0, yaw_val)
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]

                # 전체 경로 리스트에 추가
                self.path_msg.poses.append(pose)

        return True

    def timer_callback(self, event):
        # 발행할 때마다 타임스탬프를 최신화해줍니다.
        self.path_msg.header.stamp = rospy.Time.now()

        # 전체 경로의 개별 Pose에도 타임스탬프를 동기화해줍니다.
        for pose in self.path_msg.poses:
            pose.header.stamp = self.path_msg.header.stamp

        self.path_pub.publish(self.path_msg)
        rospy.loginfo_once("최초 Path 토픽 발행 완료! (latch 활성화됨)")


if __name__ == "__main__":
    try:
        publisher = PathPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass