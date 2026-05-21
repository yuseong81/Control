import csv
import math


def generate_straight_curve_straight_path(filename="path_topic.csv"):
    # --- 파라미터 설정 ---
    ds = 0.1  # 점과 점 사이의 간격 (정밀도: 10cm)

    len_straight1 = 10.0  # 첫 번째 직선 구간 길이 (10m)
    curve_radius = 5.0  # 커브 반지름 (5m)
    curve_angle_deg = 90.0  # 회전할 각도 (90도 직각 커브)
    len_straight2 = 10.0  # 두 번째 직선 구간 길이 (10m)

    path_points = []

    # 1. 첫 번째 직선 구간 (x축 방향으로 전진)
    current_x = 0.0
    current_y = 0.0
    current_yaw = 0.0  # 진행 방향 (라디안, 0은 x축 정방향)

    num_steps1 = int(len_straight1 / ds)
    for _ in range(num_steps1):
        path_points.append((current_x, current_y, current_yaw))
        current_x += ds * math.cos(current_yaw)
        current_y += ds * math.sin(current_yaw)

    # 2. 커브 구간 (좌회전 예시)
    # 호의 길이 = 반지름 * 각도(라디안)
    curve_angle_rad = math.radians(curve_angle_deg)
    len_curve = curve_radius * curve_angle_rad
    num_steps_curve = int(len_curve / ds)

    # 회전 중심점 계산 (현재 위치에서 진행 방향의 왼쪽으로 반지름만큼 떨어진 곳)
    # 좌회전이므로 현재 yaw에서 +90도 방향이 중심점 방향입니다.
    center_x = current_x + curve_radius * math.cos(current_yaw + math.pi / 2)
    center_y = current_y + curve_radius * math.sin(current_yaw + math.pi / 2)

    # 시작 각도 계산
    start_angle = current_yaw - math.pi / 2

    for i in range(num_steps_curve):
        # 현재 진행률에 따른 각도 변화
        angle_ratio = i / num_steps_curve
        target_angle = start_angle + (curve_angle_rad * angle_ratio)

        # 원 위에 있는 좌표 계산
        c_x = center_x + curve_radius * math.cos(target_angle)
        c_y = center_y + curve_radius * math.sin(target_angle)
        # 현재 진행 방향(Yaw)은 접선 방향이므로 target_angle에서 90도를 더해줍니다.
        c_yaw = target_angle + math.pi / 2

        path_points.append((c_x, c_y, c_yaw))

    # 커브가 끝난 시점의 최종 위치와 방향 업데이트
    current_x = center_x + curve_radius * math.cos(start_angle + curve_angle_rad)
    current_y = center_y + curve_radius * math.sin(start_angle + curve_angle_rad)
    current_yaw = current_yaw + curve_angle_rad

    # 3. 두 번째 직선 구간 (변경된 방향으로 전진)
    num_steps2 = int(len_straight2 / ds)
    for _ in range(num_steps2):
        path_points.append((current_x, current_y, current_yaw))
        current_x += ds * math.cos(current_yaw)
        current_y += ds * math.sin(current_yaw)

    # 마지막 점 추가
    path_points.append((current_x, current_y, current_yaw))

    # --- CSV 파일로 저장 ---
    # ROS Path 토픽 등에서 쉽게 파싱할 수 있도록 x, y, yaw(방향) 형태로 저장합니다.
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # 헤더 작성
        writer.writerow(["x", "y", "yaw"])
        # 데이터 작성
        for pt in path_points:
            writer.writerow([f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}"])

    print(
        f"성공적으로 {filename} 파일이 생성되었습니다. (총 데이터 수: {len(path_points)}개)"
    )


if __name__ == "__main__":
    generate_straight_curve_straight_path()