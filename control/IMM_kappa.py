import os
import sys

import rclpy
from rclpy.node import Node
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Path, Odometry
from erp42_msgs.msg import ControlMessage
import math as m

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stanley import Stanley
from pure_pursuit_tracker import PurePursuitController

class IMMLogger:
    def __init__(self):
        # cte_s 제거, kappa 추가
        self.history = {
            'step': [],
            'mu_s': [], 'mu_p': [],
            'steer': [],
            'kappa': [],   # 곡률 데이터
            'cte_p': []    # Pure Pursuit의 CTE
        }

    def record(self, mu, steer, kappa, cte_p):
        self.history['step'].append(len(self.history['step']))
        self.history['mu_s'].append(mu[0])
        self.history['mu_p'].append(mu[1])
        self.history['steer'].append(np.rad2deg(steer))
        self.history['kappa'].append(kappa)
        self.history['cte_p'].append(cte_p)

    def plot(self):
        if not self.history['step']: return
        
        # 4개의 그래프를 그리기 위해 subplots(4, 1)로 변경
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
        steps = self.history['step']

        # 1. 모델 확률 (IMM Model Probabilities)
        ax1.fill_between(steps, 0, self.history['mu_s'], label='Stanley', alpha=0.6, color='royalblue')
        ax1.fill_between(steps, self.history['mu_s'], 1, label='Pure Pursuit', alpha=0.4, color='tomato')
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper right')
        ax1.set_title("Interacting Multiple Model (IMM) Results")

        # 2. 최종 조향각 (Steer Angle)
        ax2.plot(steps, self.history['steer'], color='black', linewidth=1.5, label='Final Steer')
        ax2.set_ylabel("Steer [deg]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # 3. 경로 곡률 (Path Curvature - Kappa)
        ax3.plot(steps, self.history['kappa'], color='forestgreen', label='Curvature (κ)')
        ax3.set_ylabel("Kappa [1/m]")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        # 4. Pure Pursuit CTE
        ax4.plot(steps, self.history['cte_p'], color='darkorange', label='PP CTE')
        ax4.set_ylabel("CTE [m]")
        ax4.set_xlabel("Simulation Step")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

class VehicleState:
    def __init__(self, node):
        self.x = 0.0  # m
        self.y = 0.0  # m
        self.yaw = 0.0  # rad
        self.v = 0.0  # m/s

        self.node = node

        # self.node.create_subscription(
        #     Odometry, '/odometry/global', self.odom_callback, 10
        # )
        self.node.create_subscription(
            Odometry, '/localization/kinematic_state', self.odom_callback, 10
        )

    def odom_callback(self, msg):
        """Update current state from odometry"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        quat = msg.pose.pose.orientation
        self.yaw = np.arctan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        )
        
        self.v = msg.twist.twist.linear.x
        self.has_state = True

class IMMController(Node):
    def __init__(self):
        super().__init__('IMM_Node')

        self.stanley = Stanley()
        self.pp = PurePursuitController()
                
        self.num_models = 2
        self.mu = np.ones(self.num_models) / self.num_models  # 초기 모델 확률
                
        self.TPM = np.array([[0.98, 0.02],
                             [0.02, 0.98]])

        self.logger = IMMLogger()
        self.current_state = VehicleState(self)

        self.path_x = []
        self.path_y = []
        self.path_yaw = []

        self.last_target_idx = 0
        self.has_state = False
        self.has_path = False

        self.path_sub = self.create_subscription(
            Path,
            '/local_path',
            self.path_callback,
            10
        )

        self.cmd_pub = self.create_publisher(ControlMessage, '/cmd_msg', 10)

        self.timer = self.create_timer(0.05, self.control_step)

        self.alive_cnt = 0

        self.is_first_run = True
    
    def path_callback(self, msg):
        """Update reference path"""
        if len(msg.poses) < 2:
            self.has_path = False
            return
        
        # Extract path waypoints
        
        # for pose in msg.poses:
        #     self.path_x.append(pose.pose.position.x)
        #     self.path_y.append(pose.pose.position.y)
            
        #     # Calculate yaw from quaternion
        #     quat = pose.pose.orientation
        #     yaw = np.arctan2(
        #         2.0 * (quat.w * quat.z + quat.x * quat.y),
        #         1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        #     )
        #     self.path_yaw.append(yaw)

        new_x, new_y, new_yaw = [], [], []
        for pose in msg.poses:
            new_x.append(pose.pose.position.x)
            new_y.append(pose.pose.position.y)
            quat = pose.pose.orientation
            yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z))
            new_yaw.append(yaw)
        
        self.path_x = new_x
        self.path_y = new_y
        self.path_yaw = new_yaw
        
        dx = [self.current_state.x - icx for icx in new_x]
        dy = [self.current_state.y - icy for icy in new_y]
        nearest_idx = int(np.argmin(np.hypot(dx, dy)))
    
        self.stanley._last_idx = nearest_idx 
        self.pp._last_idx = nearest_idx
            
        self.has_path = True

    def update_joint_likelihood(self, kappa, cte):
        """
        곡률(kappa)과 횡방향 오차(cte)를 결합하여 우도 계산
        """
        # kappa가 클수록 PP 선호
        kappa_threshold = 0.15
        # if kappa < 0.05: # 낮은 곡률에서는 강제로 Stanley 우도 높임
        #     prob_pp_kappa = 0.0
        # else:
        #     prob_pp_kappa = 1.0 / (1.0 + np.exp(-15 * (kappa - kappa_threshold)))
        prob_st_kappa = 1.0 / (1.0 + np.exp(-15 * (kappa - kappa_threshold)))

        # cte가 클수록(예: 0.3m 이상) 복귀 능력이 좋은 PP 선호
        cte_threshold = 0.6
        prob_st_cte = 1.0 / (1.0 + np.exp(-10 * (abs(cte) - cte_threshold)))

        l_st = np.maximum(prob_st_kappa, prob_st_cte)
        
        l_pp = 1.0 - l_st

        return l_pp, l_st

    def calculate_curvature_at_index(self, target_idx):
        """
        현재 target_idx 기준 전방 5개 점을 사용하여 곡률 계산
        """
        
        step = 10  # 예: 2칸씩 띄어서 추출 (필요에 따라 1로 수정 가능)
        point_to_fit = 35
        sample_indices = [target_idx + i * step for i in range(point_to_fit)]
        
        # 경로 범위를 벗어나지 않도록 필터링
        valid_indices = [i for i in sample_indices if i < len(self.path_x)]
        
        if len(valid_indices) < 3: # 최소 3개는 있어야 피팅 가능
            return 0.0

        # 2. 로컬 좌표계 변환 (차량 기준)
        # 계산의 수치적 안정성을 위해 현재 위치를 원점으로 회전 변환합니다.
        local_x = []
        local_y = []
        
        cos_yaw = np.cos(-self.current_state.yaw)
        sin_yaw = np.sin(-self.current_state.yaw)
        
        for i in valid_indices:
            dx = self.path_x[i] - self.current_state.x
            dy = self.path_y[i] - self.current_state.y
            
            # 차량 중심 로컬 좌표계 변환
            lx = dx * cos_yaw - dy * sin_yaw
            ly = dx * sin_yaw + dy * cos_yaw
            local_x.append(lx)
            local_y.append(ly)

        try:
            poly = np.polyfit(local_x, local_y, 2)
            # 2차항 계수가 매우 작으면 직선으로 강제 확정
            if abs(poly[0]) < 1e-5: 
                return 0.0
                
            kappa = abs(2 * poly[0]) / (1 + poly[1]**2)**1.5
            return kappa if kappa > 0.001 else 0.0 # 0.01 미만 곡률은 무시
        except:
            return 0.0
    
    def control_step(self):
        try:
            self.control_step_imm()
        except Exception as e:
            self.get_logger().error(f"[IMM] controlo step 크래시: {e}", throttle_duration_sec=1.0)

            cmd = ControlMessage()
            cmd.gear = 2; cmd.brake = 200; cmd.estop = 1
            cmd.alive = self.alive_cnt
            self.cmd_pub.publish(cmd)


    def control_step_imm(self):
        
        if self.is_first_run:
            # 처음 실행 시에는 0번 인덱스 근처에서만 탐색하도록 제한
            self.last_target_idx= 0
            self.is_first_run = False

        cmd = ControlMessage()
        cmd.gear = 2
        cmd.brake = 0
        cmd.speed = 0
        cmd.steer = 0
        cmd.alive = self.alive_cnt
        self.alive_cnt = (self.alive_cnt + 1) % 256

        if not (self.current_state.has_state and self.has_path):
            cmd.estop = 2
            cmd.speed = 0
            self.cmd_pub.publish(cmd)
            self.get_logger().warn("Waiting for state and path...", throttle_duration_sec=2.0)
            return
        

        if len(self.path_x) < 2:
            cmd.estop = 1; cmd.brake = 100
            self.cmd_pub.publish(cmd)
            return
        
        # cmd.estop = 0
        # cmd.speed = 100

        is_reverse = False
        # if self.target_gear == 1: # 후진 기어일 때
        #     is_reverse = True
        
        cmd = ControlMessage()
        cmd.gear = 1 if is_reverse else 2
        cmd.speed = 30 if is_reverse else 100
        
        delta_s, self.last_target_idx, cte_s, _ = self.stanley.stanley_control(self.current_state, self.path_x, self.path_y, self.path_yaw, h_gain=0.5, c_gain=0.24, reverse=is_reverse)
        # delta_s, self.last_target_idx, cte_s, _ = self.stanley.stanley_control(self.current_state, self.path_x, self.path_y, self.path_yaw, h_gain=0.3, c_gain=0.02, reverse=is_reverse)
        delta_p, _, _ = self.pp.compute_control(self.current_state, self.path_x, self.path_y, self.path_yaw, reverse=is_reverse)

        path_len = len(self.path_x)
        self.last_target_idx = max(0, min(self.last_target_idx, path_len - 1))

        kappa = self.calculate_curvature_at_index(self.last_target_idx)
        
        # kappa_threshold = 0.2 
        # l_pp = 1.0 / (1.0 + np.exp(-15 * (kappa - kappa_threshold))) # Sigmoid 함수 사용
        # l_st = 1.0 - l_pp
        l_pp, l_st = self.update_joint_likelihood(kappa, cte_s)

        new_mu = 0.8 * self.mu + 0.2 * np.array([l_st, l_pp])
        
        if new_mu.sum() > 1e-6:
            self.mu = new_mu / new_mu.sum()
        else:
            self.mu = np.array([0.5, 0.5])

        dominant = 'Stanley' if self.mu[0] >= self.mu[1] else 'PurePursuit'
        self.get_logger().info(f'[IMM] {dominant} (idx={self.last_target_idx:.2f} | CTE={cte_s:.2f})', throttle_duration_sec = 1.0)
        # self.mu = np.array([1.0, 0.0])
        final_delta = self.mu[0] * delta_s + self.mu[1] * delta_p

        final_delta = np.clip(final_delta, m.radians((-1) * 28), m.radians(28))

        self.logger.record(self.mu, final_delta, kappa, cte_s)

        if self.last_target_idx >= int(path_len * 0.97):
            self.get_logger().warn('경로 끝 도달 - 정지', throttle_duration_sec = 1.0)
            cmd.speed = 0
            cmd.estop = 1
            cmd.brake = 200
            cmd.steer = 0
            self.cmd_pub.publish(cmd)
            return
        
        # self.get_logger().info(f'path_len : {path_len},  target_idx : {self.last_target_idx}, state_yaw : {self.current_state.yaw}')
        if is_reverse:
            cmd.steer = int(m.degrees(final_delta)) 
        else:
            cmd.steer = int(-m.degrees(final_delta))

        self.cmd_pub.publish(cmd)
    
    def show_history(self):
        self.logger.plot()

def main():
    rclpy.init(args=None)

    imm = IMMController()

    try:
        rclpy.spin(imm)
    except KeyboardInterrupt():
        imm.get_logger().info("Shutting down...")

    finally:
        imm.show_history()
        imm.destroy_node()
        rclpy.shutdown()
    

if __name__ == '__main__':
    main()