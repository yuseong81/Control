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
        self.history = {
            'step': [],
            'mu_s': [], 'mu_p': [],
            'steer': [],
            'cte_s': [], 'cte_p': []
        }

    def record(self, mu, steer, cte_s, cte_p):
        self.history['step'].append(len(self.history['step']))
        self.history['mu_s'].append(mu[0])
        self.history['mu_p'].append(mu[1])
        self.history['steer'].append(np.rad2deg(steer))
        self.history['cte_s'].append(cte_s)
        self.history['cte_p'].append(cte_p)

    def plot(self):
        if not self.history['step']: return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        steps = self.history['step']

        # 1. 모델 확률 (누적 영역 그래프)
        ax1.fill_between(steps, 0, self.history['mu_s'], label='Stanley', alpha=0.6, color='royalblue')
        ax1.fill_between(steps, self.history['mu_s'], 1, label='Pure Pursuit', alpha=0.4, color='tomato')
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper right')
        ax1.set_title("IMM Model Probabilities")

        # 2. 최종 조향각
        ax2.plot(steps, self.history['steer'], color='black', linewidth=1)
        ax2.set_ylabel("Steer Angle [deg]")
        ax2.grid(True, alpha=0.3)

        # 3. 모델별 CTE 비교
        ax3.plot(steps, self.history['cte_s'], label='Stanley CTE', alpha=0.7)
        ax3.plot(steps, self.history['cte_p'], label='PP CTE', alpha=0.7)
        ax3.set_ylabel("CTE [m]")
        ax3.set_xlabel("Simulation Step")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

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
        
        self.sigma_cte = np.array([0.4, 0.4])
        self.sigma_yaw = np.array([np.deg2rad(5), np.deg2rad(15)])

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

        self.timer = self.create_timer(0.1, self.control_step)

        self.alive_cnt = 0
    
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
        self.stanley._last_idx = int(np.argmin(np.hypot(dx, dy)))
        
        self.has_path = True

    def _get_likelihood(self, cte, yaw_err, s_cte, s_yaw):
        # 우도 계산
        error_norm = (cte**2 / (2 * s_cte**2)) + (yaw_err**2 / (2 * s_yaw**2))
        return np.exp(-np.clip(error_norm, 0, 20)) + 1e-6
    
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

        # 예측 확률 계산
        c_j = self.TPM.T @ self.mu

        is_reverse = False
        # if self.target_gear == 1: # 후진 기어일 때
        #     is_reverse = True
        
        cmd = ControlMessage()
        cmd.gear = 1
        # cmd.gear = 1 if is_reverse else 2
        cmd.speed = 30 if is_reverse else 100
        
        # delta_s, self.last_target_idx, cte_s, hdr_s = self.stanley.stanley_control(self.current_state, self.path_x, self.path_y, self.path_yaw, h_gain=0.5, c_gain=0.24, reverse=is_reverse)
        delta_s, self.last_target_idx, cte_s, hdr_s = self.stanley.stanley_control(self.current_state, self.path_x, self.path_y, self.path_yaw, h_gain=0.3, c_gain=0.02, reverse=is_reverse)
        delta_p, hdr_p, cte_p = self.pp.compute_control(self.current_state, self.path_x, self.path_y, self.path_yaw, reverse=is_reverse)

        path_len = len(self.path_x)
        self.last_target_idx = max(0, min(self.last_target_idx, path_len - 1))
        
        L_s = self._get_likelihood(cte_s, hdr_s, self.sigma_cte[0], self.sigma_yaw[0])
        L_p = self._get_likelihood(cte_p, hdr_p, self.sigma_cte[1], self.sigma_yaw[1])
        
        new_mu = np.array([L_s * c_j[0], L_p * c_j[1]])
        
        if new_mu.sum() > 1e-6:
            self.mu = new_mu / new_mu.sum()
        else:
            self.mu = np.array([0.5, 0.5])

        dominant = 'Stanley' if self.mu[0] >= self.mu[1] else 'PurePursuit'
        self.get_logger().info(f'[IMM] {dominant} (S={self.last_target_idx:.2f} | PP={self.mu[1]:.2f})', throttle_duration_sec = 1.0)
        self.mu = np.array([1.0, 0.0])
        final_delta = self.mu[0] * delta_s + self.mu[1] * delta_p

        final_delta = np.clip(final_delta, m.radians((-1) * 28), m.radians(28))

        self.logger.record(self.mu, final_delta, cte_s, cte_p)

        if self.last_target_idx >= int(path_len * 0.97):
            self.get_logger().warn('경로 끝 도달 - 정지', throttle_duration_sec = 1.0)
            cmd.speed = 0
            cmd.estop = 1
            cmd.brake = 200
            cmd.steer = 0
            self.cmd_pub.publish(cmd)
            return
        
        self.get_logger().info(f'path_len : {path_len},  target_idx : {self.last_target_idx}, state_yaw : {self.current_state.yaw}')
        if is_reverse:
            cmd.steer = int(m.degrees(final_delta)) 
        else:
            cmd.steer = int(-m.degrees(final_delta))

        self.cmd_pub.publish(cmd)
        
        # print(self.path_x)
    
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