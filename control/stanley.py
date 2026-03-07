#!/usr/bin/env python3

import rclpy
import numpy as np
import math as m
from erp42_msgs.msg import StanleyError,SerialFeedBack
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32



"""

Export module. Stanley Control Class.
Input: (state(class: State), [cx], [cy], [cyaw], last_target_idx)
Output: steer

"""

class Stanley:
    def __init__(self):
        
        self.__L = 1.240  # [m] Wheel base of vehicle
        # self.__k = self.declare_parameter("/stanley_controller/c_gain", 0.8).value
        # self.__hdr_ratio = self.declare_parameter("/stanley_controller/hdr_ratio", 0.03).value
        # self.__hdr_ratio = self.declare_parameter("/stanley_controller/hdr_ratio", 0.06).value

        
        # self.__hdr_ratio = self.declare_parameter("/stanley_controller/hdr_ratio", 0.5).value
        # self.__k = self.declare_parameter("/stanley_controller/c_gain", 0.24).value

        self.__hdr = 0.0    #heading error
        self.__ctr = 0.0    #crosstrack error
        
        self.k_v = 0.5
        self._last_idx = 0
        
    # def stanley_control(self, state, cx, cy, cyaw, last_target_idx, reverse=False):
    def stanley_control(self, state, cx, cy, cyaw, h_gain, c_gain, reverse=False):

        current_target_idx, error_front_axle, cte = self.calc_target_index(state, cx, cy, cyaw, reverse=reverse)

        # theta_e = (self.normalize_angle(
        #     cyaw[current_target_idx] - (state.yaw + (np.pi if reverse else 0.)))) * h_gain
        
        theta_e = (self.normalize_angle(
            cyaw[current_target_idx] - (state.yaw))) * h_gain
        
        hdr = self.normalize_angle(
            cyaw[current_target_idx] - (state.yaw + (np.pi if reverse else 0.)))
        
        
        theta_d = np.arctan2(c_gain * error_front_axle,
                           (self.k_v + state.v)) * (-1.0 if reverse else 1.0)

        # Field
        self.__hdr = theta_e
        self.__ctr = theta_d

        # Steering control
        delta = theta_e + theta_d

        delta = np.clip(delta, m.radians((-1) * 28), m.radians(28))

        return delta, current_target_idx, error_front_axle, hdr
    
    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle
    
    def calc_target_index(self, state, cx, cy, cyaw, reverse=False):
        
        inv_dir = -1.0 if reverse else 1.0
        fx = state.x + (self.__L / 2.0) * np.cos(state.yaw) * inv_dir
        fy = state.y + (self.__L / 2.0) * np.sin(state.yaw) * inv_dir

        # fx = state.x + (self.__L / 2.0) * np.cos(state.yaw)
        # fy = state.y + (self.__L / 2.0) * np.sin(state.yaw)

        search_range = 200
        start_idx = max(0, self._last_idx - search_range)
        end_idx = min(len(cx), self._last_idx + search_range)

        dx_list = [fx - icx for icx in cx[start_idx:end_idx]]
        dy_list = [fy - icy for icy in cy[start_idx:end_idx]]
        d_list = np.hypot(dx_list, dy_list)

        min_idx_relative = np.argmin(d_list)
        target_idx = min_idx_relative + start_idx
        self._last_idx = target_idx

        cte_value = d_list[min_idx_relative]

        vec_path_to_front = [fx - cx[target_idx], fy - cy[target_idx]]
        path_yaw = cyaw[target_idx]
        
        # 외적을 통한 부호 결정
        side = np.sin(path_yaw) * vec_path_to_front[0] - np.cos(path_yaw) * vec_path_to_front[1]
                
        if side > 0:
            error_front_axle = cte_value
        else:
            error_front_axle = -cte_value

        return target_idx, error_front_axle, cte_value

    # def calc_target_index(self, state, cx, cy, cyaw, reverse=False):
    #     """
    #     Compute index in the trajectory list of the target.
    #     :param state: (State object)
    #     :param cx: [float]
    #     :param cy: [float]
    #     :return: (int, float)
    #     """
    #     # Calc front axle position

    #     fx = state.x + self.__L * \
    #         np.cos(state.yaw) / 2.0 * (-1.0 if reverse else 1.0)
    #     fy = state.y + self.__L * \
    #         np.sin(state.yaw) / 2.0 * (-1.0 if reverse else 1.0)

    #     # Search nearest point index
    #     # dx = [fx - icx for icx in cx]
    #     # dy = [fy - icy for icy in cy]

    #     # d = np.hypot(dx, dy)
    #     # target_idx = int(np.argmin(d))
    #     # self._last_idx = target_idx

    #     # if self._last_idx >= len(cx):
    #     #     self._last_idx = 0
            
    #     search_range = 50
    #     start_idx = max(0, self._last_idx - search_range)
    #     end_idx = min(len(cx), self._last_idx + search_range)

    #     # 제한된 범위 내에서 최단 거리 점 탐색
    #     dx = [fx - icx for icx in cx[start_idx:end_idx]]
    #     dy = [fy - icy for icy in cy[start_idx:end_idx]]

    #     d = np.hypot(dx, dy)

    #     min_idx_in_range = np.argmin(d)
    #     target_idx = min_idx_in_range + start_idx
    #     self._last_idx = target_idx
    #     # target_idx = int(np.argmin(d)) + start_idx
    #     # self._last_idx = target_idx
    #     # if side < 0:
    #     #     cte = -cte

    #     # inv_dir = -1.0 if reverse else 1.0

    #     # # Project RMS error onto front axle vector
    #     # front_axle_vec = [
    #     #     inv_dir * -np.cos(state.yaw + np.pi / 2), 
    #     #     inv_dir * -np.sin(state.yaw + np.pi / 2)
    #     # ]
    #     # error_front_axle = np.dot(
    #     #     [dx[target_idx], dy[target_idx]], front_axle_vec)
        

    #     min_idx_in_range = np.argmin(d)
    #     target_idx = min_idx_in_range + start_idx
    #     self._last_idx = target_idx

    #     # 1. 순수 거리값(절대값)
    #     cte_value = d[min_idx_in_range]

    #     next_idx = min(target_idx + 1, len(cx) - 1)
    #     path_vec = [cx[next_idx] - cx[target_idx], cy[next_idx] - cy[target_idx]]
    #     error_vec = [fx - cx[target_idx], fy - cy[target_idx]]
    #     side = path_vec[0] * error_vec[1] - path_vec[1] * error_vec[0]

    #     # 2. 차량과 경로 사이의 상대 벡터
        
    #     # path_yaw = cyaw[target_idx]

    #     # # 3. 사이드 판별 (외적 방식)
    #     # side = np.sin(path_yaw) * vec_path_to_front[0] - np.cos(path_yaw) * vec_path_to_front[1]
        
    #     # 4. Steer 계산에 들어갈 최종 변수 결정 (가장 중요)
    #     if side > 0:
    #         error_front_axle = cte_value
    #     else:
    #         error_front_axle = -cte_value
        
    #     return target_idx, error_front_axle, cte_value
    