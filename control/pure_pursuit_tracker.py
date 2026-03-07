#!/usr/bin/env python3
"""
Pure Pursuit - Path tracking controller

Geometric path tracking using look-ahead distance.
Adapted from PythonRobotics Pure Pursuit implementation.
"""
import numpy as np
import math


class PurePursuitController:
    """
    Pure Pursuit controller for path tracking.
    
    Args:
        k: Look-ahead gain (look-ahead distance = k * v + Lfc)
        Lfc: Minimum look-ahead distance [m]
        Kp: Speed proportional gain
        wheelbase: Distance between front and rear axles [m]
        max_steer: Maximum steering angle [rad]
    """
    
    def __init__(self):
        self.k = 0.4
        self.Lfc = 3
        self.wheelbase = 1.04
        self.max_steer = 28 * math.pi / 180
        # self.old_nearest_point_idx = None
        self._last_idx = 0

    
    def compute_control(self, state, path_x, path_y, path_yaw, reverse=False):
        """
        Compute steering angle to follow path.
        
        Args:
            state: VehicleState object
            path_x: List of path x-coordinates
            path_y: List of path y-coordinates
            
        Returns:
            steering_angle: [rad]
            target_idx: Look-ahead target point index
            look_ahead_dist: Current look-ahead distance [m]
        """
        # Pure Pursuit은 보통 뒷바퀴 기준이므로 기준점(rx, ry)을 설정
        # 전진 시 뒷바퀴: -, 후진 시 뒷바퀴: + (Stanley와 반대 방향일 수 있음)
        inv_dir = 1.0 if reverse else -1.0 
        rear_x = state.x + (self.wheelbase / 2.0) * np.cos(state.yaw) * inv_dir
        rear_y = state.y + (self.wheelbase / 2.0) * np.sin(state.yaw) * inv_dir

        # 탐색 범위 제한 (제안하신 방식 유지)
        search_range = 200
        start_idx = max(0, self._last_idx - search_range)
        end_idx = min(len(path_yaw), self._last_idx + search_range)

        # 거리 계산 및 최단 인덱스 추출
        dx = [rear_x - icx for icx in path_x[start_idx:end_idx]]
        dy = [rear_y - icy for icy in path_y[start_idx:end_idx]]
        d = np.hypot(dx, dy)
        
        min_idx_relative = np.argmin(d)
        near_idx = min_idx_relative + start_idx
        self._last_idx = near_idx
        
        # CTE 및 사이드 판별 (제안하신 Stanley 로직 그대로 사용 가능)
        cte_value = d[min_idx_relative]
        vec_path_to_rear = [rear_x - path_x[near_idx], rear_y - path_y[near_idx]]
        side = np.sin(path_yaw[near_idx]) * vec_path_to_rear[0] - np.cos(path_yaw[near_idx]) * vec_path_to_rear[1]
        
        error_rear_axle = cte_value if side > 0 else -cte_value

        # return target_idx, error_rear_axle, cte_value

        # direction = -1.0 if reverse else 1.0
        # # Calculate rear axle position
        # rear_x = state.x + direction * (self.wheelbase / 2) * math.cos(state.yaw)
        # rear_y = state.y + direction * (self.wheelbase / 2) * math.sin(state.yaw)
        
        # # Find nearest point on path
        # if self.old_nearest_point_idx is None:
        #     # Initial search
        #     dx = [rear_x - icx for icx in path_x]
        #     dy = [rear_y - icy for icy in path_y]
        #     d = np.hypot(dx, dy)
        #     ind = np.argmin(d)
        #     self.old_nearest_point_idx = ind
        # else:
        #     # Incremental search from previous index
        #     ind = self.old_nearest_point_idx
        #     distance_this_index = math.hypot(rear_x - path_x[ind], rear_y - path_y[ind])

        #     while True:
        #         if ind + 1 >= len(path_x):
        #             break
        #         distance_next_index = math.hypot(rear_x - path_x[ind + 1], 
        #                                           rear_y - path_y[ind + 1])
        #         if distance_this_index < distance_next_index:
        #             break
        #         ind = ind + 1
        #         distance_this_index = distance_next_index
            
        #     self.old_nearest_point_idx = ind

        # near_idx = self.old_nearest_point_idx

        hdr = self._normalize_angle(path_yaw[near_idx] - state.yaw)

        def calc_lookahead_gain(v):
            # v: [m/s]
            if v < 2.0:  # 7.2 km/h 이하
                return 0.3
            elif v < 4.5:  # 18 km/h 이하
                return 0.4
            else:  # 20 km/h 이상
                return 0.4
            
        self.k = calc_lookahead_gain(state.v)

        # Calculate look-ahead distance
        Lf = self.k * abs(state.v) + self.Lfc
        
        ind = near_idx

        # Search look-ahead target point
        while Lf > math.hypot(rear_x - path_x[ind], rear_y - path_y[ind]):
            if (ind + 1) >= len(path_x):
                break  # Not exceed goal
            ind += 1
        
        # Get target point
        if ind < len(path_x):
            tx = path_x[ind]
            ty = path_y[ind]
        else:
            tx = path_x[-1]
            ty = path_y[-1]
            ind = len(path_x) - 1
        
        # Calculate steering angle
        alpha = math.atan2(ty - rear_y, tx - rear_x) - (state.yaw + (math.pi if reverse else 0))
        alpha = self._normalize_angle(alpha)
        
        # Pure pursuit geometry
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha) / Lf, 1.0)
        
        # Limit steering angle
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        if reverse:
            delta = -delta
        
        return delta, near_idx, error_rear_axle
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def reset(self):
        """Reset the controller state"""
        self.old_nearest_point_idx = None


# Test code
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    
    from planning.vehicle_model import VehicleState, BicycleModel
    import matplotlib.pyplot as plt
    
    print("Testing Pure Pursuit Controller...")
    
    # Create test path (circle)
    radius = 10.0
    theta = np.linspace(0, 2 * math.pi, 200)
    path_x = radius * np.cos(theta)
    path_y = radius * np.sin(theta)
    
    # Initial state (inside circle)
    state = VehicleState(x=0.0, y=-3.0, yaw=0.0, v=0.0, omega=0.0)
    
    # Controller and vehicle model
    controller = PurePursuitController(k=0.1, Lfc=2.0, wheelbase=2.9)
    model = BicycleModel(wheelbase=2.9)
    
    # Target speed
    target_speed = 3.0  # m/s
    
    # Simulation
    dt = 0.1
    max_time = 30.0
    time = 0.0
    
    trajectory_x = [state.x]
    trajectory_y = [state.y]
    target_idx = 0
    
    print(f"\nStarting simulation with initial state: {state}")
    
    while time < max_time:
        # Speed control (simple P controller)
        accel = controller.Kp * (target_speed - state.v)
        state.v += accel * dt
        state.v = np.clip(state.v, 0, 10.0)
        
        # Pure pursuit steering control
        delta, target_idx, Lf = controller.compute_control(
            state, path_x, path_y
        )
        
        # Update state
        state = model.update_with_steering(state, state.v, delta, dt)
        
        trajectory_x.append(state.x)
        trajectory_y.append(state.y)
        time += dt
        
        if int(time * 10) % 10 == 0:
            print(f"  t={time:.1f}s: pos=({state.x:.2f}, {state.y:.2f}), "
                  f"v={state.v:.2f} m/s, Lf={Lf:.2f}m")
        
        # Stop if close to start (completed loop)
        if time > 5.0:
            dist_to_start = math.hypot(state.x - trajectory_x[0], 
                                       state.y - trajectory_y[0])
            if dist_to_start < 1.0:
                print(f"\nCompleted loop at time {time:.1f}s!")
                break
    
    # Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(10, 10))
    
    plt.plot(path_x, path_y, '--r', linewidth=2, label='Reference Path')
    plt.plot(trajectory_x, trajectory_y, '-b', linewidth=2, label='Actual Trajectory')
    plt.plot(trajectory_x[0], trajectory_y[0], 'og', markersize=10, label='Start')
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'or', markersize=10, label='End')
    
    # Plot look-ahead visualization at a few points
    for i in range(0, len(trajectory_x), 50):
        circle = plt.Circle((trajectory_x[i], trajectory_y[i]), 
                           controller.Lfc, color='g', fill=False, alpha=0.3)
        plt.gca().add_patch(circle)
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Pure Pursuit Controller - Path Tracking')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('/tmp/pure_pursuit_controller_test.png')
    print("Saved plot to /tmp/pure_pursuit_controller_test.png")
    plt.show()
    
    print("\nPure Pursuit controller test completed!")
