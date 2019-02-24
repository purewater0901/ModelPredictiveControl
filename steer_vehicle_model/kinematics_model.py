import math
import numpy as np


class Kinematics_Model:

    def __init__(self, param):
        self.steer_lim = param.steer_lim
        self.vel_min = param.vel_min
        self.vel_max = param.vel_max
        self.wheelbase = param.wheelbase
        self.tau = param.tau
        self.steering_steady_state_error_deg = param.steering_steady_state_error_deg

    def update(self, state, input):
        v_des = input[0]
        delta_des = input[1]

        delta_des = max(min(delta_des, self.steer_lim), -self.steer_lim)
        v_des = max(min(v_des, self.vel_max), self.vel_min)

        yaw = state[2]
        delta = state[3]

        v = v_des

        d_x = v * math.cos(yaw)
        d_y = v * math.sin(yaw)
        d_yaw = v * math.tan(delta) / self.wheelbase
        d_delta = -(delta - delta_des) / self.tau

        if abs(delta - delta_des) < self.steering_steady_state_error_deg / 180 * math.pi:
            d_delta = 0

        return np.array([d_x, d_y, d_yaw, d_delta])
