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
        delta_des = input[1]  # ステアリングの入力

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

    def get_discrete_matrix(self, dt, v, curvature):

        # delta=0の周りで線形化する
        delta_r = math.atan(self.wheelbase * curvature)

        # delta_rが大きすぎる場合は無理やり40度と仮定する
        if abs(delta_r) >= math.radians(40):
            delta_r = (math.radians(40)) * np.sign(delta_r)

        cos_squared_inv = 1 / ((math.cos(delta_r)) ** 2)

        """
        A = [[0, v, 0]
            [0, 0, v/L*cos_squared_delta],
            [0, 0, -1/tau]]
        B = [[0], [0], [1/tau]]
        C = [[1,0,0],
            [0,1,0]]
        W = [[0],
             [-v*curvature],
             [0]]
        """
        A = np.array([[0, v, 0], [0, 0, v / (self.wheelbase * cos_squared_inv)], [0, 0, -1 / self.tau]])
        B = np.array([[0], [0], [1 / self.tau]])
        C = np.array([[1, 0, 0], [0, 1, 0]])
        W = np.array(
            [0, -v * curvature + v / self.wheelbase * (math.tan(delta_r) - delta_r * cos_squared_inv), 0])
        I = np.eye(3)

        A_discrete_inverse = np.linalg.inv(I - dt * 0.5 * A)
        Ad = np.dot(A_discrete_inverse, (I + dt * 0.5 * A))  # 何故か逆行列の計算をしている
        Bd = A_discrete_inverse @ B * dt
        Cd = C
        Wd = W * dt

        return Ad, Bd, Cd, Wd
