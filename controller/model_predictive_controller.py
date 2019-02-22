import numpy as np
import math
import cvxopt
import cvxpy
import scipy.interpolate as interp


class MPCController:

    def __init__(self, ref_path, param):
        self.IDX_X = 0
        self.IDX_Y = 1
        self.IDX_XY = [0, 1]
        self.IDX_YAW = 2
        self.IDX_VEL = 3
        self.IDX_CURVATURE = 4
        self.IDX_TIME = 5
        self.IDX_DELTA = 3
        self.wheelbase = param.wheelbase
        self.tau= param.tau

        self.ref_path = ref_path
        self.ref_sp = np.zeros(6)

        self.mpc_dt = param.mpc_dt
        self.mpc_n = param.mpc_n
        self.Q = param.mpc_Q
        self.R = param.mpc_R
        self.mpc_t = self.ref_sp[self.IDX_TIME]
        self.DIM_X = 3
        self.DIM_Y = 2
        self.DIM_U = 1
        self.Aex = np.zeros([self.DIM_X * self.mpc_n, self.DIM_X])
        self.Bex = np.zeros([self.DIM_X * self.mpc_n, self.DIM_U * self.mpc_n])
        self.Wex = np.zeros([self.DIM_X * self.mpc_n, 1])
        self.Cex = np.zeros([self.DIM_Y * self.mpc_n, self.DIM_X * self.mpc_n])
        self.Qex = np.zeros([self.DIM_Y * self.mpc_n, self.DIM_Y * self.mpc_n])
        self.Rex = np.zeros([self.DIM_U * self.mpc_n, self.DIM_U * self.mpc_n])
        self.mpc_ref_v = np.zeros

    def get_nearest_position(self, state):
        min_index = np.linalg.norm(self.ref_path[:, self.IDX_XY] - state[self.IDX_XY]).argmin()
        if min_index==0:
            min_index = 1
        self.ref_sp = self.ref_path[min_index, :]

    def simplify_radians(self, rad):
        """
        :param rad: radian(角度)
        :return: -pi~piの範囲内の角度
        """
        while not -2 * math.pi <= rad <= 2 * math.pi:
            if rad >= 2 * math.pi:
                rad -= 2 * math.pi
            elif rad <= -2 * math.pi:
                rad += 2 * math.pi

        if rad > math.pi:
            rad -= 2 * math.pi
        elif rad < -math.pi:
            rad += 2 * math.pi

        return rad

    def calc_error(self, state):
        self.get_nearest_position(state)
        sp_yaw = self.ref_sp[self.IDX_YAW]
        # 回転行列を定義
        rotation_matrix = np.array([[math.cos(sp_yaw), math.sin(sp_yaw)], [-math.sin(sp_yaw), math.cos(sp_yaw)]])
        error_xy = (state[self.IDX_XY] - self.ref_sp[self.IDX_XY]).T
        error_lonlat = rotation_matrix * error_xy
        error_lat = error_lonlat[1]

        # yaw角のerrorを計算する
        error_yaw = state[self.IDX_YAW] - sp_yaw
        error_yaw = self.simplify_radians(error_yaw)

        return np.array([[error_lat], [error_yaw], [state[self.IDX_DELTA]]])

    def interpolate1d(self, r_path, t):
        f_inter_x = interp.interp1d(r_path[:, self.IDX_TIME], r_path[:, 0])
        ref_x = f_inter_x(t)

        f_inter_y = interp.interp1d(r_path[:, self.IDX_TIME], r_path[:, 1])
        ref_y = f_inter_y(t)

        f_inter_yaw = interp.interp1d(r_path[:, self.IDX_TIME], r_path[:, 2])
        ref_yaw = f_inter_yaw(t)

        f_inter_vel = interp.interp1d(r_path[:, self.IDX_TIME], r_path[:, 3])
        ref_vel = f_inter_vel(t)

        f_inter_curv = interp.interp1d(r_path[:, self.IDX_TIME], r_path[:, 4])
        ref_curv = f_inter_curv(t)

        return np.array([ref_x, ref_y, ref_yaw, ref_vel, ref_curv])

    def get_error_kinematics_state_matrix_(self, dt, v, curvature):
        """
        ここではkinematicsモデルの計算を行うときの行列を定義する
        :param dt: 時刻
        :param v: 速度
        :param curvature: 曲率
        :return:
        """

        # delta=0の周りで線形化する
        delta_r = math.atan(self.wheelbase*curvature)

        # delta_rが大きすぎる場合は無理やり40度と仮定する
        if abs(delta_r) >= math.radians(40):
            delta_r = (math.radians(40))*np.sign(delta_r)

        cos_squared_inv = 1 / ((math.cos(delta_r))**2)

        """
        A = [[0, v, 0]
            [0, 0, v/L*cos_squared_delta],
            [0, 0, -1/tau]]
        B = [[0], [0], 1/tau]]
        C = [[1,0,0],
            [0,1,0]]
        W = [[0],
             [-v*curvature],
             [0]]
        """
        A = np.array([[0, v, 0], [0, 0, v/(self.wheelbase*cos_squared_inv)], [0, 0, -1/self.tau]])
        B = np.array([[0], [0], [1/self.tau]])
        C = np.array([[1, 0, 0], [0, 1, 0]])
        W = np.array([[0], [-v*curvature + v/self.wheelbase*(math.tan(delta_r) - delta_r*cos_squared_inv)], [0]])
        I = np.diag(np.array([1, 1, 1]))
        Ad = (I + dt*0.5*A) # 何故か逆行列の計算をしている
        Bd = B * dt
        Cd = C
        Wd = W*dt

        return Ad, Bd, Cd, Wd

    def create_matrix(self):
        # 時間を更新
        self.mpc_t = self.ref_sp[self.IDX_TIME]

        """
        mpc matrix for i=1
        """
        # まずは線形補間を行っていく(ref_i_に時刻mpc_tの時の位置、速度、yaw角、曲率が入る)
        ref_i_ = self.interpolate1d(self.ref_path, self.mpc_t)
        v_ = ref_i_[self.IDX_VEL]
        k_ = ref_i_[self.IDX_CURVATURE]
        Ad, Bd, Cd, Wd = self.get_error_kinematics_state_matrix_(self.mpc_dt, v_, k_)
        self.Aex[0:self.DIM_X, :] = Ad
        self.Bex[0:self.DIM_X, 0:self.DIM_U] = Bd
        self.Wex[0:self.DIM_X] = Wd
        self.Cex[0:self.DIM_Y, 0:self.DIM_X] = Cd
        self.Qex[0:self.DIM_Y, 0:self.DIM_Y] = self.Q
        self.Rex[0:self.DIM_U, 0:self.DIM_U] = self.R
        mpc_ref_v = v_

