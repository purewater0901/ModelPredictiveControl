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
        self.x0 = np.zeros(3)
        self.wheelbase = param.wheelbase
        self.tau = param.tau

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
        self.Wex = np.zeros([self.DIM_X * self.mpc_n, ])
        self.Cex = np.zeros([self.DIM_Y * self.mpc_n, self.DIM_X * self.mpc_n])
        self.Qex = np.zeros([self.DIM_Y * self.mpc_n, self.DIM_Y * self.mpc_n])
        self.Rex = np.zeros([self.DIM_U * self.mpc_n, self.DIM_U * self.mpc_n])
        self.mpc_ref_v = 0
        self.steering_rate_lim = math.radians(param.mpc_constraint_steer_rate_deg)
        self.mpc_solve_without_constraint = param.mpc_solve_without_constraint
        self.mpc_constraint_steering_deg = param.mpc_constraint_steer_rate_deg
        self.mpc_constraint_steer_rate__deg = param.mpc_constraint_steer_rate_deg
        self.mpc_sensor_delay = param.mpc_sensor_delay

    def get_nearest_position(self, state):
        min_index = np.linalg.norm(self.ref_path[:, self.IDX_XY] - state[self.IDX_XY]).argmin()
        if min_index == 0:
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
        self.get_nearest_position(state)  # 一番近い点をゲットする
        sp_yaw = self.ref_sp[self.IDX_YAW]
        # 回転行列を定義
        rotation_matrix = np.array([[math.cos(sp_yaw), math.sin(sp_yaw)], [-math.sin(sp_yaw), math.cos(sp_yaw)]])
        error_xy = (state[self.IDX_XY] - self.ref_sp[self.IDX_XY]).T
        error_lonlat = np.dot(rotation_matrix, error_xy)
        error_lat = error_lonlat[1]

        # yaw角のerrorを計算する
        error_yaw = state[self.IDX_YAW] - sp_yaw
        error_yaw = self.simplify_radians(error_yaw)

        self.x0[0] = error_lat
        self.x0[1] = error_yaw
        self.x0[2] = state[self.IDX_DELTA]

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
        I = np.diag(np.array([1, 1, 1]))
        Ad = np.dot(np.linalg.inv(I - dt * 0.5 * A), (I + dt * 0.5 * A))  # 何故か逆行列の計算をしている
        Bd = B * dt
        Cd = C
        Wd = W * dt

        return Ad, Bd, Cd, Wd

    def create_matrix(self):
        # 時間を更新
        self.mpc_t = self.ref_sp[self.IDX_TIME]

        """
        mpc matrix for i=0
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

        """
        mpc matrix for i=1:n
        """
        for i in range(1, self.mpc_n):
            # update mpc time
            if self.mpc_t > self.ref_path[len(self.ref_path) - 1, self.IDX_TIME]:
                self.mpc_t = self.ref_path[len(self.ref_path) - 1, self.IDX_TIME]
                print("mpc path is too short to predict dyanamics")

            ref_i_ = self.interpolate1d(self.ref_path, self.mpc_t)
            v_ = ref_i_[self.IDX_VEL]
            k_ = ref_i_[self.IDX_CURVATURE]
            Ad, Bd, Cd, Wd = self.get_error_kinematics_state_matrix_(self.mpc_dt, v_, k_)

            # update matrix
            idx_x_now_f = i * self.DIM_X
            idx_x_now_l = (i + 1) * self.DIM_X
            idx_x_prev_f = (i - 1) * self.DIM_X
            idx_x_prev_l = i * self.DIM_X
            idx_u_f = i * self.DIM_U
            idx_u_l = (i + 1) * self.DIM_U
            idx_y_f = i * self.DIM_Y
            idx_y_l = (i + 1) * self.DIM_Y

            self.Aex[i * self.DIM_X:(i + 1) * self.DIM_X, :] = Ad * self.Aex[idx_x_prev_f:idx_x_prev_l, :]

            for j in range(0, i):
                idx_u_col = [k for k in range(j * self.DIM_U, (j + 1) * self.DIM_U)]
                self.Bex[idx_x_now_f:idx_x_now_l, idx_u_col] = np.dot(Ad,
                                                                      self.Bex[idx_x_prev_f:idx_x_prev_l, idx_u_col])

            self.Bex[idx_x_now_f:idx_x_now_l, idx_u_f:idx_u_l] = Bd
            self.Wex[idx_x_now_f:idx_x_now_l] = np.dot(Ad, self.Wex[idx_x_prev_f:idx_x_prev_l]) + Wd
            self.Cex[idx_y_f:idx_y_l, idx_x_now_f:idx_x_now_l] = Cd
            self.Qex[idx_y_f:idx_y_l, idx_y_f:idx_y_l] = self.Q
            self.Rex[idx_u_f:idx_u_l, idx_u_f:idx_u_l] = self.R

    def convex_optimization(self):

        """
        our objective is to get optimum U. So we will so the following equation
        1/2*U'*mat1*U + mat2*U+C
        :return:
        """

        mat1_tmp1 = np.dot(self.Bex.T, self.Cex.T)
        mat1_tmp2 = np.dot(mat1_tmp1, self.Qex)
        mat1_tmp3 = np.dot(mat1_tmp2, self.Cex)
        mat1_tmp4 = np.dot(mat1_tmp3, self.Bex)
        mat1 = mat1_tmp4 + self.Rex

        mat2_tmp1 = np.dot(self.x0.T, self.Aex.T) + self.Wex.T
        mat2_tmp2 = np.dot(mat2_tmp1, self.Cex.T)
        mat2_tmp3 = np.dot(mat2_tmp2, self.Qex)
        mat2_tmp4 = np.dot(mat2_tmp3, self.Cex)
        mat2 = np.dot(mat2_tmp4, self.Bex)

        if self.mpc_solve_without_constraint:
            np.linalg.inv(-mat1) * mat2.T
        else:
            H_ = (mat1 + mat1.T) / 2
            f_ = mat2

            # add steering rate constraint
            tmp = -np.eye(self.mpc_n - 1, self.mpc_n)
            tmp[:, 1:] = tmp[:, 1:] + np.eye(self.mpc_n - 1)
            T_ = np.kron(tmp, np.array([0, 0, 1])) / self.mpc_dt
            dsteer_vec_tmp = np.dot(T_, (np.dot(self.Aex, self.x0) + self.Wex))
            G_ = np.concatenate([np.dot(T_, self.Bex), np.dot(-T_, self.Bex)])
            h_ = np.concatenate([self.steering_rate_lim * np.ones(self.mpc_n - 1) - dsteer_vec_tmp,
                                 self.steering_rate_lim * np.ones(self.mpc_n - 1) + dsteer_vec_tmp])

            lb_ = -math.radians(self.mpc_constraint_steering_deg) * np.ones(self.mpc_n * self.DIM_U)
            ub_ = math.radians(self.mpc_constraint_steering_deg) * np.ones(self.mpc_n * self.DIM_U)

            sol = cvxopt.solvers.qp(cvxopt.matrix(H_), cvxopt.matrix(f_), G=cvxopt.matrix(G_), h=cvxopt.matrix(h_))
            tmp_vec = sol['x']

            input_vec = []
            for i in range(len(sol['x'])):
                input_vec.append(tmp_vec[i])

            input_vec = np.array(input_vec)
            t_mpc = np.arange(0, self.mpc_n*self.mpc_dt, self.mpc_dt)

            f_inter = interp.interp1d(t_mpc, input_vec)
            delta_des = f_inter(self.mpc_sensor_delay)  # センサーの遅延を考える
            v_des = self.ref_sp[self.IDX_VEL]

            return np.array([v_des, delta_des])

    def calc_input(self, state):
        self.calc_error(state)
        self.create_matrix()
        return self.convex_optimization()