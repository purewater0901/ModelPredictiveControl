import math
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from config.parameter_setting import MPCConfig


class Path:

    def __init__(self):
        self.point = np.array(
            [[0.0, 0.0], [1, 0], [2, 0], [3, 0.5], [4, 1.5], [4.8, 1.5], [5, 0.8], [6, 0.5],
             [6.5, 0], [7.5, 0.5], [7, 2], [6, 3], [5, 4], [4, 2.5], [3, 3], [2, 3.5], [1.3, 2.2], [0.5, 2], [0, 3]])

        self.spline = self.calc_spline()
        self.yaw = self.calc_yaw()
        self.path = self.get_path()
        self.mpcconfig = MPCConfig()

    def plot(self):
        plt.plot(self.spline[0], self.spline[1], c='r', label='cubic spline')
        plt.plot(self.point[:, 0], self.point[:, 1], c='b', label='original line')
        plt.grid(True)
        plt.legend()
        plt.show()

    def calc_spline(self, deg=3, s=0):
        tck, u = interp.splprep([self.point[:, 0], self.point[:, 1]], k=3, s=0)
        u = np.linspace(0, 1, num=1801, endpoint=True) # 最後の要素を含む
        return np.array(interp.splev(u, tck))

    def calc_yaw(self):
        yaw = np.zeros(len(self.spline[0]))
        for i in range(1, len(self.spline[0])-1):
            x_forward = self.spline[0][i+1]
            x_backward= self.spline[0][i-1]
            y_forward = self.spline[1][i+1]
            y_backward= self.spline[1][i-1]
            yaw[i] = math.atan2(y_forward-y_backward, x_forward-x_backward)

        yaw[0] = yaw[1]
        yaw[-1] = yaw[-2]
        return yaw

    def get_path(self):
        return np.concatenate([self.spline, self.yaw.reshape(1, -1)]).T


class ReferencePath(Path):

    def __init__(self):
        super().__init__()
        self.ref_path = np.zeros([len(self.path), 6])
        self.IDX_X = 0
        self.IDX_Y = 1
        self.IDX_XY = [0, 1]
        self.IDX_XYYAW = [0, 1, 2]
        self.IDX_YAW = 2
        self.IDX_VEL = 3
        self.IDX_CURVATURE = 4
        self.IDX_TIME = 5
        self.IDX_STEER = 3

        self.path_size_scale = 15

        self.scale_path()
        self.insert_curvature()
        self.inser_relativeTime()


    def scale_path(self):
        self.path[:, self.IDX_XY] *= self.path_size_scale
        self.ref_path[:, self.IDX_XYYAW] = self.path[:, self.IDX_XYYAW]

        self.ref_path[:, self.IDX_VEL] = np.ones(len(self.path))*self.mpcconfig.vel_ref

    def insert_curvature(self):
        for i in range(1, len(self.ref_path)-1):
            p1_ = self.ref_path[i-1, self.IDX_XY]
            p2_ = self.ref_path[i, self.IDX_XY]
            p3_ = self.ref_path[i+1, self.IDX_XY]
            A_  = ((p2_[0]-p1_[0]) * (p3_[1] - p1_[1]) - (p2_[1]-p1_[1])*(p3_[0]-p1_[0]))/2
            self.ref_path[i, self.IDX_CURVATURE] = 4*A_ / (np.linalg.norm(p1_-p2_) * np.linalg.norm(p2_-p3_)*np.linalg.norm(p3_-p1_))

    def inser_relativeTime(self):
        for i in range(1, len(self.ref_path)):
            v_ = self.ref_path[i, self.IDX_VEL]
            d_ = np.linalg.norm(self.ref_path[i, self.IDX_XY] - self.ref_path[i-1, self.IDX_XY])
            dt_ = d_ / v_
            self.ref_path[i, self.IDX_TIME] = self.ref_path[i-1, self.IDX_TIME] + dt_

    def get_refpath(self):
        return self.ref_path

