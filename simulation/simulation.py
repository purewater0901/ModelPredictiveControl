import numpy as np
import matplotlib.pyplot as plt

class Simulation:

    def __init__(self, model, controller, ref_path, param):
        self.param = param()
        self.ref_path = ref_path().ref_path
        self.model = model(self.param)
        self.controller = controller(self.ref_path, self.param)
        self.x = self.param.x0
        self.ts = self.param.ts
        self.tf = self.param.tf
        self.dt = self.param.dt
        self.input_delay = self.param.input_delay
        self.delay_count = round(self.input_delay / self.dt)
        self.control_dt = self.param.control_dt
        self.control_count = round(self.control_dt / self.dt)

        self.t_vec = np.arange(self.ts, self.tf+self.dt, self.dt)
        self.state_log = np.zeros([len(self.t_vec), len(self.x)])

        tmp_u = self.controller.calc_input(self.x)

        self.input_log = np.zeros([len(self.t_vec), len(tmp_u)])
        self.input_buf = np.zeros([self.delay_count, len(tmp_u)])
        self.u = np.zeros(len(tmp_u))  # はじめの入力は0
