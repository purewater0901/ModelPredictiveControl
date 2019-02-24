import numpy as np
import matplotlib.pyplot as plt


class Simulation:

    def __init__(self, model, controller, ref_path, param):
        self.param = param()
        self.ref_path = ref_path(self.param).ref_path
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

        self.t_vec = np.arange(self.ts, self.tf + self.dt, self.dt)
        self.state_log = np.zeros([len(self.t_vec), len(self.x)])

        tmp_u = self.controller.calc_input(self.x)

        self.input_log = np.zeros([len(self.t_vec), len(tmp_u)])
        self.input_buf = np.zeros([self.delay_count, len(tmp_u)])
        self.u = np.zeros(len(tmp_u))  # はじめの入力は0
        self.simulate()
        #ref_path(self.param).plot()

    def simulate(self):

        count = 1
        for t in np.arange(self.ts, self.tf + self.dt, self.dt):
            if count % self.control_count == 0:
                """
                add input noise
                """
                x_noised = self.x + np.random.rand(len(self.param.x0)) * self.param.measurement_noise_stddev
                self.u = self.controller.calc_input(x_noised)

            """
            add input delay
            """
            self.input_buf = np.concatenate([self.u.reshape(1, -1), self.input_buf[0:len(self.input_buf) - 1, :]])
            u_delayed = self.input_buf[len(self.input_buf) - 1, :]

            """
            ルンゲクッタ
            """
            k1 = self.model.update(self.x, u_delayed)
            k2 = self.model.update(self.x + k1 * self.dt / 2, u_delayed)
            k3 = self.model.update(self.x + k2 * self.dt / 2, u_delayed)
            k4 = self.model.update(self.x + k3 * self.dt, u_delayed)
            self.x = self.x + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6

            """
            save data
            """
            self.state_log[count - 1, :] = self.x
            self.input_log[count - 1, :] = self.u.reshape(1, -1)

            if count % 1000 == 0:
                plt.plot(self.state_log[0:count, 0], self.state_log[0:count, 1], c='b', label='motion line')
                plt.plot(self.ref_path[:, 0], self.ref_path[:, 1], c='r', label='cubic spline')
                plt.show()

            count += 1
