import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


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
        self.animate()

    def simulate(self):

        count = 1
        for _ in tqdm(np.arange(self.ts, self.tf + self.dt, self.dt)):
            if count % self.control_count == 0:
                """
                add input noise
                """
                x_noised = self.x + np.random.rand(len(self.param.x0)) * self.param.measurement_noise_stddev
                self.u = self.controller.calc_input(x_noised)

            """
            add input delay
            計算された最新の入力はすぐには入力されない(遅延が生じているため)
            """
            self.input_buf = np.concatenate([self.u.reshape(1, -1), self.input_buf[0:len(self.input_buf) - 1, :]])
            u_delayed = self.input_buf[len(self.input_buf) - 1, :]  # 最後の部分を実際の入力とする

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
            count += 1

    def animate(self):
        fig = plt.figure()
        ims = []
        data_length = len(self.state_log)
        n = int(data_length / 1000)

        for i in range(1000):
            im = plt.scatter(self.state_log[i * n, 0], self.state_log[i * n, 1], color='blue')
            ims.append([im])

        # アニメーション作成
        ani = animation.ArtistAnimation(fig, ims, interval=1)
        plt.plot(self.ref_path[:, 0], self.ref_path[:, 1], color='red')

        # アニメーションの保存
        ani.save('anim.gif', writer="pillow")

        # 表示
        plt.show()
