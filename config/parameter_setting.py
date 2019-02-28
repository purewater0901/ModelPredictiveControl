import math
import numpy as np


class Config:

    def __init__(self):
        self.kmh2ms = 1000 / 3600
        self.simulation_time = 35
        self.simulation_time_step = 0.002
        self.vel_ref = 30 * self.kmh2ms

        self.tau = 0.115
        self.wheelbase = 2.69
        self.steer_lim = math.radians(30)
        self.vel_max = 10
        self.vel_min = -5
        self.input_delay = 0.05
        self.control_dt = 0.01
        self.measurement_noise_stddev = np.array([0.1, 0.1, math.radians(1.0), math.radians(0.5)])
        self.steering_steady_state_error_deg = 1

        '''
        model parameter
        '''
        self.model_name = 'kinematics'
        self.state_dim = 3
        self.output_dim = 2

        '''
        simulation parameters
        '''
        self.x0 = np.array([0.05, 0.5, 0, 0])  # initial value(x, y, yaw, delta)
        self.ts = 0
        self.dt = self.simulation_time_step
        self.tf = self.simulation_time
        self.t = np.arange(self.ts, self.tf+self.dt, self.dt)


class MPCConfig(Config):
    def __init__(self):
        super().__init__()
        self.mpc_dt = 0.05
        self.mpc_n = 60
        self.mpc_constraint_steering_deg = 30
        self.mpc_constraint_steer_rate_deg = 280
        self.mpc_model_dim = 3
        self.mpc_Q = np.diag(np.array([1, 2]))
        self.mpc_R = 0.5
        self.mpc_solve_without_constraint = False
        self.mpc_sensor_delay = self.input_delay
