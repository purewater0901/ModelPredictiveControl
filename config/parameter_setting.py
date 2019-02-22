import math
import numpy as np


class Config:
    kmh2ms = 1000 / 3600
    simulation_time = 35
    simulation_time_step = 0.002
    vel_ref = 30 * kmh2ms

    tau = 0.115
    wheelbase = 2.69
    steer_lim = math.radians(30)
    vel_max = 10
    vel_min = -5
    input_delay = 0.05
    control_dt = 0.01
    measurement_noise_stddev = np.array([0.1, 0.1, math.radians(1.0), math.radians(0.5)])
    steering_steady_state_error_deg = 1

    '''
    simulation parameters
    '''
    x0 = np.array([0, 0.5, 0, 0]) # initial value
    ts = 0
    dt = simulation_time_step
    tf = simulation_time
    t = np.arange(ts, tf+dt, dt)


class MPCConfig(Config):
    mpc_dt = 0.05
    mpc_n = 60
    mpc_constraint_steering_deg = 30
    mpc_constraint_steer_rate_deg = 280
    mpc_model_dim = 3
    mpc_Q = np.diag(np.array([1, 2]))
    mpc_R = 0.5
