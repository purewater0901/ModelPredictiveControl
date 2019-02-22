import numpy as np
import matplotlib.pyplot as plt
from path_design.create_path import ReferencePath
from config.parameter_setting import MPCConfig
from steer_vehicle_model.kinematics_model import Kinematics_Model
from controller.model_predictive_controller import MPCController

parameters = MPCConfig()
ref= ReferencePath().get_refpath()
model = Kinematics_Model(parameters)
controller = MPCController(ref, parameters)

initial_pos = parameters.x0
controller.calc_error(initial_pos)
controller.create_matrix()


