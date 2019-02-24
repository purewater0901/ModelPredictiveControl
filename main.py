from config.parameter_setting import MPCConfig
from steer_vehicle_model.kinematics_model import Kinematics_Model
from controller.model_predictive_controller import MPCController
from simulation.simulation import Simulation
from path_design.create_path import ReferencePath

if __name__ == '__main__':
    Simulation(Kinematics_Model, MPCController, ReferencePath, MPCConfig)




