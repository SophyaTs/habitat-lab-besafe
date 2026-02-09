from habitat.config import DictConfig, read_write
from habitat.config.default_structured_configs import (
    SuccessMeasurementConfig, 
    DistanceToGoalMeasurementConfig,
    RendererConfig,
    ObjectGoalSensorConfig)
import magnum as mn

def patch_task_config(config: DictConfig) -> None:
    # Patch config 
    with read_write(config):
        config.habitat.task.measurements.Success = SuccessMeasurementConfig(type="Success", success_distance  = 0.7)
        config.habitat.task.measurements.DistanceToGoal = DistanceToGoalMeasurementConfig(type="DistanceToGoal", distance_to  = "VIEW_POINTS")
        config.habitat.task.lab_sensors.sensor1 = ObjectGoalSensorConfig()
        config.habitat.simulator.renderer = RendererConfig()