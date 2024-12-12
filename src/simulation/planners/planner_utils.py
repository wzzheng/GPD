from typing import Deque

import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _get_velocity_and_acceleration,
    _se2_vel_acc_to_ego_state,
)


def global_trajectory_to_states(
    global_trajectory: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    include_ego_state: bool = True,
):
    ego_state = ego_history[-1]     # 第21帧车的状态
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    global_states = [StateSE2.deserialize(pose) for pose in global_trajectory]

    velocities, accelerations = _get_velocity_and_acceleration( # 通过导数和planning的结果估算出来的
        global_states, ego_history, timesteps
    )
    agent_states = [    # 将上述信息，实例化为一个ego state类
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            global_states, velocities, accelerations, timesteps
        )
    ]

    if include_ego_state:
        agent_states.insert(0, ego_state)   # 加入第21帧车的状态，一共返回81帧状态

    return agent_states 


def load_checkpoint(checkpoint: str):
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    
    # 删除不需要的键，这些键是用于快速抽取agent loss，存在register
    keys_to_ignore = ['agent_feature_indices', 'agent_pred_feature_indices']
    for key in keys_to_ignore:
        if key in ckpt["state_dict"]:
            del ckpt["state_dict"][key]
        
    state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    return state_dict
