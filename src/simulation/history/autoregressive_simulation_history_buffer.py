from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple, Type

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation, Sensors
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer


class AutoregressiveSimulationHistoryBuffer(SimulationHistoryBuffer):
    """Inherited, see superclass."""

    @classmethod
    def initialize_from_list(
        cls,
        buffer_size: int,
        init_buffer_size: int,
        ego_states: List[EgoState],
        observations: List[Observation],
        sample_interval: Optional[float] = None,
    ) -> AutoregressiveSimulationHistoryBuffer:
        """
        :param init_buffer_size: 初始化只取21帧
        """
        ego_state_buffer: Deque[EgoState] = deque(ego_states[-init_buffer_size:], maxlen=buffer_size)    # 一个双端队列，到达最大长度，就会先进先出
        observations_buffer: Deque[Observation] = deque(observations[-init_buffer_size:], maxlen=buffer_size)

        return cls(
            ego_state_buffer=ego_state_buffer, observations_buffer=observations_buffer, sample_interval=sample_interval
        )

    @staticmethod
    def initialize_from_scenario(
        buffer_size: int, init_buffer_size: int, scenario: AbstractScenario, observation_type: Type[Observation]
    ) -> AutoregressiveSimulationHistoryBuffer:
        """
        :param init_buffer_size: 初始化只取21帧
        """
        buffer_duration = init_buffer_size * scenario.database_interval

        if observation_type == DetectionsTracks:
            observation_getter = scenario.get_past_tracked_objects
        elif observation_type == Sensors:
            observation_getter = scenario.get_past_sensors
        else:
            raise ValueError(f"No matching observation type for {observation_type} for history!")

        past_observation = list(observation_getter(iteration=0, time_horizon=buffer_duration, num_samples=init_buffer_size))     # 从DB中得到历史22帧中，每一帧中的所有agent信息

        past_ego_states = list(
            scenario.get_ego_past_trajectory(iteration=0, time_horizon=buffer_duration, num_samples=init_buffer_size)    # 从DB中的到22帧ego的轨迹
        )

        return AutoregressiveSimulationHistoryBuffer.initialize_from_list(    # 将list变为deque，返回SimulationHistoryBuffer
            buffer_size=buffer_size,
            init_buffer_size=init_buffer_size,
            ego_states=past_ego_states,
            observations=past_observation,
            sample_interval=scenario.database_interval,
        )
