from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Type

from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.simulation import Simulation

from src.simulation.history.autoregressive_simulation_history_buffer import AutoregressiveSimulationHistoryBuffer

logger = logging.getLogger(__name__)


class AutoregressiveSimulation(Simulation):
    """Inherited, see superclass."""

    def __init__(
        self,
        simulation_setup: SimulationSetup,
        callback: Optional[AbstractCallback] = None,
        simulation_history_buffer_duration: float = 2,
        init_simulation_history_buffer_duration: float = 2,         # 初始化history_buffer时，只初始化20帧
    ):
        super().__init__(simulation_setup, callback, simulation_history_buffer_duration)

        # 模仿nuplan的设计方式
        self._init_simulation_history_buffer_duration = init_simulation_history_buffer_duration + self._scenario.database_interval
        self._init_history_buffer_size = int(self._init_simulation_history_buffer_duration / self._scenario.database_interval) + 1

    def initialize(self) -> PlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scenario，这里，我们只初始化22帧的buffer
        self._history_buffer = AutoregressiveSimulationHistoryBuffer.initialize_from_scenario(    
            buffer_size=self._history_buffer_size,
            init_buffer_size=self._init_history_buffer_size, 
            scenario=self._scenario, 
            observation_type=self._observations.observation_type()
        )

        # Initialize observations，空方法
        self._observations.initialize()

        # Add the current state into the history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

        # 现在history_buffer里有23个场景，我们需要将其减为21个，后续会利用len(history_buffer)，设置window_T
        if self._history_buffer.size == 23:
            self.pop_history_buffer_util()
            self.pop_history_buffer_util()
        elif self._history_buffer.size == 22:   # pop一次
            self.pop_history_buffer_util()
        
        assert self._history_buffer.size == 21, "only support history_buffer = 21"
            
        # Return the planner initialization structure for this simulation
        return PlannerInitialization(
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            mission_goal=self._scenario.get_mission_goal(),
            map_api=self._scenario.map_api,
        )

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, simulation can not be propagated!")

        # Measurements
        iteration = self._time_controller.get_iteration()   # iteration.index：当前迭代到第几帧
        ego_state, observation = self._history_buffer.current_state
        traffic_light_status = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))

        # Add new sample to history队列
        logger.debug(f"Adding to history: {iteration.index}")
        self._history.add_sample(   # 在队列里添加新的生成的80帧的自车轨迹，GT的当前状态的：自车state, 全部agent信息和红绿灯信息
            SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status)
        )
    
        # Propagate state to next iteration，这个controller的类型：nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller.StepSimulationTimeController
        next_iteration = self._time_controller.next_iteration()

        # Propagate state
        if next_iteration:  # nuplan.planning.simulation.controller.two_stage_controller.TwoStageController
            self._ego_controller.update_state(iteration, next_iteration, ego_state, trajectory) # trajectory有81帧，第0帧就是ego_state，这个说白了，就是根据位置估算了ego的速度，加速度这些量
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)  # 根据ego当前状态，更新agent的下一步轨迹
        else:
            self._is_simulation_running = False

        # Append new state into history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())

    # 用来pop history buffer，更早的场景
    def pop_history_buffer_util(self):
        self._history_buffer._ego_state_buffer.popleft()
        self._history_buffer._observations_buffer.popleft()