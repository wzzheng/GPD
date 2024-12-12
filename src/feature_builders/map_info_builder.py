from typing import List, Type
import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from src.features.map_info_feature import MapInfoFeature
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.actor_state.ego_state import EgoState
from .sledge_components.sledge_line_feature import compute_line_features

class MapInfoBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        radius: float = 150,
        pose_interval: float = 1.0,
        history_horizon: float = 2,
        sample_interval: float = 0.1,
        future_horizon: float = 8,
    ) -> None:
        super().__init__()
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.radius = radius
        self.pose_interval = pose_interval    # set the pose sample interval
        self.history_samples = int(self.history_horizon / sample_interval)
        self.future_samples = int(self.future_horizon / sample_interval)

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "map_cache"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return MapInfoFeature
    
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> MapInfoFeature:
        history = current_input.history

        horizon = self.history_samples + 1
        return self._build_feature(
            present_idx=0,   # 我们以第0帧为中心进行采样
            ego_state_list=history.ego_states[-horizon:],
            map_api=initialization.map_api
        )


    def get_features_from_scenario(self, scenario: AbstractScenario) -> MapInfoFeature:
        ego_cur_state = scenario.initial_ego_state

        # ego features
        past_ego_trajectory = scenario.get_ego_past_trajectory(
            iteration=0,
            time_horizon=self.history_horizon,
            num_samples=self.history_samples,
        )
        future_ego_trajectory = scenario.get_ego_future_trajectory(
            iteration=0,
            time_horizon=self.future_horizon,
            num_samples=self.future_samples,
        )
        ego_state_list = (
            list(past_ego_trajectory) + [ego_cur_state] + list(future_ego_trajectory)
        )

        return self._build_feature(
            present_idx=0,  # Set center frame to 0 to extract agents and prevent new cars from being added to the scene.
            ego_state_list=ego_state_list,
            map_api=scenario.map_api
        )

    
    def _build_feature(
        self,
        present_idx: int,
        ego_state_list: List[EgoState],
        map_api: AbstractMap,
    ):

        data = {}
        # convert map to the coordinate of each ego center
        data['ego'] = self._get_ego_features(ego_states=ego_state_list)

        # retrieve the map data as defined in Sledge
        data["map"] = self._get_concat_polygon(
            ego_state=ego_state_list[present_idx],
            map_api=map_api
        )

        return MapInfoFeature(data)


    def _get_ego_features(self, ego_states: List[EgoState]):
        """note that rear axle velocity and acceleration are in ego local frame,
        and need to be transformed to the global frame.
        """
        T = len(ego_states)

        position = np.zeros((T, 2), dtype=np.float64)
        heading = np.zeros((T), dtype=np.float64)

        for t, state in enumerate(ego_states):
            position[t] = state.rear_axle.array
            heading[t] = state.rear_axle.heading

        return {
            "position": position,
            "heading": heading,
        }
    
    def _get_concat_polygon(
        self,
        ego_state: EgoState,
        map_api: AbstractMap
    ):
        """
        Compute raw vector feature for map VQ-VAE training.
        :param ego_state: object of ego vehicle state in nuPlan
        :param map_api: object of map in nuPlan
        :return: Dict
        """

        lines = compute_line_features(
            ego_state,
            map_api,
            self.radius,
            self.pose_interval,
        )

        return {
            "lines": {
                "states": lines.states, 
                "mask": lines.mask
            }
        }