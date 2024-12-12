from typing import List, Type

import numpy as np
import shapely
from shapely.geometry import Polygon
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap, PolygonMapObject
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)

from ..features.nuplan_feature import NuplanFeature

class NuplanFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        radius: float = 100,
        history_horizon: float = 2,
        future_horizon: float = 8,
        sample_interval: float = 0.1,
        max_agents: int = 64,
        pose_interval: float = 1.0
    ) -> None:
        super().__init__()

        self.radius = radius
        self.history_horizon = history_horizon
        self.future_horizon = future_horizon
        self.history_samples = int(self.history_horizon / sample_interval)
        self.future_samples = int(self.future_horizon / sample_interval)
        self.sample_interval = sample_interval
        self.ego_params = get_pacifica_parameters()
        self.length = self.ego_params.length
        self.width = self.ego_params.width
        self.max_agents = max_agents
        self.pose_interval = pose_interval    # set the pose sample interval as defined in Sledge

        self.interested_objects_types = [   # In GPD-1, we only restructure ego and vehicle data
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            # TrackedObjectType.PEDESTRIAN,
            # TrackedObjectType.BICYCLE,
        ]

    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return NuplanFeature  # type: ignore

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "feature"

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
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

        # agents features
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=self.history_horizon,
                num_samples=self.history_samples,
            )
        ]
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=self.future_horizon,
                num_samples=self.future_samples,
            )
        ]
        tracked_objects_list = (
            past_tracked_objects + [present_tracked_objects] + future_tracked_objects
        )

        return self._build_feature(
            present_idx=0,  # Set center frame to 0 to extract agents and prevent new cars from being added to the scene.
            ego_state_list=ego_state_list,
            tracked_objects_list=tracked_objects_list,
            route_roadblocks_ids=scenario.get_route_roadblock_ids(),
            map_api=scenario.map_api,
            mission_goal=scenario.get_mission_goal(),
            traffic_light_status=scenario.get_traffic_light_status_at_iteration(0),
        )

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        history = current_input.history
        tracked_objects_list = [
            observation.tracked_objects for observation in history.observations
        ]

        horizon = self.history_samples + 1
        return self._build_feature(
            # present_idx=-1,
            present_idx=0,   # 我们以第0帧为中心进行采样
            ego_state_list=history.ego_states[-horizon:],
            tracked_objects_list=tracked_objects_list[-horizon:],
            route_roadblocks_ids=initialization.route_roadblock_ids,
            map_api=initialization.map_api,
            mission_goal=initialization.mission_goal,
            traffic_light_status=current_input.traffic_light_data,
        )

    def _build_feature(
        self,
        present_idx: int,
        ego_state_list: List[EgoState],
        tracked_objects_list: List[TrackedObjects],
        route_roadblocks_ids: list[int],
        map_api: AbstractMap,
        mission_goal: StateSE2,
        traffic_light_status: List[TrafficLightStatusData] = None,
    ):
        present_ego_state = ego_state_list[present_idx]
        query_xy = present_ego_state.center

        data = {}

        ego_features = self._get_ego_features(ego_states=ego_state_list)
        agent_features = self._get_agent_features(
            query_xy=query_xy,
            present_idx=present_idx,
            tracked_objects_list=tracked_objects_list,
        )

        data["agent"] = {}
        for k in agent_features.keys():
            data["agent"][k] = np.concatenate(
                [ego_features[k][None, ...], agent_features[k]], axis=0
            )
        
        return NuplanFeature.normalize(data, first_time=True, radius=self.radius, cur_step=0)

    def _get_ego_features(self, ego_states: List[EgoState]):
        """note that rear axle velocity and acceleration are in ego local frame,
        and need to be transformed to the global frame.
        """
        T = len(ego_states)

        position = np.zeros((T, 2), dtype=np.float64)
        heading = np.zeros((T), dtype=np.float64)
        # velocity = np.zeros((T, 2), dtype=np.float64)
        # acceleration = np.zeros((T, 2), dtype=np.float64)
        # shape = np.zeros((T, 2), dtype=np.float64)
        valid_mask = np.ones(T, dtype=bool)

        for t, state in enumerate(ego_states):
            position[t] = state.rear_axle.array
            heading[t] = state.rear_axle.heading
            # velocity[t] = rotate_round_z_axis(
            #     state.dynamic_car_state.rear_axle_velocity_2d.array,
            #     -state.rear_axle.heading,
            # )
            # acceleration[t] = rotate_round_z_axis(
            #     state.dynamic_car_state.rear_axle_acceleration_2d.array,
            #     -state.rear_axle.heading,
            # )
            # shape[t] = np.array([self.width, self.length])

        # category = np.array(
        #     self.interested_objects_types.index(TrackedObjectType.EGO), dtype=np.int8
        # )

        return {    # In GPD-1 we only restructure agent's position heading and valid_mask
            "position": position,
            "heading": heading,
            # "velocity": velocity,
            # "acceleration": acceleration,
            # "shape": shape,
            # "category": category,
            "valid_mask": valid_mask,
        }

    def _get_agent_features(
        self,
        query_xy: Point2D,
        present_idx: int,
        tracked_objects_list: List[TrackedObjects],
    ):
        # if present_idx < 0:     # Ｃorrect planTF bug in sumulation
        #     present_idx += len(tracked_objects_list)

        present_tracked_objects = tracked_objects_list[present_idx]
        present_agents = present_tracked_objects.get_tracked_objects_of_types(
            self.interested_objects_types
        )
        N, T = min(len(present_agents), self.max_agents), len(tracked_objects_list)

        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        # velocity = np.zeros((N, T, 2), dtype=np.float64)
        # shape = np.zeros((N, T, 2), dtype=np.float64)
        # category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=bool)

        if N == 0:
            return {
                "position": position,
                "heading": heading,
                # "velocity": velocity,
                # "shape": shape,
                # "category": category,
                "valid_mask": valid_mask,
            }

        agent_ids = np.array([agent.track_token for agent in present_agents])
        agent_cur_pos = np.array([agent.center.array for agent in present_agents])
        distance = np.linalg.norm(agent_cur_pos - query_xy.array[None, :], axis=1)
        agent_ids_sorted = agent_ids[np.argsort(distance)[: self.max_agents]]
        agent_ids_sorted = {agent_id: i for i, agent_id in enumerate(agent_ids_sorted)}
        
        for t, tracked_objects in enumerate(tracked_objects_list):  
            for agent in tracked_objects.get_tracked_objects_of_types(
                self.interested_objects_types
            ):
                if agent.track_token not in agent_ids_sorted:
                    continue

                idx = agent_ids_sorted[agent.track_token]
                position[idx, t] = agent.center.array
                heading[idx, t] = agent.center.heading
                # velocity[idx, t] = agent.velocity.array
                # shape[idx, t] = np.array([agent.box.width, agent.box.length])
                valid_mask[idx, t] = True

                # if t == present_idx:
                #     category[idx] = self.interested_objects_types.index(
                #         agent.tracked_object_type
                #     )

        return {
            "position": position,
            "heading": heading,
            # "velocity": velocity,
            # "shape": shape,
            # "category": category,
            "valid_mask": valid_mask,
        }

    def calculate_additional_ego_states(
        self, current_state: EgoState, prev_state: EgoState, dt=0.1
    ):
        cur_velocity = current_state.dynamic_car_state.rear_axle_velocity_2d.x
        angle_diff = current_state.rear_axle.heading - prev_state.rear_axle.heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = angle_diff / 0.1

        if abs(cur_velocity) < 0.2:
            return 0.0, 0.0  # if the car is almost stopped, the yaw rate is unreliable
        else:
            steering_angle = np.arctan(
                yaw_rate * self.ego_params.wheel_base / abs(cur_velocity)
            )
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

            return steering_angle, yaw_rate


