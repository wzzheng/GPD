from typing import List, Type
from shapely.geometry import Polygon

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, SemanticMapLayer
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)

from sledge.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from sledge.autoencoder.preprocessing.features.nuplan_feature import NuplanFeature


class FeatureBuilder(AbstractFeatureBuilder):

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "map_cache"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return NuplanFeature
    
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> NuplanFeature:
        raise NotImplementedError

    def get_features_from_scenario(self, scenario: AbstractScenario) -> NuplanFeature:
        raise NotImplementedError