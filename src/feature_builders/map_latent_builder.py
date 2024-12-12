from typing import List, Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from src.features.map_latent_feature import MapLatentFeature

class MapLatentBuilder(AbstractFeatureBuilder):

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "rvae_latent"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return MapLatentFeature
    
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> MapLatentFeature:
        raise NotImplementedError

    def get_features_from_scenario(self, scenario: AbstractScenario) -> MapLatentFeature:
        raise NotImplementedError