from typing import Type
import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from sledge.autoencoder.preprocessing.features.empty_feature import EmptyFeature

class EmptyTargetBuilder(AbstractTargetBuilder):
    """
    The target of GPD is derived from the original feature. 
    This empty target builder is beneficial as it helps avoid significant code alterations 
    and reserves space for future expansion.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "empty_feature"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return EmptyFeature

    def get_targets(self, scenario: AbstractScenario) -> EmptyFeature:
        """Inherited, see superclass."""
        return EmptyFeature(np.array(1))
