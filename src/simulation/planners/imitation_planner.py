import time
from typing import List, Optional, Type

import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from src.feature_builders.common.utils import rotate_round_z_axis

from .planner_utils import global_trajectory_to_states
from src.utils.simple_utils import mask_feature, agent_padding
from src.features.map_info_feature import MapInfoFeature
from src.features.nuplan_feature import NuplanFeature
from src.models.sledge.rvae_encode_to_latent import RVAEEncodeToLatentModel
from src.models.sledge.rvae_config import RVAEConfig

class ImitationPlanner(AbstractPlanner):
    """
    Long-term IL-based trajectory planner, with short-term RL-based trajectory tracker.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        planner: TorchModuleWrapper,
        planner_ckpt: str = None,
        replan_interval: int = 1,
        use_gpu: bool = True,
        map_encoder = RVAEEncodeToLatentModel(RVAEConfig()),    # 直接写死，map后续也不会变了
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self._map_encoder = map_encoder
        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt
        self._initialization: Optional[PlannerInitialization] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1

        self._replan_interval = replan_interval
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self._start_time = None
        self.max_agent_num = 33

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        # if self._planner_ckpt is not None:    我们已经提前load好了
        #     self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval()
        self._map_encoder = self._map_encoder.to(self.device)
        self._planner = self._planner.to(self.device)
        self._initialization = initialization

        # just to trigger numba compile, no actually meaning
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _planning(self, current_input: PlannerInput):
        self._start_time = time.perf_counter()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        input_feature = self._data_prepare(planner_feature)
        self._feature_building_runtimes.append(time.perf_counter() - self._start_time)

        _, local_trajectory = self._planner.forward(input_feature)
        local_trajectory = torch.cat(  # 删去bs维度
            [
                local_trajectory[0, :, :2],
                torch.atan2(local_trajectory[0, :, 3], local_trajectory[0, :, 2]).unsqueeze(-1)
            ], dim=-1
        )
        # local_trajectory = local_trajectory[0] 
        return local_trajectory.cpu().numpy().astype(np.float64)

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        # ego_state = current_input.history.ego_states[-1]
        ego_state = current_input.history.ego_states[-21]   # 这里我们以第0帧为中心建模，还原也是第0帧

        if self._last_plan_elapsed_step >= self._replan_interval:
            local_trajectory = self._planning(current_input)
            self._global_trajectory = self._get_global_trajectory(
                local_trajectory, ego_state
            )
            self._last_plan_elapsed_step = 0
        else:
            self._global_trajectory = self._global_trajectory[1:]

        trajectory = InterpolatedTrajectory(
            trajectory=global_trajectory_to_states(
                global_trajectory=self._global_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(self._global_trajectory) * self._step_interval,
                step_interval=self._step_interval,
            )
        )

        self._inference_runtimes.append(time.perf_counter() - self._start_time)

        self._last_plan_elapsed_step += 1

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        return report

    def _get_global_trajectory(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        global_position = (
            rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )

        return global_trajectory

    def _data_prepare(self, planner_feature):

        map_feature = MapInfoFeature.deserialize(planner_feature.data, return_vector=False, retrun_raster=True)
        map_raster = map_feature.to_feature_tensor().to_device(self.device).data['rasters']
        map_feature = self._map_encoder(map_raster)

        agent_feature = NuplanFeature.deserialize({"agent": planner_feature.data['agent']})
        agent_data = agent_feature.collate(
            [agent_feature.to_feature_tensor().to_device(self.device)]
        )

        agent_data = agent_data.data['agent']
        agent_position = agent_data['position']
        agent_heading = agent_data['heading']
        agent_mask = agent_data['valid_mask']

        # 对agent进行一次掩码
        agent_position = mask_feature(agent_position, agent_mask)
        agent_heading = mask_feature(agent_heading, agent_mask)

        # padding agent to 33
        agent_position = agent_padding(agent_position, self.max_agent_num)
        agent_heading = agent_padding(agent_heading, self.max_agent_num)
        agent_mask = agent_padding(agent_mask, self.max_agent_num)


        return {   # 组装并添加bs维度
            'agent': {
                'position': agent_position,
                'heading': agent_heading,
                'valid_mask': agent_mask
            },
            'map_latent': map_feature.unsqueeze(0)
        }


