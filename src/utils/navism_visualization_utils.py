from typing import Any, Callable, List, Tuple
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.bev import add_linestring_to_bev_ax, add_polygon_to_bev_ax, add_oriented_box_to_bev_ax
from shapely.geometry import Polygon, LineString
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from navsim.visualization.config import (
    BEV_PLOT_CONFIG,
    MAP_LAYER_CONFIG,
    AGENT_CONFIG,
)
from src.utils.sledge_utils.pdm_path import PDMPath
from src.utils.sledge_utils.pdm_array_representation import array_to_states_se2
from src.utils.sledge_utils.utils import find_consecutive_true_indices

def add_trajectory_to_bev_ax(ax: plt.Axes, trajectory: np.array, config: Dict[str, Any]) -> plt.Axes:
    """
    Add trajectory poses as lint to plot
    :param ax: matplotlib ax object
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plot parameters
    :return: ax with plot
    """
    ax.plot(
        trajectory[:, 1],
        trajectory[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        markeredgecolor=config["marker_edge_color"],
        zorder=config["zorder"],
    )
    return ax


def add_annotations_to_bev_ax(
    ax: plt.Axes, annotations: list, add_ego: bool = True
) -> plt.Axes:
    """
    Adds birds-eye-view visualization of annotations (ie. bounding boxes)
    :param ax: matplotlib ax object
    :param annotations: navsim annotations dataclass
    :param add_ego: boolean weather to add ego bounding box, defaults to True
    :return: ax with plot
    """   

    for annotation in annotations:
        agent_type = annotation["name_value"]
        box_value = annotation["box_value"]

        x, y, heading = (
            box_value[0],
            box_value[1],
            box_value[2],
        )
        box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
        agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)

        add_oriented_box_to_bev_ax(ax, agent_box, AGENT_CONFIG[agent_type])

    if add_ego:
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(0, 0, 0),
            vehicle_parameters=get_pacifica_parameters(),
        )
        add_oriented_box_to_bev_ax(
            ax, car_footprint.oriented_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=False
        )
    return ax

def add_map_to_bev_ax(ax: plt.Axes, lines) -> plt.Axes:
    """
    Adds birds-eye-view visualization of map (ie. polygons / lines)
    TODO: add more layers for visualizations (or flags in config)
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param origin: (x,y,θ) dataclass of global ego frame
    :return: ax with plot
    """    

    for line in lines:
        # 绘制路的背景
        linestring = LineString(line[line.any(-1)])
        add_linestring_to_bev_ax(
            ax, linestring, 
            {
                # "line_color": "#D3D3D3",  # 原
                "line_color": "#F2F2F2",
                "line_color_alpha": 1.0,
                "line_width": 20.0,
                "line_style": "-",
                "zorder": 1,
                "antialiased": False
            }
        )

    for line in lines:
        # 绘制线条
        linestring = LineString(line[line.any(-1)])
        add_linestring_to_bev_ax(
            ax, linestring, MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]
        )
    return ax

def add_configured_bev_on_ax(ax: plt.Axes, lines, annotations: list) -> plt.Axes:
    """
    Adds birds-eye-view visualization optionally with map, annotations, or lidar
    :param ax: matplotlib ax object
    :param map_api: nuPlans map interface
    :param frame: navsim frame dataclass
    :return: ax with plot
    """    
    add_map_to_bev_ax(ax, lines)
    add_annotations_to_bev_ax(ax, annotations)

    return ax

def get_rotate_mat(center_angle):
    return np.array(  # [2, 2]
        [
            [np.cos(center_angle), -np.sin(center_angle)],
            [np.sin(center_angle), np.cos(center_angle)],
        ],
        dtype=np.float64,
    )

def get_compress_map(lines):
    lines = lines[lines.any((-1, -2))]    # 去除全0的line
    out_lines = []
    max_length = 0
    for line in lines:
        # 保存line中连续可见的线段片段，压缩line
        line_mask = line.any(-1)
        indices_segments = find_consecutive_true_indices(line_mask)
        for indices_segment in indices_segments:
            line_segment = line[indices_segment]
            if len(line_segment) < 2:
                continue
            out_lines.append(line_segment)
            max_length = len(line_segment) if len(line_segment) > max_length else max_length

    # 定义填充函数
    def pad_array(arr, max_length):
        pad_width = max_length - arr.shape[0]
        return np.pad(arr, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    padded_arrays = [pad_array(arr, max_length) for arr in out_lines]
    
    return np.stack(padded_arrays, axis=0)



def get_fixed_and_new_map(map_T, map_T1, origin_pose, visible_radius = 32):
    """
    全部都是可见的map line
    map_T: [M, 20, 2]
    map_T1: [M, 20, 2]
    origin_pose: [3], 第T+1帧ego在第T帧ego坐标系下的位置, x,y,heading
    """
    # 1. 将 T+1 帧 map，转换到 T 帧
    rotate_mat = get_rotate_mat(- origin_pose[2])   # 这里加一个负号就是取逆了

    map_T1_to_T = np.matmul(
        map_T1, rotate_mat[np.newaxis]
    ) + origin_pose[np.newaxis, np.newaxis, :2]

    # 2.0 先对转换后的点，进行差值，避免这条路只生成了一个点，没办法画到图上
    num_lines, P, _ = map_T1_to_T.shape
    num_line_poses = 20000
    vector_states = np.zeros((num_lines, num_line_poses, 2), dtype=np.float32)
    map_T1_to_T = np.pad(map_T1_to_T, ((0, 0), (0, 0), (0, 1)), constant_values=0, mode='constant')       # 填充角度，方便后续插值

    for line_idx, line in enumerate(map_T1_to_T):
        path = PDMPath(array_to_states_se2(line))
        distances = np.linspace(0, path.length, num=num_line_poses, endpoint=True)
        vector_states[line_idx] = path.interpolate(distances, as_array=True)[..., :2]
    map_T1_to_T = vector_states

    # 2.1 用 T 帧 ego为中心，获得64×64方格外，新生成的点
    T1_mask = (np.abs(map_T1_to_T) >= visible_radius).any(-1)
    map_T1_to_T[~T1_mask] = 0
    map_T1_to_T = get_compress_map(map_T1_to_T) # 获取到有效的map line
    
    # 3. 将 T+1 帧的可见点，和 T 帧的可见点拼到一起
    len_T1 = map_T1_to_T.shape[1]
    len_T = map_T.shape[1]
    max_length = len_T1 if len_T1 > len_T else len_T
    map_T1_and_T = np.concatenate([
        np.pad(map_T1_to_T, ((0, 0), (0, max_length - len_T1), (0, 0)), constant_values=0, mode='constant'), 
        np.pad(map_T, ((0, 0), (0, max_length - len_T), (0, 0)), constant_values=0, mode='constant')
    ], axis=0)

    # 4. 转换到第帧坐标系下
    rotate_mat = get_rotate_mat(origin_pose[2])
    concat_map = np.matmul(
        map_T1_and_T - origin_pose[np.newaxis, np.newaxis, :2], 
        rotate_mat[np.newaxis]
    )

    map_zero_indices = (map_T1_and_T == 0)
    concat_map[map_zero_indices] = 0    # 避免坐标转换导致非0值改变
    return concat_map


def plot_bev_frame_with_traj(
    map_line_feature, annotations: list, ego_local_trajectory: np.array = [], gt_trajectory: np.array = [], 
    is_fixed_visual = False, 
    is_visual_multi_agent_traject = False,
    idx = 0,
    agent_trajectorys = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    is_fixed_visual: 是否要使用固定已经出现部分的可视化
    idx: 当前可视化到第几帧了
    agent_trajectorys: 可视化每一帧agent的轨迹
    """
    if is_fixed_visual:
        global last_frame_map
        global last_ego_origin
        if idx > 0:    # 第0帧直接画，不需要转化
            map_line_feature = get_fixed_and_new_map(last_frame_map, map_line_feature, last_ego_origin)

        last_frame_map = map_line_feature   # 记录上一帧map
        last_ego_origin = ego_local_trajectory[idx + 1] if idx + 1 < 80 else None   # 记录第T+1帧ego在T帧ego为中心坐标系下的坐标

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, map_line_feature, annotations)
    if len(ego_local_trajectory) > 0:
        add_trajectory_to_bev_ax(ax, ego_local_trajectory, TRAJECTORY_CONFIG["ego"])
    if len(gt_trajectory) > 0:
        add_trajectory_to_bev_ax(ax, gt_trajectory, TRAJECTORY_CONFIG["human"])
    if is_visual_multi_agent_traject:
        for agent_traj in agent_trajectorys[1:, idx]:
            non_zero_agent_traj = agent_traj[agent_traj.any(-1)]
            add_trajectory_to_bev_ax(ax, non_zero_agent_traj, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

def plot_bev_frame_with_traj_only_input_gt(
    map_line_feature, annotations: list, ego_local_trajectory: np.array = [], gt_trajectory: np.array = [], 
    is_fixed_visual = False, 
    is_visual_multi_agent_traject = False,
    idx = 0,
    agent_trajectorys = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    is_fixed_visual: 是否要使用固定已经出现部分的可视化
    idx: 当前可视化到第几帧了
    agent_trajectorys: 可视化每一帧agent的轨迹
    """

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])

    add_map_to_bev_ax(ax, map_line_feature)
    car_footprint = CarFootprint.build_from_rear_axle(
        rear_axle_pose=StateSE2(0, 0, 0),
        vehicle_parameters=get_pacifica_parameters(),
    )
    add_oriented_box_to_bev_ax(
        ax, car_footprint.oriented_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=False
    )
    # add_annotations_to_bev_ax(ax, annotations, add_ego=True)

    if len(ego_local_trajectory) > 0:
        add_trajectory_to_bev_ax(ax, ego_local_trajectory, TRAJECTORY_CONFIG["ego"])
    # if is_visual_multi_agent_traject:
    #     for agent_traj in agent_trajectorys[1:, idx]:
    #         non_zero_agent_traj = agent_traj[agent_traj.any(-1)]
    #         add_trajectory_to_bev_ax(ax, non_zero_agent_traj, TRAJECTORY_CONFIG["agent"])
            
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax

def plot_bev_frame_with_traj_only_black(
    map_line_feature, annotations: list, ego_local_trajectory: np.array = [], gt_trajectory: np.array = [], 
    is_fixed_visual = False, 
    is_visual_multi_agent_traject = False,
    idx = 0,
    agent_trajectorys = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    is_fixed_visual: 是否要使用固定已经出现部分的可视化
    idx: 当前可视化到第几帧了
    agent_trajectorys: 可视化每一帧agent的轨迹
    """

    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])            
    configure_bev_ax(ax)
    configure_ax(ax)

    return fig, ax