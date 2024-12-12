import copy
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from src.bean.sledge_vector_feature import SledgeVectorRaw, SledgeVector, SledgeVectorElement
from src.bean.sledge_raster_feature import SledgeRasterIndex
from src.utils.sledge_utils.pdm_path import PDMPath
from src.utils.sledge_utils.pdm_array_representation import array_to_states_se2

class SimpleConfig:
    # 记录sledge必要的配置
    num_lines = 50
    pixel_size = 0.25
    frame = (64, 64)
    line_dots_radius = 0
    num_line_poses = 20

    @property
    def pixel_frame(self) -> Tuple[int, int]:
        frame_width, frame_height = self.frame
        return int(frame_width / self.pixel_size), int(frame_height / self.pixel_size)

def coords_in_frame(coords: npt.NDArray[np.float32], frame: Tuple[float, float]) -> npt.NDArray[np.bool_]:
    """
    Checks which coordinates are within the given 2D frame extend.
    :param coords: coordinate array in numpy (x,y) in last axis
    :param frame: tuple of frame extend in meter
    :return: numpy array of boolean's
    """
    assert coords.shape[-1] == 2, "Coordinate array must have last dim size of 2 (ie. x,y)"
    width, height = frame

    within_width = np.logical_and(-width / 2 <= coords[..., 0], coords[..., 0] <= width / 2)
    within_height = np.logical_and(-height / 2 <= coords[..., 1], coords[..., 1] <= height / 2)

    return np.logical_and(within_width, within_height)

def pixel_in_frame(pixel: npt.NDArray[np.int32], pixel_frame: Tuple[int, int]) -> npt.NDArray[np.bool_]:
    """
    Checks if pixels indices are within the image.
    :param pixel: pixel indices as numpy array
    :param pixel_frame: tuple of raster width and height
    :return: numpy array of boolean's
    """
    assert pixel.shape[-1] == 2, "Coordinate array must have last dim size of 2 (ie. x,y)"
    pixel_width, pixel_height = pixel_frame

    within_width = np.logical_and(0 <= pixel[..., 0], pixel[..., 0] < pixel_width)
    within_height = np.logical_and(0 <= pixel[..., 1], pixel[..., 1] < pixel_height)

    return np.logical_and(within_width, within_height)


def find_consecutive_true_indices(mask: npt.NDArray[np.bool_]) -> List[npt.NDArray[np.int32]]:
    """
    Helper function for line preprocessing.
    For example, lines might exceed or return into frame.
    Find regions in mask where line is consecutively in frame (ie. to split line)

    :param mask: 1D numpy array of booleans
    :return: List of int32 arrays, where mask is consecutively true.
    """

    padded_mask = np.pad(np.asarray(mask), (1, 1), "constant", constant_values=False)

    changes = np.diff(padded_mask.astype(int))
    starts = np.where(changes == 1)[0]  # indices of False -> True
    ends = np.where(changes == -1)[0]  # indices of True -> False

    return [np.arange(start, end) for start, end in zip(starts, ends)]

def coords_to_pixel(
    coords: npt.NDArray[np.float32], frame: Tuple[float, float], pixel_size: float
) -> npt.NDArray[np.int32]:
    """
    Converts ego-centric coordinates into pixel coordinates (ie. indices)
    :param coords: coordinate array in numpy (x,y) in last axis
    :param frame: tuple of frame extend in meter
    :param pixel_size: size of a pixel
    :return: indices of pixel coordinates
    """
    assert coords.shape[-1] == 2

    width, height = frame
    pixel_width, pixel_height = int(width / pixel_size), int(height / pixel_size)
    pixel_center = np.array([[pixel_width / 2.0, pixel_height / 2.0]])
    coords_idcs = (coords / pixel_size) + pixel_center

    return coords_idcs.astype(np.int32)

def process_lines_to_vector(
    lines: SledgeVectorElement, config = SimpleConfig()
):
    num_lines = config.num_lines
    
    # 1. preprocess lines (e.g. check if in frame)
    lines_in_frame = []
    for line_states, line_mask in zip(lines.states, lines.mask):
        line_in_mask = line_states[line_mask]  # (n, 3)
        if len(line_in_mask) < 2:
            continue

        path = PDMPath(array_to_states_se2(line_in_mask))
        distances = np.arange(
            0,
            path.length + config.pixel_size,
            config.pixel_size,
        )
        line = path.interpolate(distances, as_array=True)
        frame_mask = coords_in_frame(line[..., :2], config.frame)   # 判断当前polygon，哪些点都在画布中
        indices_segments = find_consecutive_true_indices(frame_mask)    # 找出连续在画布内的部分

        for indices_segment in indices_segments:
            line_segment = line[indices_segment]
            if len(line_segment) < 3:
                continue
            lines_in_frame.append(line_segment) # 这个里面的点，都是画布中可见的

    # sort out nearest num_lines elements
    lines_distances = [np.linalg.norm(line[..., :2], axis=-1).min() for line in lines_in_frame]
    lines_in_frame = [lines_in_frame[idx] for idx in np.argsort(lines_distances)[:num_lines]]

    # 3. vectorized preprocessed lines
    vector_states = np.zeros((num_lines, config.num_line_poses, 2), dtype=np.float32)
    vector_labels = np.zeros((num_lines), dtype=bool)
    vector_labels[: len(lines_in_frame)] = True

    for line_idx, line in enumerate(lines_in_frame):
        path = PDMPath(array_to_states_se2(line))
        distances = np.linspace(0, path.length, num=config.num_line_poses, endpoint=True)
        vector_states[line_idx] = path.interpolate(distances, as_array=True)[..., :2]

    return vector_states, vector_labels

def process_lines_to_raster(
    lines: SledgeVectorElement, config = SimpleConfig()
):
    num_lines = config.num_lines
    
    # 1. preprocess lines (e.g. check if in frame)
    lines_in_frame = []
    for line_states, line_mask in zip(lines.states, lines.mask):
        line_in_mask = line_states[line_mask]  # (n, 3)
        if len(line_in_mask) < 2:
            continue

        path = PDMPath(array_to_states_se2(line_in_mask))
        distances = np.arange(
            0,
            path.length + config.pixel_size,
            config.pixel_size,
        )
        line = path.interpolate(distances, as_array=True)
        frame_mask = coords_in_frame(line[..., :2], config.frame)   # 判断当前polygon，哪些点都在画布中
        indices_segments = find_consecutive_true_indices(frame_mask)    # 找出连续在画布内的部分

        for indices_segment in indices_segments:
            line_segment = line[indices_segment]
            if len(line_segment) < 3:
                continue
            lines_in_frame.append(line_segment) # 这个里面的点，都是画布中可见的

    # sort out nearest num_lines elements
    lines_distances = [np.linalg.norm(line[..., :2], axis=-1).min() for line in lines_in_frame]
    lines_in_frame = [lines_in_frame[idx] for idx in np.argsort(lines_distances)[:num_lines]]

    # 2. rasterize preprocessed lines
    pixel_height, pixel_width = config.pixel_frame
    raster_lines = np.zeros((pixel_height, pixel_width, 2), dtype=np.float32)
    for line in lines_in_frame:

        # encode orientation as color value
        dxy = np.concatenate([np.cos(line[..., 2, None]), np.sin(line[..., 2, None])], axis=-1)
        values = 0.5 * (dxy + 1)    # 将方向向量 从[-1, 1] 的区间映射到 [0, 1]
        pixel_coords = coords_to_pixel(line[..., :2], config.frame, config.pixel_size)  # 将物理坐标转为像素坐标
        pixel_mask = pixel_in_frame(pixel_coords, config.pixel_frame)   # 检查像素是否在图像内

        pixel_coords, values = pixel_coords[pixel_mask], values[pixel_mask]
        raster_lines[pixel_coords[..., 0], pixel_coords[..., 1]] = values   # 每个像素位置存储了线条在该位置的方向信息

        if config.line_dots_radius > 0:
            thickness = -1
            if len(values) > 1:

                # NOTE: OpenCV has origin on top-left corner
                cv2.circle(
                    raster_lines,
                    (pixel_coords[0, 1], pixel_coords[0, 0]),
                    radius=config.line_dots_radius,
                    color=values[0],
                    thickness=thickness,
                )
                cv2.circle(
                    raster_lines,
                    (pixel_coords[-1, 1], pixel_coords[-1, 0]),
                    radius=config.line_dots_radius,
                    color=values[-1],
                    thickness=thickness,
                )
    raster_lines = raster_lines.sum(-1, keepdims=True) > 0     # 我们输入encode时，只需要输入0, 1图
    return raster_lines.astype(np.float32).transpose((2, 0, 1))
