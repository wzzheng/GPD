from typing import Optional, Tuple
import cv2

import numpy as np
import numpy.typing as npt
import torch
import imageio

from nuplan.common.actor_state.oriented_box import OrientedBox

from .sledge_colors import Color, BLACK, WHITE, SLEDGE_ELEMENTS
from src.utils.sledge_utils.pdm_array_representation import array_to_state_se2
from src.utils.sledge_utils.utils import coords_to_pixel
from src.models.sledge.rvae_config import RVAEConfig
from src.bean.sledge_vector_feature import (
    SledgeConfig,
    SledgeVector,
    SledgeVectorElement,
    SledgeVectorElementType,
    BoundingBoxIndex,
)


def add_border_to_raster(
    image: npt.NDArray[np.uint8], border_size: int = 1, border_color: Color = BLACK
) -> npt.NDArray[np.uint8]:
    """
    Add boarder to numpy array / image.
    :param image: image as numpy array
    :param border_size: size of border in pixels, defaults to 1
    :param border_color: color of border, defaults to BLACK
    :return: image with border.
    """
    bordered_image = cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=border_color.rgb,
    )
    return bordered_image

def get_sledge_vector_as_raster(
    sledge_vector: SledgeVector, config=RVAEConfig()
) -> npt.NDArray[np.uint8]:
    """
    Convert sledge vector into RGB numpy array for visualization.
    :param sledge_vector: dataclass of vector representation
    :param config: config dataclass of sledge autoencoder
    :param map_id: map identifier to draw if provided, defaults to None
    :return: numpy RGB image
    """

    pixel_width, pixel_height = config.pixel_frame
    image: npt.NDArray[np.uint8] = np.full((pixel_width, pixel_height, 3), WHITE.rgb, dtype=np.uint8)
    draw_dict = {
        "L": {"elem": sledge_vector.lines, "color": SLEDGE_ELEMENTS["lines"], "count": 0},
        "V": {"elem": sledge_vector.vehicles, "color": SLEDGE_ELEMENTS["vehicles"], "count": 0},
        "P": {"elem": sledge_vector.pedestrians, "color": SLEDGE_ELEMENTS["pedestrians"], "count": 0},
        "S": {"elem": sledge_vector.static_objects, "color": SLEDGE_ELEMENTS["static_objects"], "count": 0},
        "G": {"elem": sledge_vector.green_lights, "color": SLEDGE_ELEMENTS["green_lights"], "count": 0},
        "R": {"elem": sledge_vector.red_lights, "color": SLEDGE_ELEMENTS["red_lights"], "count": 0},
    }

    for key, elem_dict in draw_dict.items():
        image, counter = draw_sledge_vector_element(image, elem_dict["elem"], config, elem_dict["color"])
        draw_dict[key]["count"] = counter

    # TODO: adapt to autoencoder config
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_height = 15
    width_offset = pixel_width - 39
    height_offset = pixel_height - 99

    for i, (key, elem_dict) in enumerate(draw_dict.items()):
        count = elem_dict["count"]
        cv2.putText(
            image,
            f"{key}={count}",
            (width_offset, height_offset + (i + 1) * line_height),
            font,
            font_scale,
            elem_dict["color"].rgb,
            thickness,
            cv2.LINE_AA,
        )

        draw_dict[key]["count"] = counter

    image = add_border_to_raster(image)
    return image


def draw_sledge_vector_element(
    image: npt.NDArray[np.uint8], sledge_vector_element: SledgeVectorElement, config: SledgeConfig, color: Color
) -> Tuple[npt.NDArray[np.uint8], int]:
    """
    Draws vector element on numpy RGB image.
    :param image: numpy RGB image
    :param sledge_vector_element: vector element to draw
    :param config: dataclass config of autoencoder in sledge
    :param color: color helper class
    :return: tuple of numpy RGB image and element count
    """

    element_counter = 0
    element_type = sledge_vector_element.get_element_type()
    element_index = sledge_vector_element.get_element_index()

    for states, p in zip(sledge_vector_element.states, sledge_vector_element.mask):
        draw_element = False
        if type(p) is np.bool_:
            draw_element = p
        else:
            draw_element = p > config.threshold
        if not draw_element:
            continue

        if element_type == SledgeVectorElementType.LINE:
            image = draw_line_element(image, states, config, color)
        else:
            image = draw_bounding_box_element(image, states, config, color, element_index)
        element_counter += 1

    return image, element_counter


def draw_line_element(
    image: npt.NDArray[np.uint8], state: npt.NDArray[np.float32], config: SledgeConfig, color: Color
) -> npt.NDArray[np.uint8]:
    """
    Draws a line state (eg. of lane or traffic light) onto numpy RGB image.
    :param image: numpy RGB image
    :param state: coordinate array of line
    :param config: dataclass config of autoencoder in sledge
    :param color: color helper class
    :return: numpy RGB image
    """
    assert state.shape[-1] == 2
    line_mask = np.zeros(config.pixel_frame, dtype=np.float32)
    indices = coords_to_pixel(state, config.frame, config.pixel_size)
    coords_x, coords_y = indices[..., 0], indices[..., 1]

    # NOTE: OpenCV has origin on top-left corner
    for point_1, point_2 in zip(zip(coords_x[:-1], coords_y[:-1]), zip(coords_x[1:], coords_y[1:])):
        cv2.line(line_mask, point_1, point_2, color=1.0, thickness=1)

    cv2.circle(line_mask, (coords_x[0], coords_y[0]), radius=3, color=1.0, thickness=-1)
    cv2.circle(line_mask, (coords_x[-1], coords_y[-1]), radius=3, color=1.0, thickness=-1)
    line_mask = np.rot90(line_mask)[:, ::-1]

    image[line_mask > 0] = color.rgb
    return image


def draw_bounding_box_element(
    image: npt.NDArray[np.uint8],
    state: npt.NDArray[np.float32],
    config: SledgeConfig,
    color: Color,
    object_indexing: BoundingBoxIndex,
) -> npt.NDArray[np.uint8]:
    """
    Draws a bounding box (eg. of vehicle) onto numpy RGB image.
    :param image: numpy RGB image
    :param state: state array of bounding box
    :param config: dataclass config of autoencoder in sledge
    :param color: color helper class
    :param object_indexing: index enum of state array
    :return: numpy RGB image
    """

    # Get the 2D coordinate of the detected agents.
    raster_oriented_box = OrientedBox(
        array_to_state_se2(state[object_indexing.STATE_SE2]),
        state[object_indexing.LENGTH],
        state[object_indexing.WIDTH],
        1.0,  # NOTE: dummy height
    )
    box_bottom_corners = raster_oriented_box.all_corners()
    corners = np.asarray([[corner.x, corner.y] for corner in box_bottom_corners])  # type: ignore
    corner_indices = coords_to_pixel(corners, config.frame, config.pixel_size)

    bounding_box_mask = np.zeros(config.pixel_frame, dtype=np.float32)
    cv2.fillPoly(bounding_box_mask, [corner_indices], color=1.0, lineType=cv2.LINE_AA)

    # NOTE: OpenCV has origin on top-left corner
    bounding_box_mask = np.rot90(bounding_box_mask)[:, ::-1]

    image[bounding_box_mask > 0] = color.rgb
    return image

def draw_decode_raster_img(raster, pixel_frame: Tuple[int, int], add_border: bool = True):
    pixel_width, pixel_height = pixel_frame
    image: npt.NDArray[np.uint8] = np.full((pixel_width, pixel_height, 3), WHITE.rgb, dtype=np.uint8)

    raster = raster.squeeze(0)
    segmentation_result = torch.where(raster > 0.5, torch.ones(256, 256, device=raster.device), torch.zeros_like(raster)).numpy()

    image[segmentation_result == 1.0] = SLEDGE_ELEMENTS["lines"].rgb
    image = image[::-1, ::-1]

    if add_border:
        image = add_border_to_raster(image)

    return image

def save_gif(images, output_path, duration):
    # 将图像转换为 RGB 格式，因为 OpenCV 使用 BGR 格式
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    imageio.mimsave(output_path, images_rgb, duration=duration)


def simple_visualize_vector_in_sledge(map_pred_line, map_pred_line_mask, map_label_line, map_label_line_mask, metas):
    map_pred_line = map_pred_line.transpose(0, 1)
    map_pred_line_mask = map_pred_line_mask.transpose(0, 1)
    map_label_line = map_label_line.transpose(0, 1)
    map_label_line_mask = map_label_line_mask.transpose(0, 1)
    
    _ = torch.full((2, 2, 2), -10.0, device=map_pred_line.device)    # 避免修改大量代码，注入一个占位tensor
    for pl_b, pm_b, ll_b, lm_b, meta in zip(map_pred_line, map_pred_line_mask, map_label_line, map_label_line_mask, metas):
        # 遍历bs级别
        output_raster = []
        for pl, pm, ll, lm in zip(pl_b, pm_b, ll_b, lm_b):
            # 遍历t级别，组合成一个gif
            imgs = []
            
            sledge_vector = SledgeVector(
                SledgeVectorElement(pl, pm),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
            )
            pred_sledge_vector_raster = get_sledge_vector_as_raster(
                sledge_vector=sledge_vector.torch_to_numpy(apply_sigmoid=True)
            )
            
            imgs.append(pred_sledge_vector_raster)

            gt_sledge_vector = SledgeVector(
                SledgeVectorElement(ll, lm),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
                SledgeVectorElement(_, _[0, 0]),
            )
            gt_sledge_vector_raster = get_sledge_vector_as_raster(
                sledge_vector=gt_sledge_vector.torch_to_numpy(apply_sigmoid=True)
            )
            imgs.append(gt_sledge_vector_raster)
            output_raster.append(np.concatenate(imgs, axis=0))
        save_gif(output_raster, meta._log_name + meta._scenario_type + meta._token + '.gif', duration=0.5)
