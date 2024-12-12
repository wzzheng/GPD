import time
from typing import Any, List, Dict, Optional, Callable
import torch
import numpy as np
from PIL import Image,ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import sys
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from src.utils.navism_visualization_utils import plot_bev_frame_with_traj, plot_bev_frame_with_traj_only_input_gt, plot_bev_frame_with_traj_only_black
from src.utils.convert import np_list_ego_absolute_to_relative_poses
from src.utils.simple_utils import normalize_angle
from src.utils.greedy_decode_utils import greedy_decode
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from src.models.gpd.planning_model import PlanningModel
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
import cv2

def concat_two_images(img1: Image, img2: Image) -> Image:
    width = img1.width + img2.width
    height = max(img1.height, img2.height)
    new_img = Image.new("RGBA", (width, height))

    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img

def concat_three_images(img1: Image, img2: Image, img3) -> Image:
    width = img1.width + img2.width + img3.width
    height = max(img1.height, img2.height, img3.height)
    new_img = Image.new("RGBA", (width, height))

    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.paste(img3, (img1.width + img2.width, 0))
    return new_img

def preprocess_for_lighting_test(agent_label, map_vector_label, output_window_T):

    agent_label = agent_label.transpose(2, 1).cpu().numpy() # [bs, T, A, _]

    map_vector_label = {
        'lines': map_vector_label['lines'].cpu().numpy(),
        'mask': map_vector_label['mask'].cpu().numpy()
    }

    return agent_label, map_vector_label
# def preprocess_for_lighting_test(agent_pred, agent_label, map_vector_pred, map_vector_label, output_window_T):
#     # 将数据整理为bs, ...的形式, 并且转为np
#     map_line_pred = map_vector_pred.lines.states.reshape(-1, output_window_T, 50, 20, 2)
#     map_mask_pred = torch.gt(map_vector_pred.lines.mask.reshape(-1, output_window_T, 50), 0)

#     map_vector_pred = {
#         'lines': map_line_pred.cpu().numpy(),
#         'mask': map_mask_pred.cpu().numpy()
#     }

#     agent_pred = agent_pred.transpose(2, 1).cpu().numpy()   # [bs, T, A, _]
#     agent_label = agent_label.transpose(2, 1).cpu().numpy() # [bs, T, A, _]

#     map_vector_label = {
#         'lines': map_vector_label['lines'].cpu().numpy(),
#         'mask': map_vector_label['mask'].cpu().numpy()
#     }

#     return agent_pred, agent_label, map_vector_pred, map_vector_label


# 可视化agent
# agent_synthesize_feature: [T, A, 9]，在一个绝对坐标系下，第21帧ego为中心的坐标系，或者世界坐标系都可以，heading用arctan表示
# trajetory_global: [T, 3], 在一个绝对坐标系下即可
def visualize(agent_feature, map_lines, map_masks, is_fixed_visual=False, is_multi_agent_traj=False, save_input_gt=True, save_black_pred=True):
    # agent和ego轨迹，要转化到每一帧自车为中心的坐标系
    agent_local_positions = np_list_ego_absolute_to_relative_poses(agent_feature[None, :, None, 0, :3], agent_feature[None, :, :, :3])[0]    # [T, A, 3]
    agent_masks = agent_feature[..., 3]    # [T, A, 4]
    ego_trajectory = agent_feature[:, 0, :3]    # [T, 3]
    ego_local_trajectorys = np_list_ego_absolute_to_relative_poses(ego_trajectory[None, :, None], ego_trajectory[None, None])[0]    # [T, T, 3]

    agent_trajectorys = None    # 记录每一帧agent的坐标
    if is_multi_agent_traj:
        agent_trajectorys = np_list_ego_absolute_to_relative_poses(agent_feature[None, :, 0, None, :3], agent_feature.transpose(1, 0, 2)[:, None, :, :3])   # [A, T, T, 3]
        # traj_mask = agent_masks.transpose(1, 0)[:, np.newaxis, :].repeat(80, 1).astype(np.bool)
        traj_mask = agent_masks.transpose(1, 0)[:, np.newaxis, :].repeat(21, 1).astype(np.bool)
        agent_trajectorys[~traj_mask] = 0    # 防止转换坐标后的0, 0的agent，变得有值
        agent_level_mask = agent_masks.any(0)    # [A]
        agent_trajectorys = agent_trajectorys[agent_level_mask]
    images: List[Image.Image] = []
    images_input_gt = []
    images_black_pred = []
    for i, (agent_position, agent_mask, ego_local_trajectory, map_line, map_mask) in enumerate(zip(
        agent_local_positions, agent_masks, ego_local_trajectorys, map_lines, map_masks
    )): # 遍历T
        annotations = []
        for idx, (pose, info) in enumerate(zip(agent_position, agent_mask)):
            if idx == 0 or int(info) == 0:    # filter ego or agent不存在
                continue
            annotations.append(
                {
                    "name_value": TrackedObjectType.VEHICLE,    # 我们只能可视化agent
                    "box_value":[pose[0], pose[1], pose[2], 5.0, 2.3, 1.0] # default length, weight, height=5, 2.3, 1.0, 
                }
            )

        fig, ax = plot_bev_frame_with_traj(
            map_line_feature=map_line[map_mask], 
            annotations=annotations, 
            # ego_local_trajectory=ego_local_trajectory, 
            ego_local_trajectory=[], 
            is_fixed_visual=is_fixed_visual,
            is_visual_multi_agent_traject=is_multi_agent_traj,
            idx=i,
            agent_trajectorys=agent_trajectorys
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        images.append(Image.open(buf).copy())

        # close buffer and figure
        buf.close()
        plt.close(fig)

        if save_input_gt:
            fig, ax = plot_bev_frame_with_traj_only_input_gt(
                map_line_feature=map_line[map_mask], 
                annotations=annotations, 
                ego_local_trajectory=ego_local_trajectory, 
                is_fixed_visual=is_fixed_visual,
                is_visual_multi_agent_traject=is_multi_agent_traj,
                idx=i,
                agent_trajectorys=agent_trajectorys
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            images_input_gt.append(Image.open(buf).copy())

            # close buffer and figure
            buf.close()
            plt.close(fig)

        if save_black_pred:
            fig, ax = plot_bev_frame_with_traj_only_black(
                map_line_feature=map_line[map_mask], 
                annotations=annotations, 
                ego_local_trajectory=ego_local_trajectory, 
                is_fixed_visual=is_fixed_visual,
                is_visual_multi_agent_traject=is_multi_agent_traj,
                idx=i,
                agent_trajectorys=agent_trajectorys
            )

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            images_black_pred.append(Image.open(buf).copy())

            # close buffer and figure
            buf.close()
            plt.close(fig)

    if save_input_gt:
        return images, images_input_gt, images_black_pred
    return images
    
def start_visualize_hook(future_80_frame_feature):
    # 这个函数写一些，因为版本不同，导致需要额外预处理的代码，尽可能不修改可视化代码主干
    
    # 对v8_glattn_globalpos_allmap，我们只生成了 x, y, heading, mask，对于其他agent，shap我们用ego自带的shape，category，都设为TrackedObjectType.VEHICLE
    # future_80_frame_feature: [T, A, 4]
    T, A, _ = future_80_frame_feature.shape
    return np.concatenate(  # [T, A, 9]
        [
            future_80_frame_feature[..., :3],
            np.full((T, A, 2), 0),      # 无所谓速度是多少
            np.full((T, A, 1), 2.3),    # shape x
            np.full((T, A, 1), 5),    # shape y
            np.full((T, A, 1), 1),    # category
            future_80_frame_feature[..., 3:4] > 0.5     # mask
        ], axis=-1
    )

def autoregressive_visualize(
    map_api: NuPlanMap,
    get_global_trajectory: Optional[Callable],
    ego_state,
    features: Dict[str, Any], 
    forward_fn: Callable,
    init_window_T: int = 21,
    pred_window_T: int = 80,
):
    pred_feature = greedy_decode(features, forward_fn, init_window_T, pred_window_T)
    future_80_frame_feature = torch.cat(    # [bs, A, T, 4]
        [
            pred_feature['agent']['position'],
            pred_feature['agent']['heading'].unsqueeze(-1),
            pred_feature['agent']['valid_mask'].unsqueeze(-1)
        ],
        dim=-1
    )
    future_80_frame_feature = future_80_frame_feature[0].transpose(0, 1).cpu().numpy()  # [T, A, 4]

    agent_feature_pred = start_visualize_hook(future_80_frame_feature)    # [T, A, 9]

    global_ego_trajectory_pred = get_global_trajectory(  # 从21帧ego car坐标，转化回绝对坐标
        future_80_frame_feature[:, 0, :3].astype(np.float64), ego_state
    )

    visualize(agent_feature_pred, global_ego_trajectory_pred, map_api, gen_gif=True, file_name=f"autoregressive_visual")
    sys.exit(0)

    
def simulation_visualize(
    map_api: NuPlanMap,
    future_80_frame_feature: np.ndarray, 
    global_ego_trajectory_pred: np.ndarray,
    is_visual_GT: bool = False,
    get_210T_features_from_scenario: Optional[Callable] = None,
    get_global_trajectory: Optional[Callable] = None,
    scenario: Optional[NuPlanScenario] = None,
    ego_state = None,
    current_idx = 21   
):
    """ 
    我们仅仅可视化预测的未来80帧
        future_80_frame_feature: 模型预测的未来80帧，都以20帧ego为中心的世界坐标系
        get_features_from_scenario: 如果需要可视化GT就必须传入，用于获取GT的feature
        get_global_trajectory: 如果需要可视化GT就必须传入，用于将21帧的ego轨迹，转化到全局坐标系
        current_idx： 用于打印，这是第几次迭代
    """
    if current_idx <= 100:
        print_idx = f"{current_idx}"
    else:
        print_idx = f"{current_idx}_{time.perf_counter()}"

    agent_feature_pred = start_visualize_hook(future_80_frame_feature)    # [T, A, 9]
    visualize(agent_feature_pred, global_ego_trajectory_pred, map_api, gen_gif=True, file_name=f"log_replay_{print_idx}")

    if is_visual_GT and current_idx==21:    # 仅仅只打印一次GT
        assert get_210T_features_from_scenario is not None
        planner_feature = get_210T_features_from_scenario(scenario)    # 读入未来21秒的GT
        
        GT_231T_frame_feature = _get_agent_synthesize_feature(planner_feature.data)    # [T, A, 9]

        global_ego_trajectory_GT = get_global_trajectory(  # 从21帧ego car坐标，转化回绝对坐标
            GT_231T_frame_feature[:, 0, :3].astype(np.float64), ego_state
        )

        # 分三次可视化未来帧，210/3
        # 第一次可视化21-91帧
        visualize(GT_231T_frame_feature[21: 91], global_ego_trajectory_GT[21: 91], map_api, gen_gif=True, file_name=f"GT_21-91--->idx: 0-70")
        # 第二次可视化91-161帧
        visualize(GT_231T_frame_feature[91: 161], global_ego_trajectory_GT[91: 161], map_api, gen_gif=True, file_name=f"GT_91-161--->idx: 70-140")
        # 第三次可视化161-231帧
        visualize(GT_231T_frame_feature[161: 231], global_ego_trajectory_GT[161: 231], map_api, gen_gif=True, file_name=f"GT_161-231--->idx: 140-210")
        print("GT可视化完成")
        # sys.exit(0)

    print(f"可视化完成: {print_idx}")

def concatenate_images_horizontally(images, gap=10):
    """
    将一组同样大小的图像水平拼接，中间使用指定宽度的白色填充间隔。

    参数:
    - images: list，包含 PIL 图像对象的列表
    - gap: int，图像之间的间隔宽度（以像素为单位）

    返回:
    - 拼接后的图像
    """
    # 检查是否有图片
    if not images:
        raise ValueError("图像列表为空")

    # 获取每张图片的宽高
    width, height = images[0].size

    # 计算新图像的总宽度
    total_width = width * len(images) + gap * (len(images) - 1)

    # 创建一个新的白色背景图像
    new_image = Image.new("RGB", (total_width, height), (255, 255, 255))

    # 将每张图片粘贴到新图像上
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += width + gap  # 移动到下一个图像的起点位置

    return new_image

def draw_text_on_image(image, text, position, font_path="/lpai/volumes/autopilot-end2end-lf/xzx/auto_regressive/autoregressive_planning/Times New Roman.ttf", font_size=20, color=(0, 0, 0)):
    """
    在指定位置上写入指定文字。

    参数:
    - image: PIL图像对象
    - text: 要添加的文本
    - position: tuple, 文本左上角的位置 (x, y)
    - font_path: 字体文件路径（例如 .ttf 文件）
    - font_size: 字体大小
    - color: 文字颜色，默认黑色 (0, 0, 0)

    返回:
    - 带文字的图像
    """
    # 创建可编辑的图像对象
    draw = ImageDraw.Draw(image)

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # 如果字体文件加载失败，使用默认字体
        font = ImageFont.load_default()
        print("无法加载指定字体，使用默认字体。")

    # 在指定位置绘制文本
    draw.text(position, text, fill=color, font=font)

    return image

def _get_agent_synthesize_feature(data):
    """说明：仅仅在可视化GT时使用
    1. 没有处理的agent信息中，mask=0表示看不到，但这些agent的position都是6万多
    2. 没有处理的agent信息中，agent.heading，没有归一化，存在4.5847这种超过3.14的值，也存在小于-3.14的值
    """
    position = data["agent"]["position"] 
    heading = normalize_angle(data["agent"]["heading"])

    velocity = data["agent"]["velocity"]  
    shape = data["agent"]["shape"]
    category = data["agent"]["category"]
    valid_mask = data["agent"]["valid_mask"][..., None]
    
    agent_feature = np.concatenate(  # [A, T, 9]
        [
            position,
            heading[..., None],
            velocity,
            shape,
            category[:, None, None].repeat(position.shape[1], 1),
            valid_mask,
        ],
        -1,
    )

    # 因为这个是label，所以也需要mask一次，他看不到值，x,y都是6万多
    agent_feature = np.where(valid_mask, agent_feature, np.zeros_like(agent_feature))
    return agent_feature.transpose(1, 0, 2)

def deal_img(img, idx, pos):
    img = img.crop((65, 60, 449, 444))
    new_img = Image.new("RGB", (384, 470), color=(255, 255, 255))
    new_img.paste(img, (0, 0))

    # 写字
    def format_number(x):
        """
        将数值转换为符合格式的字符串：
        - 十位数及以上（如 11.45）：保留一位小数，如 +11.5
        - 个位数（如 9.889）：保留两位小数，如 +9.89
        """
        if abs(x) >= 10:
            return f"{x:+.1f}"  # 保留一位小数
        else:
            return f"{x:+.2f}"  # 保留两位小数
    x, y = pos[idx, 0, :2] - pos[idx-9, 0, :2]
    draw_text_on_image(new_img, f"({format_number(y)}, {format_number(x)})", (2, 389), font_size=65)
    return new_img

def deal_gif_img(img, name):
    if name == "GT":
        img = img.crop((45, 10, 480, 475))
    elif name == "Input GT":
        img = img.crop((30, 10, 465, 475))
    else:
        img = img.crop((45, 10, 465, 475))

    # 写字
    if name == "Scene Generation":
        new_img = draw_text_on_image(img, name, (40, 0), font_size=50)
    elif name == "GT":
        new_img = draw_text_on_image(img, name, (177, 0), font_size=50)
    elif name == "Input GT":
        new_img = draw_text_on_image(img, name, (150, 0), font_size=40)
    elif name == "Traffic Simulation":
        new_img = draw_text_on_image(img, name, (23, 0), font_size=50)
    elif name == "Closed-Loop Simulation":
        new_img = draw_text_on_image(img, name, (15, 0), font_size=40)
    elif name == "Map Prediction":
        new_img = draw_text_on_image(img, name, (95, 0), font_size=40)
    elif name == "Motion Planning":
        new_img = draw_text_on_image(img, name, (85, 0), font_size=40)

    return new_img


def visualize_scene_in_navism(agent_label, map_vector_label, output_window_T, metas):
    agent_label, map_vector_label = preprocess_for_lighting_test(agent_label, map_vector_label, output_window_T)
    
    # 分bs处理
    for agent_l, map_line_l, map_mask_l, meta in zip(
        agent_label, map_vector_label['lines'], map_vector_label['mask'], metas
    ):
        images_label, images_input_gt, images_pred = visualize(agent_l, map_line_l, map_mask_l)

        for idx, (img_pred, img_label, img_input_gt) in enumerate(zip(images_pred, images_label, images_input_gt)):
            dealed_pred = deal_gif_img(img_pred, "Map Prediction")
            dealed_label = deal_gif_img(img_label, "GT")
            dealed_input_gt = deal_gif_img(img_input_gt, "Input GT")

            concat_img = concat_three_images(dealed_input_gt, dealed_pred, dealed_label)
            concat_img.save(meta._log_name + meta._scenario_type + meta._token + "_init_" + str(idx) + '.png')


# def visualize_scene_in_navism(agent_pred, agent_label, map_vector_pred, map_vector_label, output_window_T, metas, save_gif=True, save_columns_fig=False, save_input_gt=True):
#     agent_pred, agent_label, map_vector_pred, map_vector_label = preprocess_for_lighting_test(agent_pred, agent_label, map_vector_pred, map_vector_label, output_window_T)
    
#     # 分bs处理
#     for agent_p, agent_l, map_line_p, map_mask_p, map_line_l, map_mask_l, meta in zip(
#         agent_pred, agent_label, map_vector_pred['lines'], map_vector_pred['mask'], map_vector_label['lines'], map_vector_label['mask'], metas
#     ):
#         # 绘制label
#         if save_input_gt:
#             images_label, images_input_gt = visualize(agent_l, map_line_l, map_mask_l, save_input_gt=save_input_gt)
#         else:
#             images_label = visualize(agent_l, map_line_l, map_mask_l)
#         # 绘制pred
#         images_pred = visualize(agent_p, map_line_p, map_mask_p)

#         if save_gif:
#             if save_input_gt:
#                 for idx, (img_pred, img_label, img_input_gt) in enumerate(zip(images_pred, images_label, images_input_gt)):
#                     dealed_pred = deal_gif_img(img_pred, "Map Prediction")
#                     dealed_label = deal_gif_img(img_label, "GT")
#                     dealed_input_gt = deal_gif_img(img_input_gt, "Input GT")

#                     concat_img = concat_three_images(dealed_input_gt, dealed_pred, dealed_label)
#                     concat_img.save(meta._log_name + meta._scenario_type + meta._token + "_" + str(idx) + '.png')
#             else:
#                 images = []
#                 for idx, (img_pred, img_label) in enumerate(zip(images_pred, images_label)):
#                     dealed_pred = deal_gif_img(img_pred, "Scene Generation")
#                     dealed_label = deal_gif_img(img_label, "GT")
#                     concat_img = concat_two_images(dealed_pred, dealed_label)
#                     concat_img.save(meta._log_name + meta._scenario_type + meta._token + "_" + str(idx) + '.png')
#                     # concat_img = concat_two_images(img_pred, img_label)
#                     # images.append(concat_img)
#                 # images[0].save(meta._log_name + meta._scenario_type + meta._token + '.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

#         if save_columns_fig:
#             gt_imgs = []
#             pred_imgs = []
#             for idx, (img_pred, img_label) in enumerate(zip(images_pred, images_label)):
#                 if (idx + 1) % 10 == 0: 
#                     pred_imgs.append(deal_img(img_pred, idx, agent_p))
#                     gt_imgs.append(deal_img(img_label, idx, agent_l))
#                     # img_label.save(meta._log_name + meta._scenario_type + meta._token + f'GT_{idx}' +'.png')
#             pred_imgs = concatenate_images_horizontally(pred_imgs, gap=3)
#             gt_imgs = concatenate_images_horizontally(gt_imgs, gap=3)
#             pred_imgs.save(meta._log_name + meta._scenario_type + meta._token + '_imgs_concat_Pred' +'.png')
#             gt_imgs.save(meta._log_name + meta._scenario_type + meta._token + '_imgs_concat_GT' +'.png')