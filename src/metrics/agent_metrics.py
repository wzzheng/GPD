import numpy as np

def average_displacement_error(pred, target, mask):
    # pred & target: [..., T, 2]
    # mask: [..., T]
    ade = np.linalg.norm(pred - target, ord=2, axis=-1) * mask

    # 注意，需要对T维度求平均，见plantf: ADE
    # return ade.mean(-1).sum()
    return ade.sum(0)   # [T]

def final_displacement_error(pred, target, mask):
    # pred & target: [..., T, 2]
    # mask: [..., T]
    # TODO ，看下，全0的agent是怎么处理的
    # 1. 找出每一个agent，最后一个mask=1的时刻，进行计算
    last_one_indices = np.argmax(mask[:, ::-1], axis=-1)
    last_one_indices = mask.shape[-1] - 1 - last_one_indices

    # 2. 从data中提取，最后一个mask=1的时刻, [bs (*A), 1, 2]

    last_visible_position_pred = np.take_along_axis(pred, last_one_indices[:, None, None], axis=1).squeeze(1)
    last_visible_position_target = np.take_along_axis(target, last_one_indices[:, None, None], axis=1).squeeze(1)
    last_visible_position_mask = np.take_along_axis(mask, last_one_indices[:, None], axis=1).squeeze(1)

    fde = np.linalg.norm(
        last_visible_position_pred - last_visible_position_target, 
        ord=2, 
        axis=-1
    ) * last_visible_position_mask

    return fde.sum()

def final_displacement_error_every_epoch(pred, target, mask):
    # pred & target: [..., T, 2]
    # mask: [..., T]
    T = pred.shape[1]
    fde = []
    for t in range(1, T + 1):
        fde.append(final_displacement_error(
            pred=pred[:, :t],
            target=target[:, :t],
            mask=mask[:, :t]
        ))
    return np.array(fde)


# nuplan 定义的
def collision_by_radius_check(positions: np.ndarray, radius_threshold: float):
    # positions [bs, A, T, 2]
    distences = np.linalg.norm(
        positions[:, :, np.newaxis] - positions[:, np.newaxis],
        axis=-1
    )

    collisions = (distences < radius_threshold) & (distences > 0)   # 避免和自己计算，也过滤看不到的点

    return collisions.any(axis=(-2))    # [bs, A]   有碰撞出现的agent，就被记为1

import numpy as np

def get_bounding_box(x, y, heading, length=4.084, width=1.85):
    """
    计算每辆车的边界框四个顶点。

    参数：
    - x, y: 车辆位置
    - heading: 车辆朝向（以弧度为单位）
    - length: 车辆长度
    - width: 车辆宽度

    返回：
    - bbox: 边界框的四个顶点坐标 [4, 2]
    """
    # 车辆中心到边界框顶点的相对坐标
    dx = length / 2
    dy = width / 2
    corners = np.array([
        [-dx, -dy],
        [-dx, dy],
        [dx, dy],
        [dx, -dy]
    ])  # [4, 2]

    # 旋转矩阵
    cos_theta = np.cos(heading)
    sin_theta = np.sin(heading)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # 计算旋转后的顶点坐标
    bbox = corners @ rotation_matrix.T  # [4, 2]
    bbox += np.array([x, y])  # 平移到车辆位置
    return bbox  # [4, 2]

def check_bbox_overlap(bbox1, bbox2):
    """
    判断两个边界框是否发生碰撞（重叠）
    使用了分离轴定理（Separating Axis Theorem）进行检测。

    参数：
    - bbox1: 第一个边界框顶点坐标 [4, 2]
    - bbox2: 第二个边界框顶点坐标 [4, 2]

    返回：
    - overlap: 布尔值，表示是否碰撞
    """
    def separating_axis_theorem(bbox1, bbox2):
        # 检查分离轴是否存在
        axes = np.array([bbox1[1] - bbox1[0], bbox1[2] - bbox1[1], bbox2[1] - bbox2[0], bbox2[2] - bbox2[1]])
        axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)

        for axis in axes:
            # 投影 bbox1 和 bbox2 到当前轴
            projections1 = bbox1 @ axis
            projections2 = bbox2 @ axis

            # 获得投影的最小值和最大值
            min1, max1 = projections1.min(), projections1.max()
            min2, max2 = projections2.min(), projections2.max()

            # 如果区间不重叠，则存在分离轴
            if max1 < min2 or max2 < min1:
                return False
        return True

    return separating_axis_theorem(bbox1, bbox2)

def check_collisions(agent_coords, mask, car_length=4.084, car_width=1.85):
    bs, A, T, _ = agent_coords.shape
    collisions = np.zeros((bs, A, T), dtype=bool)  # 初始化碰撞矩阵

    for t in range(T):
        # 获取每个时间步的车辆坐标和朝向
        coords_t = agent_coords[:, :, t, :2]  # [bs, A, 2]
        headings_t = agent_coords[:, :, t, 2]  # [bs, A]

        for batch in range(bs):
            # 使用 mask 过滤不可见车辆
            visible_indices = np.where(mask[batch, :, t] == 1)[0]  # 获取 mask 为 1 的索引

            # 计算每辆车的边界框
            bboxes = [get_bounding_box(coords_t[batch, i, 0], coords_t[batch, i, 1], headings_t[batch, i], car_length, car_width)
                      for i in visible_indices]

            # 检查可见车辆之间的碰撞情况
            for i, idx1 in enumerate(visible_indices):
                for j, idx2 in enumerate(visible_indices[i + 1:], i + 1):
                    bbox1 = bboxes[i]
                    bbox2 = bboxes[j]

                    # 判断是否重叠
                    overlaps = check_bbox_overlap(bbox1, bbox2)

                    # 如果重叠，则标记碰撞
                    collisions[batch, idx1, t] |= overlaps
                    collisions[batch, idx2, t] |= overlaps

    return collisions