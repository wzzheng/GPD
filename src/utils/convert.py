from typing import List
import numpy as np
import torch
import numba

def matrix_from_pose(pose):
    """
    Converts a 2D pose to a 3x3 transformation matrix

    :param pose: 2D pose (x, y, yaw)
    :return: 3x3 transformation matrix
    """
    return torch.tensor(
        [
            [torch.cos(pose[2]), -torch.sin(pose[2]), pose[0]],
            [torch.sin(pose[2]), torch.cos(pose[2]), pose[1]],
            [0, 0, 1],
        ]
    )

def list_ego_matrix_from_pose(poses):
    """
    poses: [..., 3]
    """
    transposition_poses = torch.zeros(*poses.shape, 3)  # [..., 3, 3]

    transposition_poses[..., 0, 0] = torch.cos(poses[..., 2])
    transposition_poses[..., 0, 1] = -torch.sin(poses[..., 2])
    transposition_poses[..., 0, 2] = poses[..., 0]
    transposition_poses[..., 1, 0] = torch.sin(poses[..., 2])
    transposition_poses[..., 1, 1] = torch.cos(poses[..., 2])
    transposition_poses[..., 1, 2] = poses[..., 1]
    transposition_poses[..., 2, 2] = 1.0
    return transposition_poses  # [..., 3, 3]

@numba.njit
def np_list_ego_matrix_from_pose(poses):
    """
    poses: [..., 3]
    """
    transposition_poses = np.zeros((*poses.shape, 3))  # [..., 3, 3]

    transposition_poses[..., 0, 0] = np.cos(poses[..., 2])
    transposition_poses[..., 0, 1] = -np.sin(poses[..., 2])
    transposition_poses[..., 0, 2] = poses[..., 0]
    transposition_poses[..., 1, 0] = np.sin(poses[..., 2])
    transposition_poses[..., 1, 1] = np.cos(poses[..., 2])
    transposition_poses[..., 1, 2] = poses[..., 1]
    transposition_poses[..., 2, 2] = 1.0
    return transposition_poses  # [..., 3, 3]


def list_ego_matrix_from_pose(poses):
    """
    poses: [..., 3]
    """
    transposition_poses = torch.zeros((*poses.shape, 3))  # [..., 3, 3]

    transposition_poses[..., 0, 0] = torch.cos(poses[..., 2])
    transposition_poses[..., 0, 1] = -torch.sin(poses[..., 2])
    transposition_poses[..., 0, 2] = poses[..., 0]
    transposition_poses[..., 1, 0] = torch.sin(poses[..., 2])
    transposition_poses[..., 1, 1] = torch.cos(poses[..., 2])
    transposition_poses[..., 1, 2] = poses[..., 1]
    transposition_poses[..., 2, 2] = 1.0
    return transposition_poses  # [..., 3, 3]

@numba.njit
def np_list_ego_2_2_rotate_metric(poses):
    """
    poses: [..., 3]
    """
    transposition_poses = np.zeros((*poses.shape, 3))  # [..., 3, 3]

    transposition_poses[..., 0, 0] = np.cos(poses[..., 2])
    transposition_poses[..., 0, 1] = -np.sin(poses[..., 2])
    transposition_poses[..., 0, 2] = poses[..., 0]
    transposition_poses[..., 1, 0] = np.sin(poses[..., 2])
    transposition_poses[..., 1, 1] = np.cos(poses[..., 2])
    transposition_poses[..., 1, 2] = poses[..., 1]
    transposition_poses[..., 2, 2] = 1.0
    return transposition_poses  # [..., 3, 3]

@numba.njit
def np_list_ego_3_1_matrix_from_pose(poses):
    """
    poses: [bs, 1, M, 2]
    输出: [[x], [y], [1]]
    """
    bs, _, M, _ = poses.shape
    transposition_poses = np.zeros((bs, 1, M, 3, 1))  # [bs, 1, M, 3, 1]

    transposition_poses[..., 0, 0] = poses[..., 0]
    transposition_poses[..., 1, 0] = poses[..., 1]
    transposition_poses[..., 2, 0] = 1
    return transposition_poses  # [bs, 1, M, 3, 1]

def list_ego_3_1_matrix_from_pose(poses):
    """
    poses: [bs, T, M, 2]
    输出: [[x], [y], [1]]
    """
    bs, T, M, _ = poses.shape
    transposition_poses = torch.zeros((bs, T, M, 3, 1))  # [bs, 1, M, 3, 1]

    transposition_poses[..., 0, 0] = poses[..., 0]
    transposition_poses[..., 1, 0] = poses[..., 1]
    transposition_poses[..., 2, 0] = 1
    return transposition_poses  # [bs, 1, M, 3, 1]


def pose_from_matrix(transform_matrix):
    """
    Converts a 3x3 transformation matrix to a 2D pose
    :param transform_matrix: 3x3 transformation matrix
    :return: 2D pose (x, y, yaw)
    """
    if transform_matrix.shape != (3, 3):
        raise RuntimeError(f"Expected a 3x3 transformation matrix, got {transform_matrix.shape}")
    # Map a angle in range [-π, π]
    heading = torch.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

    return torch.tensor([transform_matrix[0, 2], transform_matrix[1, 2], heading])

def list_ego_pose_from_matrix(transform_matrix):
    # [..., 3, 3]
    heading = torch.atan2(transform_matrix[:, :, :, 1, 0], transform_matrix[:, :, :, 0, 0]) # [...]

    # 构造 re 张量
    re = torch.zeros(*heading.shape, 3) # [..., 3]
    re[:, :, :, 0] = transform_matrix[:, :, :, 0, 2]
    re[:, :, :, 1] = transform_matrix[:, :, :, 1, 2]
    re[:, :, :, 2] = heading

    return re

@numba.njit
def np_list_ego_pose_from_matrix(transform_matrix):
    # [..., 3, 3]
    heading = np.arctan2(transform_matrix[:, :, :, 1, 0], transform_matrix[:, :, :, 0, 0]) # [...]

    # 构造 re 张量
    re = np.zeros((*heading.shape, 3)) # [..., 3]
    re[:, :, :, 0] = transform_matrix[:, :, :, 0, 2]
    re[:, :, :, 1] = transform_matrix[:, :, :, 1, 2]
    re[:, :, :, 2] = heading

    return re


@numba.njit
def np_list_ego_2_2_rotate_matrix_from_pose(poses):
    """
    poses:  [bs, T, 1, 1]
    输出：[bs, T, 1, 2, 2]，仅仅包含角度的旋转矩阵
    """
    bs, T, _, _ = poses.shape
    transposition_poses = np.zeros((bs, T, 1, 2, 2))

    transposition_poses[..., 0, 0] = np.cos(poses[..., 0])
    transposition_poses[..., 0, 1] = -np.sin(poses[..., 0])
    transposition_poses[..., 1, 0] = np.sin(poses[..., 0])
    transposition_poses[..., 1, 1] = np.cos(poses[..., 0])
    return transposition_poses  # [bs, T, 1, 2, 2]


def absolute_to_relative_poses(origin_pose, absolute_poses):
    """
    Converts a list of poses from absolute to relative coordinates using an origin pose.
    :origin [3], x, y, head
    :param absolute_poses: list of absolute poses to convert, [A, 3], x, y, head
    :return: list of converted relative poses, list[A, 3]
    """
    absolute_transforms = torch.stack([matrix_from_pose(pose) for pose in absolute_poses], dim=0)
    origin_transform = matrix_from_pose(origin_pose)
    origin_transform = torch.linalg.inv(origin_transform)
    relative_transforms = origin_transform @ absolute_transforms
    relative_poses = torch.stack([pose_from_matrix(transform_matrix) for transform_matrix in relative_transforms], dim=0)

    return relative_poses.to(origin_pose)

def relative_to_absolute_poses(origin_pose, relative_poses):
    """
    Converts poses from relative to absolute coordinates using an origin pose.
    :param origin_pose: Reference origin pose
    :param relative_poses: list of relative poses to convert
    :return: list of converted absolute poses
    """
    relative_transforms = torch.stack([matrix_from_pose(pose) for pose in relative_poses])
    origin_transform = matrix_from_pose(origin_pose)
    absolute_transforms = origin_transform @ relative_transforms
    absolute_poses = torch.stack([pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms], dim=0)

    return absolute_poses.to(origin_pose)

def convert_absolute_to_relative_se2_array(origin, state_se2_array):
    """
    第一次使用的转化方法
    Converts an position from global to relative coordinates.
    :param origin: origin pose of relative coords system, [x, y, heading]
    :param state_se2_array: array of SE2 states with (x,y,θ) in last dim， shape: [points_num, 3]
    :return: SE2 coords array in relative coordinates, shape [points_num, 3]
    """
    def normalize_angle(angle):
        """
        Map a angle in range [-π, π]
        :param angle: any angle as float
        :return: normalized angle
        """
        return torch.arctan2(torch.sin(angle), torch.cos(angle))
    origin = origin.type(torch.float64)
    theta = -origin[2]
    origin_array = origin.unsqueeze(dim=0)  # [1, 3]

    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]], device=theta.device)

    points_rel = state_se2_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])
    return points_rel.type(torch.float32)

def list_ego_absolute_to_relative_poses(origin_pose, absolute_poses):
    """
    转化origin: [bs, T, 1, 3], poses: [bs, T, A, 3]或[bs, 1, M, 3]或[bs, T, 1, 3], 转置矩阵
    """
    absolute_transforms = list_ego_matrix_from_pose(absolute_poses) # [..., 3, 3]
    origin_transform = list_ego_matrix_from_pose(origin_pose)   # [bs, T, 1, 3, 3]
    origin_transform = torch.linalg.inv(origin_transform)
    relative_transforms = origin_transform @ absolute_transforms    # [bs, T, num, 3, 3]
    relative_poses = list_ego_pose_from_matrix(relative_transforms)

    return relative_poses.to(origin_pose)

def list_ego_relative_to_absolute_poses(origin_pose, relative_poses):
    """
    转化origin: [bs, T, 1, 3], pose: [bs, T, M*P, 3]
    """
    relative_transforms = list_ego_matrix_from_pose(relative_poses) # [..., 3, 3]
    origin_transform = list_ego_matrix_from_pose(origin_pose)   # [bs, T, 1, 3, 3]
    absolute_transforms = origin_transform @ relative_transforms    # [bs, T, Entity_num, 3, 3]
    absolute_poses = list_ego_pose_from_matrix(absolute_transforms)

    return absolute_poses.to(origin_pose)

# 已检查无误
def np_list_ego_absolute_to_relative_poses(origin_pose, absolute_poses):
    """
    nparray版本
    转化origin: [bs, T, 1, 3], poses: [bs, T, A, 3]或[bs, 1, M, 3]或[bs, T, 1, 3], 转置矩阵
    """
    absolute_transforms = np_list_ego_matrix_from_pose(absolute_poses) # [..., 3, 3]
    origin_transform = np_list_ego_matrix_from_pose(origin_pose)   # [bs, T, 1, 3, 3]
    origin_transform = np.linalg.inv(origin_transform)
    relative_transforms = origin_transform @ absolute_transforms    # [bs, T, num, 3, 3]
    relative_poses = np_list_ego_pose_from_matrix(relative_transforms)

    return relative_poses

# 已检查无误
def np_list_ego_relative_to_absolute_poses(origin_pose, relative_poses):
    """
    nparray版本
    转化origin: [bs, T, 1, 3], poses: [bs, T, A, 3]或[bs, 1, M, 3]或[bs, T, 1, 3], 转置矩阵
    """
    relative_transforms = np_list_ego_matrix_from_pose(relative_poses) # [..., 3, 3]
    origin_transform = np_list_ego_matrix_from_pose(origin_pose)   # [bs, T, 1, 3, 3]
    relative_transforms = origin_transform @ relative_transforms    # [bs, T, num, 3, 3]
    relative_poses = np_list_ego_pose_from_matrix(relative_transforms)

    return relative_poses

# 已查验无误，但转换回去的版本，不能直接取消逆，需要测试，和思考
def np_list_ego_absolute_to_relative_only_rotate(origin_pose, absolute_velocities):
    """
    nparray版本，转换速度，仅仅只需要旋转矩阵就可以了
    转化origin: [bs, T, 1, 3], 传入一个theta角就可以, velocities: [bs, T, A, 2]
    """
    absolute_transforms = absolute_velocities[..., None] # [bs, T, A, 2, 1]， 将速度转为[2, 1]的形式
    origin_transform = np_list_ego_2_2_rotate_matrix_from_pose(origin_pose[..., None, 2])   # [bs, T, 1, 2, 2]，注意这里输入的origin是[bs, T, 1, 1]
    origin_transform = np.linalg.inv(origin_transform)
    relative_transforms = origin_transform @ absolute_transforms    # [bs, T, A, 2, 1]
    return relative_transforms.squeeze(-1)    # [bs, T, A, 2]，x和y

# 已查验无误，转换回去的版本也无误
def np_list_ego_absolute_to_relative_poses_without_heading(origin_pose, absolute_poses):
    """
    nparray版本
    转化origin: [bs, T, 1, 3], poses: [bs, 1, M, 2]
    """
    absolute_transforms = np_list_ego_3_1_matrix_from_pose(absolute_poses) # [bs, 1, M, 3, 1]
    origin_transform = np_list_ego_matrix_from_pose(origin_pose)   # [bs, T, 1, 3, 3]
    origin_transform = np.linalg.inv(origin_transform)
    relative_transforms = origin_transform @ absolute_transforms    # [bs, T, num, 3, 1]

    bs, _, M, _ = absolute_poses.shape
    T = origin_pose.shape[1]
    # 构造 re 张量
    re = np.zeros((bs, T, M, 2)) # [..., 3]
    re[..., 0] = relative_transforms[..., 0, 0]
    re[..., 1] = relative_transforms[..., 1, 0]
    return re

# 已查验无误
def np_list_ego_relative_to_absolute_poses_without_heading(origin_pose, relative_poses):
    """
    nparray版本
    转化origin: [bs, T, 1, 3], poses: [bs, 1, M, 2]
    """
    absolute_transforms = np_list_ego_3_1_matrix_from_pose(relative_poses) # [bs, 1, M, 3, 1]
    origin_transform = np_list_ego_matrix_from_pose(origin_pose)   # [bs, T, 1, 3, 3]
    relative_transforms = origin_transform @ absolute_transforms    # [bs, T, num, 3, 1]

    bs, _, M, _ = relative_poses.shape
    T = origin_pose.shape[1]
    # 构造 re 张量
    re = np.zeros((bs, T, M, 2)) # [..., 3]
    re[..., 0] = relative_transforms[..., 0, 0]
    re[..., 1] = relative_transforms[..., 1, 0]
    return re

# 已查验无误
def list_ego_relative_to_absolute_poses_without_heading(origin_pose, relative_poses):
    """
    torch版本
    转化origin: [bs, T, 1, 3], poses: [bs, 1, M, 2]
    """
    absolute_transforms = list_ego_3_1_matrix_from_pose(relative_poses) # [bs, 1, M, 3, 1]
    origin_transform = list_ego_matrix_from_pose(origin_pose)   # [bs, T, 1, 3, 3]
    relative_transforms = origin_transform @ absolute_transforms    # [bs, T, num, 3, 1]

    bs, _, M, _ = relative_poses.shape
    T = origin_pose.shape[1]
    # 构造 re 张量
    re = torch.zeros((bs, T, M, 2)) # [..., 3]
    re[..., 0] = relative_transforms[..., 0, 0]
    re[..., 1] = relative_transforms[..., 1, 0]
    return re.to(origin_pose)

if __name__ == '__main__':
    # origin = torch.randn(3)
    # pose = torch.randn((5, 3))
    # re1 = absolute_to_relative_poses(origin, pose)
    # re2 = convert_absolute_to_relative_se2_array(origin, pose)  # 这两个是等价的
    # print(re1)

    # # 恢复验证
    # pose = torch.randn((2, 5, 3))   # [T, A, xyh]，以两个时刻为例，每个时刻的第0个A是ego
    # absolute_to_T_0 = absolute_to_relative_poses(pose[0, 0], pose[0])   # 将0时刻的几个agent转化到以0时刻自车为中心
    # absolute_to_T_1 = absolute_to_relative_poses(pose[1, 0], pose[1])
    # T_0_to_absolute = relative_to_absolute_poses(pose[0, 0], absolute_to_T_0)   # 等价于pose[0]
    # T_1_to_absolute = relative_to_absolute_poses(pose[1, 0], absolute_to_T_1)

    
    # T_1_agent_in_T_0 = absolute_to_relative_poses(pose[0, 0], pose[1])   # 将1时刻的agent也转化到以0时刻自车为中心的坐标
    # absolute_to_T_0_to_T_1 = absolute_to_relative_poses(T_1_agent_in_T_0[0], T_1_agent_in_T_0)  # 等价于absolute_to_T_1，都是以T_1时刻的ego car作为中心的坐标系
    # T_1_to_T_0 = relative_to_absolute_poses(T_1_agent_in_T_0[0], absolute_to_T_0_to_T_1) # absolute_to_T_0_to_T_1
    # T_1_to_T_0_to_absolute = relative_to_absolute_poses(pose[0, 0], T_1_to_T_0)
    # print(pose)
    # # 相当于，我们只需要记录一下1时刻的ego car，再0时刻ego坐标系下的x, y, h就可以还原回来了


    # 测试合并乘法是不是效果一样 bs, T, A, 3
    # stack_pose = []
    # pose = torch.randn((2, 3, 5, 3))
    # bs, T, A, _ = pose.shape
    # for b in range(bs): 
    #     for t in range(T):
    #         origin = pose[b, t, 0] # shape:[3], 内容：ego的x, y, head
    #         stack_pose.append(
    #             absolute_to_relative_poses(
    #                 origin_pose=origin,
    #                 absolute_poses=pose[b, t]
    #             )
    #         )
    # pose_tr1 = torch.stack(stack_pose, dim=0).view(bs, T, A, 3)

    # origin = pose[:, :, 0].unsqueeze(2)  # [bs, T, 1, 3]
    # pose_tr2 = list_ego_absolute_to_relative_poses(origin, pose)

    # print(pose_tr1)    
    # print("#" * 10)
    # print(pose_tr2)

    # # 测试[bs, M, 3]
    # stack_pose = []
    # origin_pose = torch.randn((2, 3, 1, 3))  # [bs, T, 1, 3]
    # map_point = torch.randn((2, 5, 3))
    # bs, T, A, _ = origin_pose.shape
    # M = map_point.shape[1]
    # for b in range(bs): 
    #     for t in range(T):
    #         origin = origin_pose[b, t, 0] # shape:[3], 内容：ego的x, y, head
    #         stack_pose.append(
    #             absolute_to_relative_poses(
    #                 origin_pose=origin,
    #                 absolute_poses=map_point[b]
    #             )
    #         )
    # pose_tr1 = torch.stack(stack_pose, dim=0).view(bs, T, M, 3)

    # origin = origin_pose  # [bs, T, 1, 3]
    # pose_tr2 = list_ego_absolute_to_relative_poses(origin, map_point.unsqueeze(1))  # [bs, 1, M ,3]

    # print(pose_tr1)    
    # print("#" * 10)
    # print(pose_tr2)

    # 测试转置矩阵
    # stack_pose = []
    # pose = torch.randn((2, 3, 1, 3))    # [bs, T, 1, 3]
    # origin_pose = torch.randn((2, 3, 1, 3))  # [bs, T, 1, 3]
    # bs, T, A, _ = pose.shape
    # for b in range(bs): 
    #     for t in range(T):
    #         origin = origin_pose[b, t, 0]
    #         if t == 0:
    #             # 对于0时刻，我们不需要记录其相对于绝对坐标的偏移
    #             stack_pose.append(torch.tensor([0, 0, 0]).unsqueeze(0))
    #             prior = absolute_to_relative_poses(origin, pose[b, 1, 0].unsqueeze(0))
    #         elif t + 1 < T:
    #             # 记录1时刻在0时刻自车为中心的坐标系下的坐标
    #             stack_pose.append(prior)
    #             # 处理下一时刻
    #             prior = absolute_to_relative_poses(origin, pose[b, t + 1, 0].unsqueeze(0))
    #         elif t + 1 == T:
    #             # 记录最后一个时刻
    #             stack_pose.append(prior)
    # pose_tr1 = torch.cat(stack_pose, dim=0).view(bs, T, 1, 3)

    # origin = origin_pose  # [bs, T, 1, 3]
    # pose_tr2 = list_ego_absolute_to_relative_poses(origin[:, :-1], pose[:, 1:])

    # print(pose_tr1.shape)    
    # print(pose_tr1)    
    # print("#" * 10)
    # print(pose_tr2)

    """
    测试：np_list_ego_absolute_to_relative_velocity
    """
    # origin = np.array([0, 0, np.pi / 6])    # 转90度
    # velocity = np.random.randn(22, 2) # [A, 2]
    # print(velocity)
    # print('原本')
    # # 正确的转换：
    # rotate_mat = np.array(
    #     [
    #         [np.cos(origin[2]), -np.sin(origin[2])],
    #         [np.sin(origin[2]), np.cos(origin[2])],
    #     ],
    #     dtype=np.float64,
    # )
    # true_ab_re = np.matmul(velocity, rotate_mat)    # [A, 2]
    # print(true_ab_re)
    # print('hhhh')
    # # 测试的转换
    # test_ab_re = np_list_ego_absolute_to_relative_velocity(origin[None, None, None, :], velocity[None, None, ...])  # [1, 1, A, 2]
    # print(test_ab_re[0, 0])
    # print('转换')

    """
    测试：np_list_ego_absolute_to_relative_poses_without_heading
    """
    # # origin = np.array([1, 1, np.pi / 2])   # 平移1
    # origin = np.array([12, 1.7, np.pi / 6])   # 平移1
    # poses = np.random.randn(15, 2) # [M, 2]
    # print(poses)
    # print('原本')
    # # 正确的转换：
    # center_xy, center_angle = origin[:2], origin[2] 

    # rotate_mat = np.array(
    #     [
    #         [np.cos(center_angle), -np.sin(center_angle)],
    #         [np.sin(center_angle), np.cos(center_angle)],
    #     ],
    #     dtype=np.float64,
    # )
    # true_ab_re = np.matmul(     # origin
    #         poses - center_xy, rotate_mat
    #     )
    # print(true_ab_re)
    # print('hhhh')
    # # 测试的转换
    # test_ab_re = np_list_ego_absolute_to_relative_poses_without_heading(origin[None, None, None, :], poses[None, None, ...])  # [1, 1, M, 2]
    # print(test_ab_re[0, 0])
    # print('转换')
    # # 转换回去测试
    # test_re_ab = np_list_ego_relative_to_absolute_poses_without_heading(origin[None, None, None, :], test_ab_re)  # [1, 1, M, 2]
    # print(test_re_ab[0, 0])
    # print('转换')    

    """
    测试：np_list_ego_absolute_to_relative_poses
    """
    origin = np.array([1, 1, np.pi / 2])   # 平移1
    # origin = np.array([12, 1.7, np.pi / 6])
    poses = np.random.randn(15, 3) # [A, 3]
    print(poses)
    print('原本')
    # 正确的转换：
    center_xy, center_angle = origin[:2], origin[2] 

    rotate_mat = np.array(
        [
            [np.cos(center_angle), -np.sin(center_angle)],
            [np.sin(center_angle), np.cos(center_angle)],
        ],
        dtype=np.float64,
    )
    true_ab_re_pose = np.matmul(     # origin
            poses[:, :2] - center_xy[None, ...], rotate_mat
        )
    true_ab_re_head = poses[:, 2] - center_angle
    print(np.concatenate([true_ab_re_pose, true_ab_re_head[:, None]], -1))
    print('hhhh')
    # 测试的转换
    test_ab_re = np_list_ego_absolute_to_relative_poses(origin[None, None, None, :], poses[None, None, ...])  # [1, 1, M, 2]
    print(test_ab_re[0, 0])
    print('转换')
    # 转换回去测试
    test_re_ab = np_list_ego_relative_to_absolute_poses(origin[None, None, None, :], test_ab_re)  # [1, 1, M, 2]
    print(test_re_ab[0, 0])
    print('转换')    