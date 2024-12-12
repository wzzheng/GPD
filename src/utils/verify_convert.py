import copy
import torch
from src.utils.simple_utils import rotate_matrix_from_pose

"""
只是用于验证，不与其他py文件关联
"""

for i in range(10000):  # 随机测试10000次
    # origin = torch.randn(3) # [3]
    # poses = torch.randn(15, 3) # [A, 3]
    origin = torch.tensor([[[1, 1, torch.pi / 2], [0, 0, 0]], 
                            [[0, 0, -torch.pi / 2], [1, 1, 0]]])   # [2, 2, 3] -> [bs, T, 3]
    poses = torch.randn(2, 2, 15, 3) # [bs, T, A, 3]
    true_ab_re = copy.deepcopy(poses)
    bs, T, A, _ = poses.shape

    # absolute->relative
    relative_rotate_matrix = rotate_matrix_from_pose(origin[..., 2])    # [bs, T, 2, 2]
    relative_pose = torch.matmul(
        poses[..., :2] - origin[..., None, :2], relative_rotate_matrix
    )
    relative_heading = poses[..., 2] - origin[..., None, 2]
    
    # relative->absolute
    absolute_rotate_matrix = rotate_matrix_from_pose(- origin[..., 2])  # 这里加一个负号就是取逆了
    absolute_pose = torch.matmul(
        relative_pose, absolute_rotate_matrix
    ) + origin[..., None, :2]
    absolute_heading = relative_heading + origin[..., None, 2]
    
    
    diff_heading = torch.max(torch.abs(poses[..., 2] - absolute_heading))
    diff_pose = torch.max(torch.abs(poses[..., :2] - absolute_pose))
    difference_heading = difference_heading if difference_heading >= diff_heading else diff_heading
    difference_pose = difference_pose if difference_pose >= diff_pose else diff_pose
    
print(difference_heading)
print(difference_pose)


"""
    存档，验证相对--->绝对，覆盖test_step即可用
"""
def test_step(
    self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
) -> None:
    """
    Step called for each batch example during testing.
    为了尽可能和源码解耦，测试代码写在这里，后续定稿了可以拆分不同函数

    :param batch: example batch
    :param batch_idx: batch's index (unused)
    :return: model's loss tensor
    """
    # 验证绝对---相对，绝对的第21帧，应该和相对的第20帧值一样
    def _get_max_index(tensor):
        # 获取展平的最大值索引
        flat_index = torch.argmax(tensor).item()

        # 计算多维索引
        multi_dim_index = []
        for dim in reversed(tensor.shape):
            multi_dim_index.append(flat_index % dim)
            flat_index = flat_index // dim

        multi_dim_index = tuple(reversed(multi_dim_index))
        return multi_dim_index
    
    # 验证相对---绝对：先验证单个的转换，再验证全部数据的转换是否成功
    data = batch[0]['feature'].data 
    raw_GT_data = copy.deepcopy(data)   # debug
    raw_GT_data['agent']['heading'] = normalize_angle(raw_GT_data['agent']['heading'])
    # mask
    raw_GT_data['agent']['heading'] = mask_feature(raw_GT_data['agent']['heading'], raw_GT_data['agent']['valid_mask'])
    raw_GT_data['agent']['position'] = mask_feature(raw_GT_data['agent']['position'], raw_GT_data['agent']['valid_mask'])
    
    
    GT_data = copy.deepcopy(data)   # debug
    for key in ['position', 'heading', 'velocity', 'shape', 'valid_mask']:  # 整合absolute_agent_GT的一些预处理
        GT_data['agent'][key] = GT_data['agent'][key].transpose(1, 2)        
    GT_data['agent']['position'] = GT_data['agent']['position'][:, 1:]  # 第0帧被扔掉了
    GT_data['agent']['heading'] = GT_data['agent']['heading'][:, 1:]
    GT_data['agent']['velocity'] = GT_data['agent']['velocity'][:, 1:]
    
    # HACK 需要导入A_bounday
    absolute_agent_GT = PlanningModel.get_agent_synthesize_feature(GT_data, A_boundary=33)    # [bs, T, A_boundary, 10]，这个函数已经mask和padding过了，不需要再mask
    absolute_agent_GT[..., 2] = normalize_angle(absolute_agent_GT[..., 2])  # GT数据里的角度不在-pi, pi

    relative_data = PlanningModel.data_preprocess(data)
    def convert_relative_to_absolute(relative_data: Tensor, origin_pose: Tensor) -> None:
        """
            relative_data.data.shape = [bs, A, dim]，直接修改原始数据，后续不再使用，所以不需要deepcopy
            oringin.shape = [bs, 3], 因为需要循环串行的转化每一帧
        """
        assert len(origin_pose.shape) == 2
        
        origin = origin_pose.clone().unsqueeze(1)       # [bs, 1, 3]
        relative_data = InputAgentBean(relative_data)       # TODO 这里现阶段，relative_data只有三维，是可以用InputAgentBean，后续扩充的时候需要注意
        origin_position = origin[..., :2]           # [bs, 1, 2]
        origin_heading = origin[..., 2:3]           # [bs, 1, 1]
        absolute_rotate_matrix = rotate_matrix_from_pose(- origin_heading[:, 0, 0])    # [bs, 2, 2], 这里加一个负号就是取逆了
        
        # 处理agent
        relative_data.position = torch.matmul(   # [bs, A, 2]
            relative_data.position, absolute_rotate_matrix
        ) + origin_position
        relative_data.heading = normalize_angle(relative_data.heading + origin_heading)  # [bs, A, 1]        
    
    # 将数据还原到GT状态下，以第21帧自车为中心的绝对坐标系下，在这里的矩阵数字应该是19，后续用矩阵数字
    absolute_data = {}
    absolute_data["agent_synthesize_feature"] = relative_data["agent_synthesize_feature"][:, 20:].clone()
    bs, T, A, _ = absolute_data["agent_synthesize_feature"].shape 
    
    # ego_origin = absolute_data["agent_synthesize_feature"][:, :, 0:1, :3].clone()  # [bs, T, 1, 3]，表示，最后换算完表示，第T帧的ego在19帧ego为中心的坐标系下的绝对坐标，这里多一维，为了满足convert_relative_to_absolute函数
    for t in range(1, T):   
        # 预测的第0帧，也就是全局第20帧，本身就在19帧坐标系下，不需要转化
        # 预测的第1帧，也就是全局第21帧，本身在20帧为中心的坐标系下，因此我们只要用第20帧ego在第19帧坐标系下的坐标对其进行转化就可以将21帧的所有点转化到19帧坐标系下
        # 预测的第2帧，也就是全局第22帧，本身在21帧为中心的坐标系下，因此我们只要用第21帧ego在第19帧坐标系下的坐标对其进行转化就可以将22帧的所有点转化到19帧坐标系下
        # convert_relative_to_absolute(ego_origin[:, t : t+1], ego_origin[:, t-1])  # 仅转化ego
        convert_relative_to_absolute(
            absolute_data["agent_synthesize_feature"][:, t, :, :3],   # [bs, A, 3]
            absolute_data["agent_synthesize_feature"][:, t - 1, 0, :3]      # [bs, 3]
        )
    # 对转换完的absolute_data，还需要mask一次，因为对于哪些看不到的点，我们将其置为0，但坐标变换完就不一定是0了，会影响误差，因此还要mask一次
    absolute_data["agent_synthesize_feature"] = mask_feature(absolute_data["agent_synthesize_feature"], absolute_data["agent_synthesize_feature"][..., -1])
    print(torch.max(torch.abs(absolute_agent_GT[:, 20:, :, :2] - absolute_data["agent_synthesize_feature"][:, :, :, :2])))
    print(torch.max(torch.abs(absolute_agent_GT[:, 20:, :, 2] - absolute_data["agent_synthesize_feature"][:, :, :, 2])))
    print(torch.max(torch.abs(raw_GT_data['agent']['position'].transpose(1, 2)[:, 21:, :] - absolute_data["agent_synthesize_feature"][:, :, :, :2])))
    print(torch.max(torch.abs(raw_GT_data['agent']['heading'].transpose(1, 2)[:, 21:, :] - absolute_data["agent_synthesize_feature"][:, :, :, 2])))
