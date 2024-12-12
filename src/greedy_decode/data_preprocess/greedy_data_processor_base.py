from src.utils.simple_utils import mask_feature, agent_padding

class GreedyDataProcessorBase:
    """
        对于不同的模拟方法，需要不同的数据预处理方案，
        Base版本是:
            input   101 map agent ego，greedy自行处理
            output  80 map agent ego
    """

    def __call__(self, data, input_window_T, output_window_T, max_agent_num=33):
        agent_data = data['feature'].data['agent']
        
        agent_position = agent_data['position']
        agent_heading = agent_data['heading']
        agent_mask = agent_data['valid_mask']

        # 对agent进行一次掩码
        agent_position = mask_feature(agent_position, agent_mask)
        agent_heading = mask_feature(agent_heading, agent_mask)

        # padding agent to 33
        agent_position = agent_padding(agent_position, max_agent_num)
        agent_heading = agent_padding(agent_heading, max_agent_num)
        agent_mask = agent_padding(agent_mask, max_agent_num)

        # 组装train和label
        data_train, data_label = self.package_output(
            agent_position, agent_heading, agent_mask, 
            map_latent_data=data['rvae_latent'].data['encoding_indice'],
            map_info_data=data['map_cache'].data,
            input_window_T=input_window_T,
            output_window_T=output_window_T
        )
        return data_train, data_label
    
    @staticmethod
    def package_output(
        agent_position, agent_heading, agent_mask, 
        map_latent_data, map_info_data, 
        input_window_T, output_window_T
    ):

        # 组装train和label，给出全部T信息，greedy的时候自行裁剪
        data_train = {
            'agent': {
                'position': agent_position,
                'heading': agent_heading,
                'valid_mask': agent_mask
            },
            'map_latent': map_latent_data
        }

        end_T = input_window_T + output_window_T
        # data_label = {
        #     'agent': {
        #         'position': agent_position[:, :, input_window_T: end_T],
        #         'heading': agent_heading[:, :, input_window_T: end_T],
        #         'valid_mask': agent_mask[:, :, input_window_T: end_T]
        #     },
        #     'map_latent': map_latent_data[:, input_window_T: end_T],
        #     "map_vector": {
        #         'lines': map_info_data['lines'][:, input_window_T: end_T],
        #         'mask': map_info_data['masks'][:, input_window_T: end_T]
        #     }
        # }
        data_label = {
            'agent': {
                'position': agent_position[:, :, :input_window_T],
                'heading': agent_heading[:, :, :input_window_T],
                'valid_mask': agent_mask[:, :, :input_window_T]
            },
            'map_latent': map_latent_data[:, :input_window_T],
            "map_vector": {
                'lines': map_info_data['lines'][:, :input_window_T],
                'mask': map_info_data['masks'][:, :input_window_T]
            }
        }

        return data_train, data_label