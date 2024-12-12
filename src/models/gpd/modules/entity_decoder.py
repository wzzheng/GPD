import torch.nn as nn

class EntityDecoder(nn.Module):
    def __init__(
        self, 
        d_model=128,
        code_book_size=128
    ):
        super(EntityDecoder, self).__init__()
        self.d_model = d_model
        
        # agent解码
        hidden = 2 * d_model
        self.agent_position_decoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.agent_heading_decoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.agent_valid_mask_decoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),  
            nn.Linear(hidden, 1),  # 二分类
        )
        self.map_latent_decoder = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),  
            nn.Linear(hidden, code_book_size),
        )

    def forward(self, x_agent, x_map):

        agent_position = self.agent_position_decoder(x_agent)
        agent_heading = self.agent_heading_decoder(x_agent)
        agent_valid_mask = self.agent_valid_mask_decoder(x_agent)

        map_latent = self.map_latent_decoder(x_map)

        return {
            'agent': {
                'position': agent_position,
                'heading': agent_heading,
                'valid_mask': agent_valid_mask
            },
            'map_latent': map_latent
        }