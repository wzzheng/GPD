import torch.nn as nn
from ..planTF.layers.common_layers import build_mlp

class PlanningDecoder(nn.Module):
    def __init__(
        self, 
        d_model=128,
        future_steps=80,
        out_channels=4,
        is_decode_sincos=True
    ):
        super(PlanningDecoder, self).__init__()
        self.future_steps = future_steps
        if is_decode_sincos:
            self.out_channels = 4
        else:
            self.out_channels = 3
            
        self.agent_predictor = build_mlp(d_model, [d_model * 2, future_steps * 2], norm="ln")

        hidden = 2 * d_model

        self.ego_predictor = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * self.out_channels),
        )

    def forward(self, x):
        bs, A, _ = x.shape
        agent_position = self.agent_predictor(x[:, 1:])
        ego_position_heading = self.ego_predictor(x[:, 0])

        agent_position = agent_position.view(bs, A-1, self.future_steps, 2)
        ego_position_heading = ego_position_heading.view(bs, self.future_steps, self.out_channels)

        return agent_position, ego_position_heading