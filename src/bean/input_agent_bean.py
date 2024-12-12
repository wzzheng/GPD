from typing import Union
from torch import Tensor

import numpy as np

from dataclasses import dataclass
@dataclass
class InputAgentBean:

    def __init__(self, data: Union[Tensor, np.ndarray]):
        self.data = data

    @property
    def position(self):
        return self.data[..., :2]

    @property
    def heading(self):
        return self.data[..., 2]
    
    @property
    def valid_mask(self):
        return self.data[..., 3]
    
    # 这个只有计算metric用
    def get_position_x(self):
        return self.data[..., 0]
    
    def get_position_y(self):
        return self.data[..., 1]