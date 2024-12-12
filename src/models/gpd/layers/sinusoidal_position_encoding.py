import torch
import torch.nn as nn
import math

def get_sinusoid_encoding_table(n_position:int, d_hid:int, base:float=10000.0, ntk_alpha:float=1.0, 
                                scaling_factor:float=1.0):
    ''' sinusoid编码
        
        :param n_position: int, 位置长度
        :param d_hid: int, 位置编码长度
        :param padding_idx: padding的token_ids
        :param ntk_alpha: int, 要扩展的倍数
        :param scaling_factor: int, chatglm中32k的插值
        :return: [seq_len, d_hid]
    '''
    if (ntk_alpha is not None) and (ntk_alpha != 1):
        base = base * ntk_alpha ** (d_hid / (d_hid-2))
    
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(base) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    if (scaling_factor is not None) and (scaling_factor != 1):
        position = position / scaling_factor
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table

class SinusoidalPositionEncoding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(self, max_position, embedding_size, freeze=True):
        super(SinusoidalPositionEncoding, self).__init__()
        self.position_embeddings = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(max_position, embedding_size), freeze=freeze)
         
    def forward(self, position_ids):
        return self.position_embeddings(position_ids)