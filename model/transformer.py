import torch.nn as nn
import torch
from .attention import MultiHeadAttLayer
from .encoder import Encoder
from .decoder import Decoder
import torch.nn.functional as F
import copy
import math

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)

class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, num_head):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, num_head)
        self.decoders = nn.ModuleList([copy.deepcopy(
            Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, exponential_descrease(s), num_head)
        ) for s in range(num_decoders)])
        
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

