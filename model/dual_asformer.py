import torch
import torch.nn as nn
import torch.nn.functional as F
from .lidtcn import LIDTCNBlock
from .attention import MultiHeadAttLayer

class DualASFormer(nn.Module):
    def __init__(self, in_dim, num_classes, num_layers=10, num_f_maps=64, kernel_size=3, dropout=0.5,
                 att_heads=4, att_type='sliding_att', block_len=64):
        super(DualASFormer, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.input_proj = nn.Conv1d(in_dim, num_f_maps, 1)
        self.lidtcn_layers = nn.ModuleList([
            LIDTCNBlock(num_f_maps, num_f_maps, kernel_size, dropout, d1=1 << i, d2=3 << i) for i in range(num_layers)
        ])

        self.att_layers = nn.ModuleList([
            MultiHeadAttLayer(num_f_maps, num_f_maps, num_f_maps, r1=2, r2=2, r3=2,
                              bl=block_len, stage='encoder', att_type=att_type, num_head=att_heads)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        x = self.input_proj(x)

        for tcn_layer, att_layer in zip(self.lidtcn_layers, self.att_layers):
            x_tcn = tcn_layer(x)
            x_att = att_layer(x_tcn, None, mask)
            x = x + x_att

        out = self.decoder(x) * mask[:, 0:1, :]
        return out