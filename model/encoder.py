import torch
import torch.nn as nn
from .lidtcn import LIDTCNBlock
from .attention import MultiHeadAttLayer

class EncoderAttModule(nn.Module):
    """Attention Module for the Encoder using LIDTCN and Multi-Head Attention."""
    def __init__(self, d1, d2, in_channels, out_channels, r1, r2, att_type, stage, alpha, num_head):
        super(EncoderAttModule, self).__init__()
        self.feed_forward = LIDTCNBlock(in_channels, out_channels, 3, 0.5, d1, d2)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = MultiHeadAttLayer(in_channels, in_channels, out_channels, r1, r1, r2, d1, stage, att_type, num_head)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]
    
class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, num_head):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList()
        linear_growth_m = 6
        for i in range(num_layers):
            d1 = 2 ** i
            d2 = (i + 1) * linear_growth_m
            self.layers.append(
                EncoderAttModule(d1, d2, num_f_maps, num_f_maps, r1, r2, 'sliding_att', 'encoder', 1, num_head)
            )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature
