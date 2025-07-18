import torch
import torch.nn as nn
from .lidtcn import LIDTCNBlock
from .attention import MultiHeadAttLayer
from .basic import ConvFeedForward

class AttModule(nn.Module):
    """Original Attention Module (for Decoder) now upgraded with Multi-Head Attention."""
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, num_head):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = MultiHeadAttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, stage, att_type, num_head)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, alpha, num_head):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, 'sliding_att', 'decoder', alpha, num_head) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.gate_W1 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.gate_W2 = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature_layer = layer(feature, fencoder, mask)
            gate = self.sigmoid(self.gate_W1(feature) + self.gate_W2(fencoder))
            feature = gate * fencoder + (1 - gate) * feature_layer
        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature