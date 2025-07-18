import torch
import torch.nn as nn
import torch.nn.functional as F

class LIDTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, d1, d2):
        super(LIDTCNBlock, self).__init__()
        self.conv_d1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * d1 // 2, dilation=d1)
        self.conv_d2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * d2 // 2, dilation=d2)
        self.conv_merge = nn.Conv1d(out_channels * 2, out_channels, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        out1 = self.conv_d1(x)
        out2 = self.conv_d2(x)
        out_cat = torch.cat([out1, out2], dim=1)
        out = self.conv_merge(out_cat)
        out = self.dropout(self.relu(out))
        return out + res