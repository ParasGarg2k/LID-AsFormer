import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .attention import MultiHeadAttLayer
from .encoder import Encoder
from .decoder import Decoder
    
class ConvFeedForward(nn.Module):
    """Feed-forward block with a single dilated convolution, used by the Decoder."""
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer(x)