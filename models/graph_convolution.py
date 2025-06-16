import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvBlock(nn.Module): # pojedynczy blok w naszych warstwach
    def __init__(self, in_channels, out_channels, coarse=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4)
        self.norm = nn.BatchNorm1d(out_channels)
        self.coarse = coarse

    # to jest prawie na pewno źle
    def forward(self, Y, L):
        Y_perm = Y.permute(0, 2, 1)
        YL = torch.bmm(L, Y_perm)
        YL = YL.permute(0, 2, 1)
        out = YL + Y
        out = self.conv(out) # (B, 128, L)
        out = self.norm(out)
        out = F.relu(out)
        out = out + self.coarse * Y
        return out
