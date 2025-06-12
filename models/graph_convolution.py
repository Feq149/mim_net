import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvBlock(nn.Module):#pojedynczy blok w naszych warstwach
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.adjoint = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, Y, L):
        #Wzór z pracy - konwolucja, przykładamy laplasjan, konwolucja.
        Y1 = self.conv(Y)
        Y1 = torch.bmm(Y1, L)  #bmm-batch matrix multiplication
        Y2 = F.relu(Y1)
        Y3 = self.adjoint(Y2)
        return -Y3
