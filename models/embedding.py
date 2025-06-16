import torch.nn as nn

# mapowanie S \cup S+ do nf
class InputEmbedding(nn.Module):
    def __init__(self, in_channels: int, nf: int):
        super().__init__()
        self.embed = nn.Conv1d(in_channels, nf, kernel_size=1)

    def forward(self, x):
        return self.embed(x)

# mapowanie nf do 3D
class OutputProjection(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        self.project = nn.Conv1d(nf, 3, kernel_size=1)

    def forward(self, x):
        return self.project(x)


# cofamy się z 3D do nf
class ReverseEmbedding(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        self.embed = nn.Conv1d(3, nf, kernel_size=1)

    def forward(self, x):
        return self.embed(x)

# wracamy do S \cup S+
class ReverseOutput(nn.Module):
    def __init__(self, nf: int, out_channels: int = 40):
        super().__init__()
        self.project = nn.Conv1d(nf, out_channels, kernel_size=1)

    def forward(self, x):
        return self.project(x)
