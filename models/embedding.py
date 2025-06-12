import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    """
    Embeds input sequence (S) and PSSM (S+) into latent space.
    Input shape: (batch_size, n_channels_in, seq_len)
    Output shape: (batch_size, nf, seq_len)
    """
    def __init__(self, in_channels: int, nf: int):
        super().__init__()
        self.embed = nn.Conv1d(in_channels, nf, kernel_size=1)

    def forward(self, x):
        return self.embed(x)


class OutputProjection(nn.Module):
    """
    Projects latent representation back to 3D coordinates.
    Input shape: (batch_size, nf, seq_len)
    Output shape: (batch_size, 3, seq_len)
    """
    def __init__(self, nf: int):
        super().__init__()
        self.project = nn.Conv1d(nf, 3, kernel_size=1)

    def forward(self, x):
        return self.project(x)


class ReverseEmbedding(nn.Module):
    """
    Projects coordinates back into latent space (reverse of OutputProjection).
    Input shape: (batch_size, 3, seq_len)
    Output shape: (batch_size, nf, seq_len)
    """
    def __init__(self, nf: int):
        super().__init__()
        self.embed = nn.Conv1d(3, nf, kernel_size=1)

    def forward(self, x):
        return self.embed(x)


class ReverseOutput(nn.Module):
    """
    Projects latent space back to one-hot + PSSM (S and S+).
    Output should ideally be (batch_size, 40, seq_len) => 20 amino acids + 20 PSSM
    """
    def __init__(self, nf: int, out_channels: int = 40):
        super().__init__()
        self.project = nn.Conv1d(nf, out_channels, kernel_size=1)

    def forward(self, x):
        return self.project(x)
